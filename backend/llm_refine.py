"""
LLM-backed refinement of OCR'd reaction conditions.

Uses Anthropic Claude (Haiku 4.5) via OpenRouter's OpenAI-compatible API
since this VPS only has OPENROUTER_API_KEY (no standalone ANTHROPIC_API_KEY).
The model is vision-capable: we hand it the cropped reaction image, the raw
EasyOCR strings, and the detected reactant/product SMILES so it can correct
common OCR mistakes (HCl read as HCI, lowercase l <-> 1 <-> I, dropped
subscripts H2O -> HO, etc.) using full reaction context.

Module is import-safe even when OPENROUTER_API_KEY is unset; in that case
LLM_ENABLED is False and refine_with_claude() falls back to identity-pass.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
import re
import time
from collections import OrderedDict
from typing import Any

log = logging.getLogger("rxnim.llm_refine")

# ---------- Config -------------------------------------------------------
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
# Primary; if not available the OpenRouter call surfaces a 404 which we let
# bubble to the caller's fallback handler.
OPENROUTER_MODEL = os.environ.get("OPENROUTER_REFINE_MODEL", "anthropic/claude-haiku-4-5")
LLM_TIMEOUT_S = float(os.environ.get("LLM_REFINE_TIMEOUT_S", "20.0"))
LLM_MAX_TOKENS = int(os.environ.get("LLM_REFINE_MAX_TOKENS", "1024"))

# Token pricing (USD per million tokens) — Haiku 4.5.  These are coarse
# estimates used only to populate the response metadata; OpenRouter also
# returns its own `cost` field in usage which we prefer when present.
PRICING_PER_MTOK = {
    "anthropic/claude-haiku-4-5": {"in": 1.0, "out": 5.0},
    "anthropic/claude-3.5-haiku": {"in": 1.0, "out": 5.0},
}

LLM_ENABLED: bool = bool(os.environ.get("OPENROUTER_API_KEY"))

# Lazy openai client singleton (imports gated so the module imports cleanly
# even before pip install completes during a cold build).
_client = None


def _get_client():
    global _client
    if _client is not None:
        return _client
    if not LLM_ENABLED:
        return None
    try:
        from openai import AsyncOpenAI
    except ImportError as e:  # pragma: no cover
        log.error("openai SDK not installed: %s", e)
        return None
    _client = AsyncOpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url=OPENROUTER_BASE_URL,
        timeout=LLM_TIMEOUT_S,
        # Set OpenRouter referrer headers so dashboard usage attribution works
        default_headers={
            "HTTP-Referer": "https://rxnim.sebland.com",
            "X-Title": "rxnim",
        },
    )
    return _client


# ---------- Cache --------------------------------------------------------
# Simple in-memory LRU + TTL cache.  Cap at CACHE_MAX entries.  TTL prevents
# stale entries from accumulating forever in long-running containers.
CACHE_MAX = int(os.environ.get("LLM_REFINE_CACHE_MAX", "500"))
CACHE_TTL_S = int(os.environ.get("LLM_REFINE_CACHE_TTL_S", str(24 * 60 * 60)))  # 24h
_cache: "OrderedDict[str, tuple[float, dict]]" = OrderedDict()
_cache_lock = asyncio.Lock()


def _cache_key(image_bytes: bytes, raw_ocr: list[str], smiles: dict) -> str:
    h = hashlib.sha256()
    h.update(image_bytes)
    h.update(b"|")
    for s in raw_ocr:
        h.update(s.encode("utf-8", errors="replace"))
        h.update(b"\x1f")
    h.update(b"|")
    h.update(json.dumps(smiles, sort_keys=True, default=str).encode("utf-8"))
    return h.hexdigest()


async def _cache_get(key: str) -> dict | None:
    async with _cache_lock:
        ent = _cache.get(key)
        if ent is None:
            return None
        ts, payload = ent
        if time.time() - ts > CACHE_TTL_S:
            _cache.pop(key, None)
            return None
        # Touch LRU
        _cache.move_to_end(key)
        return payload


async def _cache_put(key: str, payload: dict) -> None:
    async with _cache_lock:
        _cache[key] = (time.time(), payload)
        _cache.move_to_end(key)
        while len(_cache) > CACHE_MAX:
            _cache.popitem(last=False)


# ---------- Prompt -------------------------------------------------------
SYSTEM_PROMPT = (
    "You verify OCR output from a chemistry reaction-scheme image. Given the "
    "cropped image, raw OCR text guess for the conditions/reagents/catalysts, "
    "and the detected reactant/product SMILES, return a clean JSON object with "
    "corrected condition strings as plain chemistry text. Common OCR mistakes: "
    "lowercase 'l' read as 'I' (HCl->HCI), '1' read as 'l' or 'I', '0' read as "
    "'O', subscripts dropped (H2O->HO, NO2->NO). Use the image and SMILES "
    "context to infer correct text. Return ONLY a JSON object: "
    '{"conditions": ["str", ...], "notes": "str or null"}. No markdown fences. '
    'Keep "notes" empty unless something genuinely ambiguous remains.'
)


def _build_user_text(raw_ocr: list[str], smiles: dict) -> str:
    lines = ["Raw OCR conditions (may have errors):"]
    if raw_ocr:
        for s in raw_ocr:
            lines.append(f"- {s}")
    else:
        lines.append("- (none)")
    reactants = smiles.get("reactants") or []
    products = smiles.get("products") or []
    lines.append(f"Reactants SMILES: {reactants}")
    lines.append(f"Products SMILES: {products}")
    lines.append("Return cleaned conditions as JSON.")
    return "\n".join(lines)


# ---------- Tolerant JSON parsing ----------------------------------------
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)


def _parse_json_loose(text: str) -> dict:
    """Strip code fences / preambles and parse.  Raises ValueError on fail."""
    if not text:
        raise ValueError("empty response")
    cleaned = text.strip()
    cleaned = _FENCE_RE.sub("", cleaned).strip()
    # If model added prose, grab the first {...} block.
    if not cleaned.startswith("{"):
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not m:
            raise ValueError(f"no JSON object found in: {text!r}")
        cleaned = m.group(0)
    obj = json.loads(cleaned)
    if not isinstance(obj, dict):
        raise ValueError(f"top-level JSON not object: {type(obj).__name__}")
    return obj


# ---------- Cost compute -------------------------------------------------
def _compute_cost(model: str, usage: Any) -> float | None:
    """Prefer OpenRouter's own `cost` if the SDK surfaces it; else estimate."""
    if usage is None:
        return None
    # OpenRouter sets `cost` on the usage object via `usage.cost`.  The
    # openai SDK keeps unknown fields on the object — try both attribute and
    # dict access.
    for accessor in (
        lambda u: getattr(u, "cost", None),
        lambda u: u.get("cost") if hasattr(u, "get") else None,
        lambda u: u.model_extra.get("cost") if hasattr(u, "model_extra") and u.model_extra else None,
    ):
        try:
            v = accessor(usage)
            if v is not None:
                return float(v)
        except Exception:
            continue
    # Fallback: estimate from token counts
    pin = getattr(usage, "prompt_tokens", None) or 0
    pout = getattr(usage, "completion_tokens", None) or 0
    rates = PRICING_PER_MTOK.get(model)
    if not rates:
        return None
    return round(pin * rates["in"] / 1_000_000 + pout * rates["out"] / 1_000_000, 6)


# ---------- Public API ---------------------------------------------------
async def refine_with_claude(
    cropped_reaction_png_bytes: bytes,
    raw_ocr_conditions: list[str],
    detected_smiles: dict,
) -> dict:
    """Call OpenRouter -> Claude Haiku 4.5 (vision) to clean up OCR conditions.

    Returns:
        {"conditions": [str,...], "notes": str|None,
         "model_used": str, "cost_usd": float|None,
         "cached": bool, "latency_ms": int}

    On any failure (auth, timeout, parse) returns a fallback dict echoing
    the raw OCR strings with model_used="fallback-raw" and an error note.
    """
    if not LLM_ENABLED:
        return {
            "conditions": list(raw_ocr_conditions),
            "notes": None,
            "model_used": "fallback-disabled",
            "cost_usd": None,
            "cached": False,
            "latency_ms": 0,
        }

    # Empty OCR -> nothing to do
    if not raw_ocr_conditions:
        return {
            "conditions": [],
            "notes": None,
            "model_used": "skip-empty",
            "cost_usd": None,
            "cached": False,
            "latency_ms": 0,
        }

    key = _cache_key(cropped_reaction_png_bytes, raw_ocr_conditions, detected_smiles)
    hit = await _cache_get(key)
    if hit is not None:
        out = dict(hit)
        out["cached"] = True
        return out

    client = _get_client()
    if client is None:
        return {
            "conditions": list(raw_ocr_conditions),
            "notes": None,
            "model_used": "fallback-no-client",
            "cost_usd": None,
            "cached": False,
            "latency_ms": 0,
        }

    user_text = _build_user_text(raw_ocr_conditions, detected_smiles)
    b64 = base64.b64encode(cropped_reaction_png_bytes).decode("ascii")
    image_url = f"data:image/png;base64,{b64}"

    t0 = time.time()
    try:
        resp = await client.chat.completions.create(
            model=OPENROUTER_MODEL,
            max_tokens=LLM_MAX_TOKENS,
            timeout=LLM_TIMEOUT_S,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                },
            ],
        )
    except Exception as e:
        log.warning("openrouter call failed: %s", e)
        return {
            "conditions": list(raw_ocr_conditions),
            "notes": None,
            "model_used": "fallback-raw",
            "cost_usd": None,
            "cached": False,
            "latency_ms": int((time.time() - t0) * 1000),
            "error": f"{type(e).__name__}: {e}",
        }

    latency_ms = int((time.time() - t0) * 1000)
    try:
        text = resp.choices[0].message.content or ""
        parsed = _parse_json_loose(text)
        conditions = parsed.get("conditions")
        if not isinstance(conditions, list):
            raise ValueError(f"conditions not a list: {type(conditions).__name__}")
        # Coerce to strings, strip empty
        conditions = [str(c).strip() for c in conditions if c is not None and str(c).strip()]
        notes = parsed.get("notes")
        if notes is not None and not isinstance(notes, str):
            notes = str(notes)
        if isinstance(notes, str) and not notes.strip():
            notes = None
    except Exception as e:
        log.warning("openrouter response parse failed (%s): %r", e, text[:300] if 'text' in locals() else None)
        return {
            "conditions": list(raw_ocr_conditions),
            "notes": None,
            "model_used": "fallback-raw",
            "cost_usd": None,
            "cached": False,
            "latency_ms": latency_ms,
            "error": f"parse: {type(e).__name__}: {e}",
        }

    cost = _compute_cost(OPENROUTER_MODEL, getattr(resp, "usage", None))
    payload = {
        "conditions": conditions,
        "notes": notes,
        "model_used": OPENROUTER_MODEL,
        "cost_usd": cost,
        "cached": False,
        "latency_ms": latency_ms,
    }
    await _cache_put(key, payload)
    return payload


def cache_stats() -> dict:
    """Lightweight introspection for /api/health."""
    return {
        "size": len(_cache),
        "max": CACHE_MAX,
        "ttl_s": CACHE_TTL_S,
    }
