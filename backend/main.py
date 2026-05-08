"""
rxnim.sebland.com FastAPI backend.

Wraps the RxnIM project's local CPU inference path:
- rxn/reaction/interface.Reaction (Pix2Seq detector + tokenizer)
- molscribe.MolScribe (atom-level OCSR)
- easyocr.Reader (condition text OCR)

Skips the upstream HF Space's GPT-4o "Reaction Image Parsing Workflow" -
that requires Azure OpenAI key and we want a self-hosted inference path.

Carries forward gotchas from the RingLeader autopsy:
- ensure_rgb() before every model call (RGBA -> silent zero on RxnScribe)
- Singleton model load at startup, NOT per request
- asyncio.wait_for per-request timeout (executor futures cannot be cancelled)
- fix_svg() + RDKit smiles->SVG with sanitize=False fallback
- Pillow.Image.format -> MIME for content sniffing, NOT extension
- Thread caps via env: OMP_NUM_THREADS=2 etc set in compose
"""
from __future__ import annotations

import asyncio
import gc
import io
import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import psutil
import torch
from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

# HEIC/HEIF support for iPhone photos (default Apple format).  Without this
# Pillow.Image.open() raises UnidentifiedImageError on HEIC bytes and the
# /api/predict endpoint returns 415 silently to mobile Safari users.
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except Exception as _heif_err:  # pragma: no cover
    register_heif_opener = None

# rxn package is sibling-imported; sys.path tweak so rxn.reaction.* resolves
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rxn.reaction import Reaction  # noqa: E402

# ---------- Logging ------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("rxnim")

# ---------- Globals ------------------------------------------------------
APP_ROOT = Path(__file__).resolve().parent.parent
CKPT_PATH = APP_ROOT / "rxn" / "model" / "model.ckpt"
STATIC_DIR = APP_ROOT / "static"

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
PREDICT_TIMEOUT_S = 180.0
MAX_WORKERS = 2  # CPU-only box; 2 keeps RAM tight

_executor = ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix="rxnim")
_queue_active = 0
_queue_waiting = 0
_model: Reaction | None = None
_model_load_error: str | None = None

# ---------- Helpers ------------------------------------------------------
def ensure_rgb(img: Image.Image) -> Image.Image:
    """RingLeader §4b-iii fix: paste-on-white for RGBA before convert("RGB").

    Naive .convert("RGB") collapses alpha onto BLACK, which on RxnScribe-style
    Pix2Seq decoders silently returns 0 reactions.  Paste through alpha mask
    onto an explicit white canvas first.
    """
    if img.mode in ("RGBA", "LA", "P"):
        white = Image.new("RGB", img.size, (255, 255, 255))
        if img.mode == "P":
            img = img.convert("RGBA")
        white.paste(img, mask=img.split()[-1] if img.mode == "RGBA" else None)
        return white
    if img.mode != "RGB":
        return img.convert("RGB")
    return img


def fix_svg(svg_str: str) -> str:
    """RingLeader §4b-iv: strip hardcoded width/height, add preserveAspectRatio.
    RDKit's MolDraw2DSVG writes width='400px' height='300px' which clobbers
    container sizing.  Strip them and add a sane preserveAspectRatio.
    """
    svg_str = re.sub(r'(<svg[^>]*?)\s+width=[\'"][^\'"]*[\'"]', r'\1', svg_str, count=1)
    svg_str = re.sub(r'(<svg[^>]*?)\s+height=[\'"][^\'"]*[\'"]', r'\1', svg_str, count=1)
    if 'preserveAspectRatio' not in svg_str:
        svg_str = re.sub(r'(<svg\b)', r'\1 preserveAspectRatio="xMidYMid meet"', svg_str, count=1)
    return svg_str


def smiles_to_svg(smiles: str, width: int = 300, height: int = 200) -> str | None:
    """RingLeader §4b-vi: try RDKit normal sanitize, fall back sanitize=False."""
    if not smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
            if mol is None:
                return None
            try:
                Chem.SanitizeMol(
                    mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES
                )
            except Exception:
                pass
        drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        return fix_svg(drawer.GetDrawingText())
    except Exception as e:
        log.debug("smiles_to_svg failed for %r: %s", smiles, e)
        return None


def _detect_mime(data: bytes, fallback_filename: str = "") -> str | None:
    """RingLeader §4b-ix: sniff bytes via Pillow, NOT extension parsing."""
    try:
        with Image.open(io.BytesIO(data)) as img:
            fmt = (img.format or "").lower()
            if fmt:
                return f"image/{fmt}"
    except Exception:
        return None
    return None


def _serialize_predictions(preds: list[Any]) -> dict:
    """Convert model predictions (list of reaction dicts/dataclasses) to JSON.

    The rxn package's predict_image_file returns a list of reaction dicts
    with keys 'reactants', 'conditions', 'products' and bbox+smiles+text fields.
    """
    out_reactions = []
    for rid, rx in enumerate(preds, start=1):
        # rx may be a dict already (it is, looking at the upstream)
        if not isinstance(rx, dict):
            rx = dict(rx) if hasattr(rx, '__dict__') else {}
        reactants = []
        for r in rx.get('reactants', []) or []:
            smi = r.get('smiles')
            entry = {
                'smiles': smi,
                'text': r.get('text'),
                'bbox': r.get('bbox'),
                'confidence': r.get('confidence'),
                'svg': smiles_to_svg(smi) if smi else None,
            }
            reactants.append(entry)
        products = []
        for p in rx.get('products', []) or []:
            smi = p.get('smiles')
            entry = {
                'smiles': smi,
                'text': p.get('text'),
                'bbox': p.get('bbox'),
                'confidence': p.get('confidence'),
                'svg': smiles_to_svg(smi) if smi else None,
            }
            products.append(entry)
        conditions = []
        for c in rx.get('conditions', []) or []:
            conditions.append({
                'smiles': c.get('smiles'),
                'text': c.get('text'),
                'role': c.get('role'),
                'bbox': c.get('bbox'),
            })
        out_reactions.append({
            'reaction_id': rid,
            'reactants': reactants,
            'conditions': conditions,
            'products': products,
        })
    return {'reactions': out_reactions}


async def _run_in_executor_tracked(fn, *args, timeout_s: float = PREDICT_TIMEOUT_S):
    global _queue_active, _queue_waiting
    loop = asyncio.get_running_loop()
    _queue_waiting += 1
    try:
        future = loop.run_in_executor(_executor, fn, *args)
    finally:
        _queue_waiting -= 1
    _queue_active += 1
    try:
        result = await asyncio.wait_for(future, timeout=timeout_s)
        return result
    finally:
        _queue_active -= 1


def _get_rss_gb() -> float:
    try:
        return psutil.Process(os.getpid()).memory_info().rss / 1_000_000_000
    except Exception:
        return 0.0


def _verify_weights() -> str | None:
    """Improvement #3 from review: verify ckpt sizes before binding port."""
    if not CKPT_PATH.exists():
        return f"missing checkpoint at {CKPT_PATH}"
    sz_mb = CKPT_PATH.stat().st_size / 1_000_000
    if sz_mb < 400 or sz_mb > 500:
        return f"model.ckpt size {sz_mb:.1f}MB is outside expected 432MB range"
    return None


# ---------- App ----------------------------------------------------------
app = FastAPI(title="rxnim", version="0.1.0")


@app.on_event("startup")
async def _startup():
    global _model, _model_load_error
    err = _verify_weights()
    if err:
        _model_load_error = err
        log.error("weight integrity check failed: %s", err)
        return
    try:
        log.info("loading Reaction model from %s ...", CKPT_PATH)
        t0 = time.time()
        _model = Reaction(str(CKPT_PATH), device=torch.device("cpu"))
        log.info("model loaded in %.1fs (RSS %.2fGB)", time.time() - t0, _get_rss_gb())
    except Exception as e:
        _model_load_error = f"{type(e).__name__}: {e}"
        log.exception("model load failed")


@app.get("/healthz")
async def healthz():
    return {"ok": True}


@app.get("/api/health")
async def api_health():
    return {
        "model_loaded": _model is not None,
        "model_error": _model_load_error,
        "rss_gb": round(_get_rss_gb(), 3),
        "queue": {"active": _queue_active, "waiting": _queue_waiting, "max": MAX_WORKERS},
        "version": app.version,
    }


def _do_predict(image_path: str) -> dict:
    """Sync predict path - called inside executor thread."""
    if _model is None:
        raise RuntimeError(f"model not loaded: {_model_load_error}")
    # Always pre-process to RGB to avoid silent-zero on RGBA (RingLeader §4b-iii)
    img = Image.open(image_path)
    img = ensure_rgb(img)
    img.save(image_path, format="PNG")
    preds = _model.predict_image_file(image_path, molscribe=True, ocr=True)
    return _serialize_predictions(preds)


@app.post("/api/predict")
async def api_predict(file: UploadFile = File(...)):
    t_start = time.time()
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty upload")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"upload exceeds {MAX_UPLOAD_BYTES} bytes")
    mime = _detect_mime(data, fallback_filename=file.filename or "")
    if not mime or not mime.startswith("image/"):
        raise HTTPException(status_code=415, detail=f"unsupported media type: {mime or 'unknown'}")
    if _model is None:
        raise HTTPException(status_code=503, detail=f"model not loaded: {_model_load_error}")

    # Save to disk so the upstream's predict_image_file path works unchanged
    tmp_dir = Path("/tmp/rxnim")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / f"upload_{int(time.time()*1000)}_{os.getpid()}.png"
    tmp_path.write_bytes(data)

    try:
        result = await _run_in_executor_tracked(_do_predict, str(tmp_path))
        elapsed_ms = int((time.time() - t_start) * 1000)
        result["processing_time_ms"] = elapsed_ms
        result["model_version"] = "rxnim-pix2seq-cpu"
        result["image_dims"] = list(Image.open(tmp_path).size)
        return result
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="inference timeout")
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        gc.collect()


# Mount static (drag-drop UI)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    idx = STATIC_DIR / "index.html"
    if idx.exists():
        return FileResponse(str(idx))
    return HTMLResponse("<h1>rxnim</h1><p>UI not built</p>")


# Allow large request bodies
@app.middleware("http")
async def big_body(request: Request, call_next):
    return await call_next(request)
