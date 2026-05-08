#!/usr/bin/env python3
"""Run all real_*.png reaction images through the live rxnim API.

Phase-4 test plan:
- 1 warmup probe (discarded timing)
- 3 concurrent requests of real_002_*  (RingLeader §5 cascade-trap check)
- 110 sequential POSTs against /root/ringleader/tests/reactions/real_*.png
- 4 malformed-input cases (1x1 px, RGBA, 10MB random, 0-byte)
- Output: /tmp/rxnim-astrazeneca-results.json + .md summary

Note: Sam's prompt referred to "AstraZeneca" - the only AZ-tagged data in
ringleader is single-molecule images (molecules_realworld/*_az_*.png) NOT
reaction schemes.  RxnIM is designed for reactions, so we use the
real_*.png Wikimedia named-reaction set (110 imgs, same as RingLeader's
RxnScribe validation corpus).
"""
from __future__ import annotations

import argparse
import concurrent.futures
import io
import json
import statistics
import sys
import time
from pathlib import Path

import requests
from PIL import Image

API_BASE = "http://localhost:20040"
RXN_DIR = Path("/root/ringleader/tests/reactions")


def post_image(image_path: Path, timeout: float = 200.0) -> dict:
    """POST one image, return result dict (with timing + image meta)."""
    t0 = time.time()
    with image_path.open("rb") as f:
        files = {"file": (image_path.name, f, "image/png")}
        try:
            r = requests.post(f"{API_BASE}/api/predict", files=files, timeout=timeout)
        except requests.exceptions.RequestException as e:
            return {
                "image": image_path.name,
                "status": "EXCEPTION",
                "error": str(e),
                "elapsed_s": round(time.time() - t0, 2),
            }
    elapsed = time.time() - t0
    record = {
        "image": image_path.name,
        "elapsed_s": round(elapsed, 2),
        "http_status": r.status_code,
    }
    if r.status_code == 200:
        d = r.json()
        rxns = d.get("reactions", [])
        record["status"] = "OK" if rxns else "OK_EMPTY"
        record["n_reactions"] = len(rxns)
        record["n_reactants"] = sum(len(rx.get("reactants", [])) for rx in rxns)
        record["n_products"] = sum(len(rx.get("products", [])) for rx in rxns)
        record["n_conditions"] = sum(len(rx.get("conditions", [])) for rx in rxns)
        record["smiles"] = []
        for rx in rxns:
            for r2 in rx.get("reactants", []) + rx.get("products", []):
                if r2.get("smiles"):
                    record["smiles"].append(r2["smiles"])
    else:
        record["status"] = "HTTP_ERROR"
        try:
            record["error"] = r.json().get("detail", "")
        except Exception:
            record["error"] = r.text[:200]
    return record


def warmup():
    print("=== WARMUP ===", flush=True)
    img = RXN_DIR / "real_002_suzuki_reaction_v_1.png"
    rec = post_image(img)
    print(f"warmup: {rec.get('elapsed_s')}s status={rec.get('status')} n={rec.get('n_reactions')}", flush=True)
    return rec


def concurrency_check():
    print("=== CONCURRENCY (3x simultaneous) ===", flush=True)
    img = RXN_DIR / "real_002_suzuki_reaction_v_1.png"
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
        futs = [ex.submit(post_image, img, 90.0) for _ in range(3)]
        recs = [f.result() for f in futs]
    elapsed = time.time() - t0
    for i, r in enumerate(recs):
        print(f"  c{i}: {r.get('elapsed_s')}s status={r.get('status')}", flush=True)
    print(f"  total wall: {elapsed:.1f}s, all 200: {all(r.get('http_status') == 200 for r in recs)}", flush=True)
    return {"records": recs, "wall_s": elapsed}


def sequential_run(n_images: int | None = None):
    print(f"=== SEQUENTIAL on real_*.png ({'all' if not n_images else n_images}) ===", flush=True)
    files = sorted(RXN_DIR.glob("real_*.png"))
    if n_images:
        files = files[:n_images]
    records = []
    t0 = time.time()
    for i, f in enumerate(files):
        rec = post_image(f, timeout=200)
        records.append(rec)
        marker = "OK" if rec.get("status") == "OK" else (
            "EMPTY" if rec.get("status") == "OK_EMPTY" else f"FAIL({rec.get('status')})"
        )
        print(f"  [{i+1:3}/{len(files)}] {f.name:55} {rec.get('elapsed_s', 0):6.2f}s {marker} n={rec.get('n_reactions', 0)}",
              flush=True)
    print(f"=== sequential done in {time.time()-t0:.1f}s ===", flush=True)
    return records


def malformed_cases():
    print("=== MALFORMED INPUT CASES ===", flush=True)
    out = {}
    # (a) 1x1 image
    buf = io.BytesIO()
    Image.new("RGB", (1, 1), (255, 255, 255)).save(buf, "PNG")
    r = requests.post(f"{API_BASE}/api/predict", files={"file": ("tiny.png", buf.getvalue(), "image/png")}, timeout=120)
    out["1x1_image"] = {"status": r.status_code, "body": r.json() if r.status_code != 500 else r.text[:200]}
    print(f"  1x1: {r.status_code}", flush=True)

    # (b) RGBA image - this is the silent-zero RingLeader §4b-iii case
    buf = io.BytesIO()
    Image.new("RGBA", (200, 200), (255, 0, 0, 128)).save(buf, "PNG")
    r = requests.post(f"{API_BASE}/api/predict", files={"file": ("rgba.png", buf.getvalue(), "image/png")}, timeout=120)
    out["rgba_image"] = {"status": r.status_code, "body": r.json() if r.status_code != 500 else r.text[:200]}
    print(f"  rgba: {r.status_code}", flush=True)

    # (c) 10MB random bytes (should 415 unsupported media)
    rnd = bytes([0xff] * (10 * 1024 * 1024 + 100))
    r = requests.post(f"{API_BASE}/api/predict", files={"file": ("rand.bin", rnd, "application/octet-stream")}, timeout=30)
    out["10mb_random"] = {"status": r.status_code, "body": r.text[:200]}
    print(f"  10mb_random: {r.status_code}", flush=True)

    # (d) 0-byte upload
    r = requests.post(f"{API_BASE}/api/predict", files={"file": ("empty", b"", "image/png")}, timeout=30)
    out["empty"] = {"status": r.status_code, "body": r.text[:200]}
    print(f"  empty: {r.status_code}", flush=True)

    return out


def summarize(seq_records: list[dict]) -> dict:
    """Generate stats from sequential records."""
    n = len(seq_records)
    ok = sum(1 for r in seq_records if r.get("status") == "OK")
    empty = sum(1 for r in seq_records if r.get("status") == "OK_EMPTY")
    err = sum(1 for r in seq_records if r.get("status") not in ("OK", "OK_EMPTY"))
    times = [r["elapsed_s"] for r in seq_records if r.get("elapsed_s")]
    times_ok = [r["elapsed_s"] for r in seq_records if r.get("status") == "OK"]
    return {
        "n_images": n,
        "n_ok": ok,
        "n_ok_empty": empty,
        "n_err": err,
        "pct_non_empty": round(100 * ok / n, 1) if n else 0.0,
        "median_s": round(statistics.median(times), 2) if times else 0.0,
        "p95_s": round(sorted(times)[int(0.95 * len(times))], 2) if len(times) >= 20 else None,
        "max_s": round(max(times), 2) if times else 0.0,
        "min_s": round(min(times), 2) if times else 0.0,
        "median_s_ok_only": round(statistics.median(times_ok), 2) if times_ok else 0.0,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=None, help="limit images (for quick smoke)")
    p.add_argument("--out-json", default="/tmp/rxnim-astrazeneca-results.json")
    p.add_argument("--out-md", default="/tmp/rxnim-astrazeneca-results.md")
    p.add_argument("--skip-concurrency", action="store_true")
    args = p.parse_args()

    # Health check
    h = requests.get(f"{API_BASE}/api/health", timeout=10).json()
    print(f"=== HEALTH ===\n  {json.dumps(h, indent=2)}", flush=True)
    if not h.get("model_loaded"):
        print("ERROR: model not loaded", file=sys.stderr); sys.exit(1)

    warmup_rec = warmup()
    conc_res = None if args.skip_concurrency else concurrency_check()
    seq_records = sequential_run(args.n)
    malformed = malformed_cases()
    summary = summarize(seq_records)

    full = {
        "test_corpus": str(RXN_DIR),
        "n_images_requested": args.n or "all",
        "warmup": warmup_rec,
        "concurrency": conc_res,
        "malformed": malformed,
        "summary": summary,
        "records": seq_records,
    }
    Path(args.out_json).write_text(json.dumps(full, indent=2))
    print(f"\n=== SUMMARY ===\n  {json.dumps(summary, indent=2)}")
    print(f"\nwrote {args.out_json}")

    md = f"""# rxnim Phase-4 test results

Corpus: {RXN_DIR} (real_*.png from RingLeader's named-reaction Wikimedia set)

## Note on the "AstraZeneca set"

The user's prompt mentioned an AstraZeneca image set.  The ringleader tree
has `tests/molecules_realworld/*_az_*.png` (~30 single molecules) but no
AstraZeneca reaction-scheme images.  RxnIM is a reaction parser, so we ran
against the 110 `real_*.png` Wikimedia named-reaction schemes - the same
corpus RingLeader's `production-hardening` PR used to validate RxnScribe.

## Summary

| metric | value |
|---|---|
| images tested | {summary['n_images']} |
| reactions found | {summary['n_ok']} ({summary['pct_non_empty']}%) |
| empty (no reactions) | {summary['n_ok_empty']} |
| errors / non-200 | {summary['n_err']} |
| median latency | {summary['median_s']}s |
| p95 latency | {summary['p95_s']}s |
| max latency | {summary['max_s']}s |
| median latency (OK only) | {summary['median_s_ok_only']}s |

## Warmup

- {warmup_rec.get('elapsed_s', '?')}s, status={warmup_rec.get('status')}, found {warmup_rec.get('n_reactions', 0)} reaction(s)

## Concurrency check (3x simultaneous)

"""
    if conc_res:
        md += f"- wall time: {conc_res['wall_s']:.1f}s\n"
        for i, r in enumerate(conc_res['records']):
            md += f"- c{i}: {r.get('elapsed_s')}s status={r.get('status')}\n"
    else:
        md += "- skipped\n"

    md += "\n## Malformed inputs\n\n"
    for k, v in malformed.items():
        md += f"- **{k}**: HTTP {v['status']}\n"

    md += "\n## Per-image (top 20 by latency)\n\n| image | latency (s) | status | reactions |\n|---|---|---|---|\n"
    for rec in sorted(seq_records, key=lambda r: r.get('elapsed_s', 0), reverse=True)[:20]:
        md += f"| `{rec['image']}` | {rec.get('elapsed_s', 0)} | {rec.get('status')} | {rec.get('n_reactions', 0)} |\n"

    Path(args.out_md).write_text(md)
    print(f"wrote {args.out_md}")


if __name__ == "__main__":
    main()
