# rxnim

Self-hosted CPU-only chemistry reaction image parser. Drop a reaction-scheme image (PNG / JPEG / WebP / HEIC, up to 10 MB), get back structured JSON with reactant / condition / product SMILES plus rendered SVG depictions. Live at **https://rxnim.sebland.com/**.

This is a thin FastAPI wrapper around the local-inference path of the [CYF200127/RxnIM](https://huggingface.co/spaces/CYF200127/RxnIM) Hugging Face Space, deployed as a single-container CPU service. The upstream Space's GPT-4o "Reaction Image Parsing Workflow" is intentionally **not** included — see "What's not included" below.

## Model components

| Component | Role | Approx size on disk |
| --- | --- | ---: |
| **Pix2Seq** reaction detector (`rxn/model/model.ckpt`) | localizes reactions, reactants, products, condition text in the image and emits a token-stream description | ~432 MB |
| **MolScribe** atom-level OCSR (HuggingFace `thomas0809/MolScribe`) | per-molecule SMILES from the cropped reactant/product bounding boxes | ~140 MB |
| **EasyOCR** | text recognition for condition labels (reagents, solvents, temperatures) | ~50 MB |

Total RSS at idle: ~2.1 GB. Total RSS during inference: ~3 GB. CPU-only inference; 5–30 s per image typical, up to ~90 s under saturated queue.

## Local development

```bash
docker compose up --build
```

The container binds `:20040` on the host (mapped to container `:8000`). Health check at `/api/health`. Static UI at `/`.

For the deployed `rxnim.sebland.com` instance, Caddy reverse-proxies `host.docker.internal:20040` and the TLS cert is a 15-year Cloudflare Origin Certificate (the local origin cert files `cf-origin.pem` / `cf-origin.key` are gitignored — re-issue from the CF dashboard if lost).

## API

`POST /api/predict` — `multipart/form-data` with a single `file` field. Returns:

```json
{
  "reactions": [
    {
      "reaction_id": 1,
      "reactants": [{"smiles": "...", "bbox": [...], "svg": "<svg>...</svg>"}],
      "conditions": [{"text": "Pd(PPh3)4, K2CO3, 80°C", "role": "..."}],
      "products":  [{"smiles": "...", "bbox": [...], "svg": "<svg>...</svg>"}]
    }
  ],
  "processing_time_ms": 28000,
  "model_version": "rxnim-pix2seq-cpu",
  "image_dims": [W, H]
}
```

`GET /api/health` — model load state, RSS, queue depth.

## What's not included

- **No GPU LLM path.** The upstream HF Space chains the local detector into a GPT-4o "Reaction Image Parsing Workflow" wrapper that produces a clean reaction graph and adjudicates ambiguous molecule reads. That requires Azure OpenAI credentials and is intentionally omitted here — this is a self-hosted, vendor-neutral inference service. If you want LLM-backed post-processing, run the detector here and pipe `reactions[*]` into your own LLM.
- **No GPU acceleration.** Pinned to CPU PyTorch wheels (`torch==2.4.1+cpu`). Adding CUDA support is a one-line `requirements.txt` swap, but the dockerfile assumes CPU and the 6 GB memory limit assumes CPU footprint.
- **No fine-tuning / training.** Inference only. The Pix2Seq checkpoint comes from the upstream HF Space's `rxn/model/model.ckpt`.
- **No batch endpoint.** One image per request. The `ThreadPoolExecutor` is sized to 2 workers to keep RSS within the 6 GB cap.

## Origin

Forked-in-spirit from [CYF200127/RxnIM](https://huggingface.co/spaces/CYF200127/RxnIM) (the `upstream/` directory is a git submodule of that HF Space). The lessons baked into `backend/main.py` (RGBA → silent-zero on Pix2Seq, RDKit `MolDraw2DSVG` width/height stripping, MIME sniffing via Pillow not extension, `asyncio.wait_for` per-request timeouts) are carried over from a previous parallel-evolution project, RingLeader.
