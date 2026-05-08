# rxnim - CPU-only chemistry reaction image parser
# Wraps the upstream HF Space (CYF200127/RxnIM) Pix2Seq + MolScribe + EasyOCR
# stack.  Skips the GPT-4o "Reaction Image Parsing Workflow" (Azure OpenAI
# vendor lock-in) - we serve the local inference path only.
#
# Image budget: ~3.5GB target.  Disk is at 78% on the host; we aggressively
# clean caches between layers.

FROM python:3.11-slim

WORKDIR /app

# System deps:
# - libgl1, libglib2.0-0: opencv runtime
# - libxrender1, libxext6: rdkit drawing
# - poppler-utils: cairosvg / image conversion
# - git: pip can pull from git URLs
# - curl: healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libxrender1 libxext6 git curl \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# CPU-only PyTorch wheel.  RingLeader §3: upstream packages claim torch<2.0
# but actually run fine on torch 2.x.  We use 2.4 which has wheels for
# Python 3.11 from the CPU index.
RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.4.1 torchvision==0.19.1

# Backend deps
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# MolScribe is not on PyPI - install from git with --no-deps because its
# torch<2.0 pin is fiction (RingLeader §3).  We already have torch 2.4 above.
RUN pip install --no-cache-dir --no-deps "MolScribe @ git+https://github.com/thomas0809/MolScribe.git@main"

# Copy in the upstream rxn/ + molscribe/ packages (from the cloned HF Space)
COPY upstream/rxn /app/rxn
COPY upstream/molscribe /app/molscribe

# Pre-download model weights at build time so cold-starts are quick.
# RingLeader gotcha #6.
COPY backend/download_model.py /app/backend/download_model.py
RUN python /app/backend/download_model.py

# Backend code last - changes here don't blow away the model layer
COPY backend/main.py /app/backend/main.py

# Frontend
COPY static /app/static

# Disk cleanup (improvement #2 from review)
RUN apt-get purge -y --auto-remove \
    && rm -rf /root/.cache/pip /tmp/* /var/lib/apt/lists/* \
    && find / -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# RingLeader ARM64-thread-cap pattern - applies to AMD64 too on a small box
ENV OMP_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    PYTORCH_NO_CUDA_MEMORY_CACHING=1 \
    TRANSFORMERS_NO_ADVISORY_WARNINGS=1 \
    PYTHONUNBUFFERED=1

EXPOSE 8000

# Healthcheck so docker can detect stuck containers
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fs http://localhost:8000/healthz || exit 1

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--timeout-keep-alive", "300", "--workers", "1"]
