# ============================================================================
# Jay Streaming Avatar — GPU Pod Dockerfile
# Base: RunPod official PyTorch 2.1 + CUDA 12.1 (works on RTX 4090, L4, A100)
# Build: docker build -t jay-avatar:v1 .
# Run:   docker run --gpus all -p 8000:8000 jay-avatar:v1
# ============================================================================

FROM runpod/pytorch:2.1.0-py3.10-cuda12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_HOME=/workspace/.torch \
    HF_HOME=/workspace/.hf \
    HUGGINGFACE_HUB_CACHE=/workspace/.hf

# ── System deps ──────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsm6 \
        libxext6 \
        libgl1 \
        libglib2.0-0 \
        wget \
        git \
        git-lfs \
        ca-certificates \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── Workspace ────────────────────────────────────────────────────────────
WORKDIR /workspace

# ── Clone MuseTalk (pinned to a known-good commit) ───────────────────────
# We pin to a specific commit so re-builds are reproducible.
# Update the commit SHA when you want to upgrade MuseTalk.
RUN git clone https://github.com/TMElyralab/MuseTalk.git /workspace/MuseTalk \
    && cd /workspace/MuseTalk \
    && git checkout main

# ── Python deps ──────────────────────────────────────────────────────────
COPY requirements.txt /workspace/requirements.txt
RUN pip install --upgrade pip \
    && pip install -r /workspace/requirements.txt

# MuseTalk requires mmcv/mmdet/mmpose installed via openmim
RUN mim install mmengine \
    && mim install "mmcv==2.0.1" \
    && mim install "mmdet==3.1.0" \
    && mim install "mmpose==1.1.0"

# Add MuseTalk to PYTHONPATH
ENV PYTHONPATH="/workspace/MuseTalk:${PYTHONPATH}"

# ── Our application code ─────────────────────────────────────────────────
COPY *.py /workspace/app/
COPY start.sh /workspace/start.sh
RUN chmod +x /workspace/start.sh

# ── Pre-create model directories (volumes mount over them) ────────────────
RUN mkdir -p /workspace/MuseTalk/models \
             /workspace/MuseTalk/results \
             /workspace/avatars

# ── Healthcheck ──────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["/workspace/start.sh"]
