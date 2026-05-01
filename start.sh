#!/bin/bash
# ============================================================================
# Jay Streaming Avatar — GPU Pod entrypoint
# ============================================================================
set -e

cd /workspace

echo "═══════════════════════════════════════════════════════════════════════"
echo "Jay Streaming Avatar — GPU Pod boot"
echo "Time: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
echo "═══════════════════════════════════════════════════════════════════════"

# ── 1. Verify GPU is available ───────────────────────────────────────────
if ! nvidia-smi > /dev/null 2>&1; then
  echo "❌ ERROR: nvidia-smi not available. GPU pod required."
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

# ── 2. Download models if missing ────────────────────────────────────────
# RunPod typically mounts a persistent volume to /workspace/MuseTalk/models.
# On first run the volume is empty — we download all weights (~10GB).
# On subsequent runs we skip this step.
MODELS_DIR=/workspace/MuseTalk/models

if [ ! -f "$MODELS_DIR/musetalkV15/unet.pth" ]; then
  echo "[boot] Models missing — downloading (~10GB, takes 5-10 min)…"
  cd /workspace/MuseTalk
  if [ -f download_weights.sh ]; then
    bash download_weights.sh
  else
    # Fallback if MuseTalk repo layout changed
    python -c "
from huggingface_hub import snapshot_download
snapshot_download('TMElyralab/MuseTalk', local_dir='/workspace/MuseTalk/models', local_dir_use_symlinks=False)
"
  fi
  cd /workspace
  echo "[boot] ✅ Models downloaded"
else
  echo "[boot] ✅ Models already present"
fi

# ── 3. Verify default avatar is prepared ─────────────────────────────────
AVATAR_DIR=/workspace/avatars/default
if [ ! -f "$AVATAR_DIR/latents.pt" ]; then
  echo "[boot] Default avatar not prepared. Will use first-run preparation."
  echo "[boot] ⚠️  To prepare a custom avatar, run:"
  echo "       python /workspace/app/prepare_avatar.py --image avatar.png --name default"
fi

# ── 4. Start FastAPI server ──────────────────────────────────────────────
cd /workspace/app
echo "[boot] Starting FastAPI on :8000"
exec uvicorn server:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 1 \
  --loop uvloop \
  --http httptools \
  --log-level info \
  --no-access-log
