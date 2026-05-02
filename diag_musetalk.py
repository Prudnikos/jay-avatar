"""
MuseTalk install diagnostic script.

Run this on the Pod to verify MuseTalk is installable and prints what
exact API is available (which can vary between MuseTalk versions).

Usage:
    cd /workspace/MuseTalk    # or wherever the repo is
    python /workspace/app/diag_musetalk.py

It NEVER raises — only prints. Output drives our integration plan.
"""
from __future__ import annotations

import sys
import os
import traceback
from pathlib import Path

print("=" * 70)
print("MuseTalk diagnostic")
print("=" * 70)
print(f"Python:  {sys.version.split()[0]}")
print(f"CWD:     {os.getcwd()}")
print(f"PYPATH:  {sys.path[:3]}")

# ── Check torch / cuda ──────────────────────────────────────────────────────
print("\n[1] PyTorch / CUDA")
try:
    import torch
    print(f"  torch:                {torch.__version__}")
    print(f"  cuda available:       {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  cuda device:          {torch.cuda.get_device_name(0)}")
        print(f"  cuda compute cap:     {torch.cuda.get_device_capability(0)}")
        print(f"  vram total (GB):      {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}")
except Exception as e:
    print(f"  ERR: {e}")

# ── Check MuseTalk repo location ────────────────────────────────────────────
print("\n[2] MuseTalk repo")
candidates = [
    Path("/workspace/MuseTalk"),
    Path("/opt/MuseTalk"),
    Path(os.getenv("MUSETALK_ROOT", "/workspace/MuseTalk")),
]
musetalk_root = None
for p in candidates:
    if p.exists() and (p / "musetalk").exists():
        musetalk_root = p
        break
if not musetalk_root:
    print(f"  ERR: MuseTalk not found in any of: {[str(c) for c in candidates]}")
    sys.exit(0)
print(f"  found at:             {musetalk_root}")

# Add to path
sys.path.insert(0, str(musetalk_root))

# ── List models available ───────────────────────────────────────────────────
print("\n[3] Models on disk")
models_dir = musetalk_root / "models"
if not models_dir.exists():
    print(f"  ERR: {models_dir} does not exist")
else:
    for sub in sorted(models_dir.iterdir()):
        if sub.is_dir():
            files = list(sub.iterdir())
            sizes = sum(f.stat().st_size for f in files if f.is_file()) / 1e9
            print(f"  {sub.name}/ ({len(files)} files, {sizes:.2f} GB)")
            for f in files:
                if f.is_file() and f.stat().st_size > 1_000_000:
                    print(f"    - {f.name}: {f.stat().st_size/1e6:.1f} MB")

# ── Try imports ─────────────────────────────────────────────────────────────
print("\n[4] Module imports")

def try_import(module_name: str, attr: str | None = None) -> None:
    try:
        m = __import__(module_name, fromlist=[attr] if attr else [])
        if attr:
            obj = getattr(m, attr, None)
            if obj is None:
                print(f"  {module_name}.{attr}: ATTR NOT FOUND")
                return
            print(f"  {module_name}.{attr}: OK ({type(obj).__name__})")
        else:
            print(f"  {module_name}: OK")
    except Exception as e:
        print(f"  {module_name}: ERR — {type(e).__name__}: {e}")

try_import("musetalk.utils.utils", "load_all_model")
try_import("musetalk.utils.utils", "datagen")
try_import("musetalk.utils.preprocessing", "get_landmark_and_bbox")
try_import("musetalk.utils.blending", "get_image")
try_import("musetalk.whisper.audio2feature", "Audio2Feature")
try_import("musetalk.utils.face_parsing", "FaceParsing")
try_import("scripts.realtime_inference", "Avatar")
try_import("mmcv")
try_import("mmpose")
try_import("mmdet")
try_import("diffusers")

# ── If Avatar class is available, dump its signature ─────────────────────────
print("\n[5] Avatar class API")
try:
    from scripts.realtime_inference import Avatar
    import inspect
    sig = inspect.signature(Avatar.__init__)
    print(f"  Avatar.__init__{sig}")
    methods = [m for m in dir(Avatar) if not m.startswith('_') and callable(getattr(Avatar, m))]
    print(f"  Methods:        {methods}")
    for m in ['inference', 'init', 'process_frames']:
        if hasattr(Avatar, m):
            method = getattr(Avatar, m)
            try:
                print(f"  Avatar.{m}{inspect.signature(method)}")
            except Exception:
                pass
except Exception as e:
    print(f"  ERR: {type(e).__name__}: {e}")
    traceback.print_exc()

# ── Try loading actual models (heavy — only if everything else passed) ──────
print("\n[6] Model load test")
try:
    from musetalk.utils.utils import load_all_model
    import inspect
    sig = inspect.signature(load_all_model)
    print(f"  load_all_model{sig}")
    print(f"  Defaults: {[(p.name, p.default) for p in sig.parameters.values()]}")
except Exception as e:
    print(f"  ERR: {e}")

print("\n" + "=" * 70)
print("Diagnostic complete. Send this output to Claude.")
print("=" * 70)
