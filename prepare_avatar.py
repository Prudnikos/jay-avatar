"""
One-time avatar preparation.

Usage:
    python prepare_avatar.py --image /path/to/face.png --name slug-123

Reads a single source image, runs MuseTalk's preprocessing
(face detection → bbox → VAE encode → latents) and stores the
resulting artifacts to $AVATARS_ROOT/<name>/.

Files produced:
    source.png      — the source image (resized to 256x256 if smaller)
    coords.json     — face bbox detected in source
    latents.pt      — VAE-encoded face latents (PyTorch tensor)

Without this preparation step, the avatar runs in DEMO mode (static
frame). After running, set MODE=musetalk on the pod and restart.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from config import CFG


def prepare(image_path: Path, name: str, bbox_shift: int = 0) -> None:
    out_dir = CFG.avatars_root / name
    out_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    # Save source (always)
    cv2.imwrite(str(out_dir / "source.png"), img)
    h, w = img.shape[:2]
    logger.info(f"[Prep] source: {w}x{h}")

    # ── Try MuseTalk preprocessing ───────────────────────────────────
    try:
        import torch
        from musetalk.utils.preprocessing import get_landmark_and_bbox
        from musetalk.utils.utils import load_all_model
    except Exception as e:
        logger.warning(
            f"[Prep] MuseTalk not importable, saving image only "
            f"(avatar will run in DEMO mode): {e}"
        )
        # Write fallback bbox = whole image
        (out_dir / "coords.json").write_text(json.dumps({"bbox": [0, 0, w, h]}))
        return

    # Detect face bbox
    coord_list, _ = get_landmark_and_bbox([str(image_path)], bbox_shift)
    if not coord_list or coord_list[0] is None:
        logger.error("[Prep] No face detected — using full image as bbox")
        bbox = (0, 0, w, h)
    else:
        bbox = tuple(map(int, coord_list[0]))  # x1, y1, x2, y2
    logger.info(f"[Prep] bbox: {bbox}")
    (out_dir / "coords.json").write_text(json.dumps({"bbox": list(bbox)}))

    # Load VAE only (fast — UNet not needed)
    unet_path = CFG.musetalk_root / "models" / "musetalkV15" / "unet.pth"
    unet_cfg = CFG.musetalk_root / "models" / "musetalkV15" / "musetalk.json"
    vae, unet, pe = load_all_model(
        unet_model_path=str(unet_path),
        unet_config=str(unet_cfg),
        version=CFG.musetalk_version,
    )

    # Crop face from source, encode to latents
    x1, y1, x2, y2 = bbox
    face_crop = img[y1:y2, x1:x2]
    face_crop = cv2.resize(face_crop, (256, 256))
    # Normalise to MuseTalk's expected input layout
    face_t = torch.from_numpy(face_crop).float().to("cuda") / 255.0
    face_t = face_t.permute(2, 0, 1).unsqueeze(0)   # 1, 3, 256, 256
    with torch.no_grad():
        latents = vae.get_latents_for_unet(face_t)
    torch.save(latents.cpu(), out_dir / "latents.pt")
    logger.info(f"[Prep] saved latents.pt ({latents.shape})")
    logger.info(f"[Prep] ✅ avatar '{name}' ready at {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare a Jay avatar for streaming")
    ap.add_argument("--image", type=Path, required=True, help="Source face image")
    ap.add_argument("--name", type=str, required=True, help="Avatar name (e.g. business slug)")
    ap.add_argument("--bbox_shift", type=int, default=0, help="MuseTalk bbox_shift (default 0)")
    args = ap.parse_args()

    if not args.image.exists():
        logger.error(f"Image not found: {args.image}")
        sys.exit(1)
    prepare(args.image, args.name, args.bbox_shift)


if __name__ == "__main__":
    main()
