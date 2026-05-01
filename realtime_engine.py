"""
Realtime MuseTalk engine.

Two modes:
    1. PRODUCTION mode (mode="musetalk"):
       Imports the MuseTalk model from /workspace/MuseTalk and runs
       streaming inference.
    2. DEMO mode (mode="demo"):
       Skips MuseTalk entirely. For each audio chunk it emits the
       avatar's source frame unchanged. Useful for validating the
       WebSocket + fmp4 + browser MSE pipeline end-to-end without a
       working MuseTalk install. Switch via env: MODE=demo.

Why this dual mode:
    MuseTalk has fragile, version-specific imports (mmcv, mmdet, mmpose,
    diffusers all need to align). If the GPU pod boots and MuseTalk
    fails to import, we still want the WS server to be reachable and
    return diagnostic frames so we can iterate on the rest of the
    pipeline. Set MODE=musetalk explicitly once the install is verified.

Avatar preparation:
    Avatars live in $AVATARS_ROOT/<name>/ with files:
        source.png       — base 256x256 face image (or video first frame)
        latents.pt       — VAE-encoded face latents (preprocessed)
        coords.json      — face bbox + crop coords
    See prepare_avatar.py to generate these.
"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional, Tuple

import cv2
import numpy as np
from loguru import logger

from config import CFG


@dataclass
class Avatar:
    name: str
    source_frame_bgr: np.ndarray            # (H, W, 3) uint8
    bbox: Tuple[int, int, int, int]         # x1, y1, x2, y2 in source frame
    latents: Optional[object] = None        # VAE latents tensor (production mode)


class RealtimeEngine:
    """
    Loads MuseTalk once, then performs streaming inference.

    The engine is expected to be created at server startup and reused
    for the lifetime of the pod.  It is NOT thread-safe — wrap with a
    lock if you call it from multiple async tasks (server.py does this).
    """

    def __init__(self, mode: Optional[str] = None) -> None:
        self.mode = (mode or os.getenv("MODE", "musetalk")).lower()
        self.lock = threading.Lock()
        self.avatars: dict[str, Avatar] = {}
        self._model_loaded = False
        self._musetalk_modules = None        # lazy import target

        if self.mode == "musetalk":
            self._load_musetalk_models()
        elif self.mode == "demo":
            logger.warning("[Engine] DEMO mode — frames will be static, no lipsync")
        else:
            raise ValueError(f"Unknown MODE: {self.mode}")

        self._load_all_avatars()

    # ── MuseTalk model loading ───────────────────────────────────────
    def _load_musetalk_models(self) -> None:
        """
        Import MuseTalk and load its core models into VRAM.

        We isolate the import here so a missing dep produces one clear
        log line instead of breaking server startup.
        """
        try:
            import torch  # noqa: F401
            from musetalk.utils.utils import load_all_model
            from musetalk.utils.preprocessing import get_landmark_and_bbox  # noqa: F401
            from musetalk.utils.blending import get_image as paste_face_back  # noqa: F401
            from musetalk.whisper.audio2feature import Audio2Feature
        except Exception as e:
            logger.exception(f"[Engine] MuseTalk import failed; falling back to DEMO mode")
            self.mode = "demo"
            return

        t0 = time.time()
        try:
            # MuseTalk 1.5 entrypoint signature (subject to upstream changes):
            #   load_all_model(unet_model_path, unet_config, version)
            unet_path = CFG.musetalk_root / "models" / "musetalkV15" / "unet.pth"
            unet_cfg = CFG.musetalk_root / "models" / "musetalkV15" / "musetalk.json"
            self.vae, self.unet, self.pe = load_all_model(
                unet_model_path=str(unet_path),
                unet_config=str(unet_cfg),
                version=CFG.musetalk_version,
            )
            # Whisper feature extractor
            whisper_path = CFG.musetalk_root / "models" / "whisper" / "tiny.pt"
            self.audio_processor = Audio2Feature(model_path=str(whisper_path))

            self._model_loaded = True
            self._musetalk_modules = {
                "load_all_model": load_all_model,
            }
            logger.info(f"[Engine] MuseTalk loaded in {time.time()-t0:.1f}s")
        except Exception as e:
            logger.exception(f"[Engine] MuseTalk load failed; falling back to DEMO mode")
            self.mode = "demo"

    # ── Avatar loading ───────────────────────────────────────────────
    def _load_all_avatars(self) -> None:
        if not CFG.avatars_root.exists():
            CFG.avatars_root.mkdir(parents=True, exist_ok=True)

        # Each subdirectory is an avatar, except those starting with _ or .
        # (used for things like 'source_images' that hold raw uploads).
        # Also skip dirs that don't contain a source.png — they aren't avatars yet.
        names = []
        for p in CFG.avatars_root.iterdir():
            if not p.is_dir():
                continue
            if p.name.startswith("_") or p.name.startswith("."):
                continue
            if not (p / "source.png").exists():
                logger.info(f"[Engine] skipping '{p.name}' (no source.png)")
                continue
            names.append(p.name)

        if not names:
            logger.warning(
                f"[Engine] No avatars found in {CFG.avatars_root}. "
                f"Run prepare_avatar.py to create one."
            )
            self._generate_placeholder_avatar()
            names = ["default"]

        for name in names:
            try:
                self.avatars[name] = self._load_avatar(name)
                logger.info(f"[Engine] avatar loaded: {name}")
            except Exception as e:
                logger.warning(f"[Engine] avatar '{name}' failed to load: {e}")

    def _load_avatar(self, name: str) -> Avatar:
        avatar_dir = CFG.avatars_root / name
        src_path = avatar_dir / "source.png"
        coords_path = avatar_dir / "coords.json"
        latents_path = avatar_dir / "latents.pt"

        if not src_path.exists():
            raise FileNotFoundError(f"Missing source.png in {avatar_dir}")
        source = cv2.imread(str(src_path))
        if source is None:
            raise RuntimeError(f"Could not read {src_path}")

        if coords_path.exists():
            coords = json.loads(coords_path.read_text())
            bbox = tuple(coords["bbox"])  # x1,y1,x2,y2
        else:
            # Use full image as bbox fallback
            h, w = source.shape[:2]
            bbox = (0, 0, w, h)

        latents = None
        if self.mode == "musetalk" and latents_path.exists():
            try:
                import torch
                latents = torch.load(latents_path, map_location="cuda")
            except Exception as e:
                logger.warning(f"[Engine] latents load failed for {name}: {e}")

        return Avatar(
            name=name,
            source_frame_bgr=source,
            bbox=bbox,
            latents=latents,
        )

    def _generate_placeholder_avatar(self) -> None:
        """Create a 256x256 gradient placeholder so DEMO mode has something to show."""
        avatar_dir = CFG.avatars_root / "default"
        avatar_dir.mkdir(parents=True, exist_ok=True)

        # Pretty gradient (Jay's brand colors: indigo→violet)
        h, w = 256, 256
        img = np.zeros((h, w, 3), dtype=np.uint8)
        for y in range(h):
            for x in range(w):
                t = (x + y) / (w + h)
                # BGR — indigo (#6366f1 = 99,102,241) to violet (#8b5cf6 = 139,92,246)
                b = int(241 + (246 - 241) * t)
                g = int(102 + (92 - 102) * t)
                r = int(99 + (139 - 99) * t)
                img[y, x] = (b, g, r)
        # Add a "J"
        cv2.putText(img, "J", (88, 175), cv2.FONT_HERSHEY_DUPLEX, 4.5,
                    (255, 255, 255), 8, cv2.LINE_AA)

        cv2.imwrite(str(avatar_dir / "source.png"), img)
        (avatar_dir / "coords.json").write_text(
            json.dumps({"bbox": [0, 0, w, h]})
        )
        logger.info(f"[Engine] placeholder avatar created at {avatar_dir}")

    # ── Inference ─────────────────────────────────────────────────────
    def stream_frames(
        self,
        audio_pcm: np.ndarray,
        avatar_name: str = "default",
        fps: int = 25,
    ) -> Generator[np.ndarray, None, None]:
        """
        Yield BGR uint8 frames synchronised to the audio.

        For DEMO mode, yields the static source frame for each timestep.
        For PRODUCTION mode, runs MuseTalk's UNet+VAE pipeline.

        Parameters
        ----------
        audio_pcm : np.ndarray
            float32, mono, 16 kHz.
        avatar_name : str
            Name of avatar to use.
        fps : int
            Output frames-per-second.
        """
        avatar = self.avatars.get(avatar_name) or self.avatars.get("default")
        if avatar is None:
            raise RuntimeError("No avatar available")

        n_frames = int(np.ceil(len(audio_pcm) / 16000.0 * fps))
        if n_frames <= 0:
            return

        if self.mode == "demo" or not self._model_loaded:
            yield from self._demo_frames(avatar, n_frames)
            return

        # Production mode — MuseTalk inference.
        # NOTE: this is a structural sketch. Exact tensor shapes/calls
        # match MuseTalk 1.5 realtime_inference.py at the time of writing
        # (Apr 2025). If MuseTalk upstream changes, update this method.
        try:
            yield from self._musetalk_frames(avatar, audio_pcm, fps, n_frames)
        except Exception as e:
            logger.exception(f"[Engine] inference error, degrading to demo frames: {e}")
            yield from self._demo_frames(avatar, n_frames)

    # ── Demo / fallback ───────────────────────────────────────────────
    def _demo_frames(
        self, avatar: Avatar, n_frames: int
    ) -> Generator[np.ndarray, None, None]:
        # Resize source to canonical 256x256 if needed
        frame = avatar.source_frame_bgr
        if frame.shape[0] != 256 or frame.shape[1] != 256:
            frame = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_AREA)
        # Add a tiny "speaking" indicator that toggles each second so it's
        # visually clear that frames are flowing through the pipeline.
        for i in range(n_frames):
            f = frame.copy()
            phase = (i // 12) % 2  # toggles ~2x per second at 25fps
            if phase == 0:
                cv2.circle(f, (220, 230), 6, (0, 255, 0), -1)
            yield f

    # ── MuseTalk path ─────────────────────────────────────────────────
    def _musetalk_frames(
        self,
        avatar: Avatar,
        audio_pcm: np.ndarray,
        fps: int,
        n_frames: int,
    ) -> Generator[np.ndarray, None, None]:
        """
        Real MuseTalk inference. Called only when self._model_loaded is True.

        This is intentionally batched (CFG.batch_size) to maximise GPU
        utilisation while still emitting frames as soon as a batch finishes.
        """
        import torch
        from musetalk.utils.utils import datagen
        from musetalk.utils.blending import get_image as paste_face_back

        # 1. Audio → whisper features (one feature vector per frame)
        whisper_features = self.audio_processor.audio2feat(audio_pcm)
        # whisper_features: list/tensor of length ~ n_frames

        # 2. Build batches: (whisper_chunk, latent_chunk).
        # MuseTalk's `datagen` walks face latents + audio features in lockstep.
        # We assume avatar.latents is a tensor of (T_avatar, C, H, W).
        if avatar.latents is None:
            raise RuntimeError(f"Avatar '{avatar.name}' has no latents — re-run prepare_avatar.py")

        gen = datagen(
            whisper_chunks=whisper_features,
            vae_encode_latents=avatar.latents,
            batch_size=CFG.batch_size,
            delay_frame=0,
        )

        emitted = 0
        for whisper_batch, latent_batch in gen:
            with torch.no_grad():
                whisper_batch = whisper_batch.to("cuda", dtype=torch.float16 if CFG.use_float16 else torch.float32)
                latent_batch = latent_batch.to("cuda", dtype=torch.float16 if CFG.use_float16 else torch.float32)

                # Positional embeddings for audio
                audio_pe = self.pe(whisper_batch)
                # UNet predicts new face latents
                pred_latents = self.unet.model(
                    latent_batch, 0, encoder_hidden_states=audio_pe
                ).sample
                # VAE decode → face crops (256x256 RGB-ish, depending on VAE)
                recon = self.vae.decode_latents(pred_latents)

            # recon shape: (B, H, W, 3) uint8 (in MuseTalk's convention)
            for face_crop in recon:
                if emitted >= n_frames:
                    return
                # Paste face back into the original frame at avatar.bbox
                full = paste_face_back(
                    avatar.source_frame_bgr.copy(),
                    face_crop,
                    avatar.bbox,
                )
                # Ensure 256x256 output (matches our muxer's video stream)
                if full.shape[0] != 256 or full.shape[1] != 256:
                    full = cv2.resize(full, (256, 256), interpolation=cv2.INTER_AREA)
                yield full
                emitted += 1
