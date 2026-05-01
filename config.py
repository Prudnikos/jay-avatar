"""
Configuration loaded from environment variables.

Required ENV vars (set as RunPod Pod environment variables):
    JAY_AVATAR_TOKEN_SECRET   shared secret with Cloudflare Worker for JWT signing
    ELEVENLABS_API_KEY        TTS provider API key

Optional:
    ALLOWED_ORIGINS           CSV; default permissive ("*") for dev
    DEFAULT_VOICE_ID          ElevenLabs voice id default
    SEGMENT_DURATION_S        segment length in seconds (default 1.5)
    BBOX_SHIFT                MuseTalk bbox shift (default 0)
    BATCH_SIZE                MuseTalk batch (default 4)
    LOG_LEVEL                 default INFO
"""
import os
from pathlib import Path
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # ── Auth ─────────────────────────────────────────────────────────
    token_secret: str = os.getenv("JAY_AVATAR_TOKEN_SECRET", "")

    # ── TTS ──────────────────────────────────────────────────────────
    elevenlabs_api_key: str = os.getenv("ELEVENLABS_API_KEY", "")
    default_voice_id: str = os.getenv("DEFAULT_VOICE_ID", "pFZP5JQG7iQjIQuC4Bku")

    # ── CORS ─────────────────────────────────────────────────────────
    allowed_origins: tuple = tuple(
        o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()
    )

    # ── Streaming ────────────────────────────────────────────────────
    segment_duration_s: float = float(os.getenv("SEGMENT_DURATION_S", "1.5"))
    fps: int = int(os.getenv("FPS", "25"))
    sample_rate: int = 16000     # MuseTalk's whisper-tiny expects 16kHz
    audio_bitrate: str = os.getenv("AUDIO_BITRATE", "64k")
    video_bitrate: str = os.getenv("VIDEO_BITRATE", "1M")

    # ── MuseTalk ─────────────────────────────────────────────────────
    musetalk_root: Path = Path(os.getenv("MUSETALK_ROOT", "/workspace/MuseTalk"))
    musetalk_version: str = os.getenv("MUSETALK_VERSION", "v15")
    bbox_shift: int = int(os.getenv("BBOX_SHIFT", "0"))
    batch_size: int = int(os.getenv("BATCH_SIZE", "4"))
    use_float16: bool = os.getenv("USE_FLOAT16", "true").lower() == "true"

    # ── Avatars ──────────────────────────────────────────────────────
    avatars_root: Path = Path(os.getenv("AVATARS_ROOT", "/workspace/avatars"))
    default_avatar: str = os.getenv("DEFAULT_AVATAR", "default")

    # ── Misc ─────────────────────────────────────────────────────────
    log_level: str = os.getenv("LOG_LEVEL", "INFO").upper()


CFG = Config()


def assert_ready() -> None:
    """Validate required configuration at startup. Raises on failure."""
    missing = []
    if not CFG.token_secret:
        missing.append("JAY_AVATAR_TOKEN_SECRET")
    if not CFG.elevenlabs_api_key:
        missing.append("ELEVENLABS_API_KEY")
    if not CFG.musetalk_root.exists():
        raise RuntimeError(
            f"MuseTalk repo not found at {CFG.musetalk_root}. "
            f"Re-build the Docker image or set MUSETALK_ROOT."
        )
    if missing:
        raise RuntimeError(
            f"Missing required env vars: {', '.join(missing)}. "
            f"Set them as RunPod Pod env vars."
        )
