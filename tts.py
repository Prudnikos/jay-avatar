"""
ElevenLabs TTS client.

Returns PCM 16-bit mono 16kHz, ready to feed into MuseTalk's whisper.

We use the synchronous endpoint with `output_format=pcm_16000` for now;
WebSocket streaming TTS will come in Phase 2 (smaller first-byte latency).
"""
import asyncio
import io
from typing import Optional

import httpx
import numpy as np
from loguru import logger

from config import CFG


ELEVENLABS_BASE = "https://api.elevenlabs.io/v1"


async def synthesize_pcm(
    text: str,
    voice_id: Optional[str] = None,
    model_id: str = "eleven_turbo_v2_5",  # fastest, multilingual
    timeout_s: float = 12.0,
) -> np.ndarray:
    """
    Generate speech for `text`, return float32 mono PCM at 16kHz.

    Returns
    -------
    np.ndarray of shape (N,), dtype float32, range [-1, 1], 16 kHz mono
    """
    voice = voice_id or CFG.default_voice_id
    url = f"{ELEVENLABS_BASE}/text-to-speech/{voice}"
    headers = {
        "xi-api-key": CFG.elevenlabs_api_key,
        "Content-Type": "application/json",
        "Accept": "audio/pcm",
    }
    body = {
        "text": text,
        "model_id": model_id,
        "voice_settings": {"stability": 0.45, "similarity_boost": 0.75, "speed": 1.0},
    }
    params = {"output_format": "pcm_16000"}

    async with httpx.AsyncClient(timeout=timeout_s) as client:
        resp = await client.post(url, headers=headers, json=body, params=params)
        if resp.status_code != 200:
            raise RuntimeError(
                f"ElevenLabs TTS failed {resp.status_code}: {resp.text[:200]}"
            )
        pcm_bytes = resp.content

    # ElevenLabs returns int16 little-endian mono at 16 kHz.
    samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    logger.info(f"[TTS] {len(samples)} samples ({len(samples)/16000:.2f}s) for {len(text)} chars")
    return samples
