"""
Jay Streaming Avatar — GPU Pod main server.

Endpoints
─────────
GET  /health                    Liveness probe (used by RunPod / our worker).
GET  /info                      Build info, loaded avatars, mode.
WS   /avatar/{token}            One streaming session.

WebSocket protocol (client ↔ server)
────────────────────────────────────
Client sends one or more JSON messages:
    { "type": "speak", "text": "Hello there" }
    { "type": "stop" }                    # interrupt current speech (barge-in)
    { "type": "ping" }                    # keepalive

Server sends:
    { "type": "ready" }                   # session opened, avatar loaded
    { "type": "segment_start", "id": <int> }
    <binary frame>                        # fmp4 segment bytes (one per chunk)
    { "type": "segment_end", "id": <int>, "duration_ms": <int> }
    { "type": "done", "total_ms": <int> }
    { "type": "error", "message": "..." }

The token is passed as the URL path component:
    wss://<pod-host>/avatar/<token>
The token is verified against JAY_AVATAR_TOKEN_SECRET (HMAC-SHA256).
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from auth import TokenError, TokenPayload, verify_token
from config import CFG, assert_ready
from mp4_segment_muxer import FMP4Muxer, MIME_TYPE, chunk_audio_for_segments
from realtime_engine import RealtimeEngine
from tts import synthesize_pcm

# ── Boot ──────────────────────────────────────────────────────────────────
assert_ready()

app = FastAPI(title="Jay Streaming Avatar", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(CFG.allowed_origins) if CFG.allowed_origins else ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Heavy work: instantiate the engine once. Holds MuseTalk weights in VRAM.
ENGINE: Optional[RealtimeEngine] = None
ENGINE_LOCK = asyncio.Lock()
BOOT_TIME = time.time()


@app.on_event("startup")
async def _startup() -> None:
    global ENGINE
    logger.info("[boot] Loading RealtimeEngine…")
    t0 = time.time()
    ENGINE = RealtimeEngine()  # blocks until model is on GPU
    logger.info(f"[boot] Engine ready in {time.time()-t0:.1f}s, mode={ENGINE.mode}")


# ── HTTP endpoints ────────────────────────────────────────────────────────
@app.get("/health")
async def health(request: Request) -> JSONResponse:
    if ENGINE is None:
        return JSONResponse({"status": "starting"}, status_code=503)
    return JSONResponse({
        "status": "ok",
        "mode": ENGINE.mode,
        "uptime_s": int(time.time() - BOOT_TIME),
        "avatars": list(ENGINE.avatars.keys()),
        "mime": MIME_TYPE,
    })


@app.get("/info")
async def info() -> JSONResponse:
    return JSONResponse({
        "service": "jay-streaming-avatar",
        "version": "1.0.0",
        "mode": ENGINE.mode if ENGINE else "starting",
        "avatars": list(ENGINE.avatars.keys()) if ENGINE else [],
        "config": {
            "fps": CFG.fps,
            "segment_duration_s": CFG.segment_duration_s,
            "sample_rate": CFG.sample_rate,
            "musetalk_version": CFG.musetalk_version,
        },
    })


# ── WebSocket ─────────────────────────────────────────────────────────────
@app.websocket("/avatar/{token}")
async def avatar_socket(ws: WebSocket, token: str) -> None:
    # 1. Verify token before accepting (avoids spending sockets on attackers)
    try:
        payload = verify_token(token)
    except TokenError as e:
        await ws.close(code=4401, reason=f"auth: {e}")
        return

    if ENGINE is None:
        await ws.close(code=4503, reason="engine starting")
        return

    await ws.accept()
    sid = payload.sid
    avatar_name = payload.avatar
    logger.info(f"[ws {sid}] open slug={payload.slug} avatar={avatar_name}")

    # Track current generation task for barge-in
    current_task: Optional[asyncio.Task] = None

    await ws.send_json({"type": "ready", "avatar": avatar_name, "mime": MIME_TYPE})

    async def _generate(text: str, voice_id: Optional[str]) -> None:
        """Generate one speech response and stream segments to the client."""
        gen_t0 = time.time()

        # Acquire engine lock — ENGINE is not thread-safe.
        async with ENGINE_LOCK:
            try:
                # 2. TTS: text → PCM 16kHz
                tts_t0 = time.time()
                pcm = await synthesize_pcm(text, voice_id=voice_id)
                logger.info(f"[ws {sid}] TTS {time.time()-tts_t0:.2f}s "
                            f"({len(pcm)/16000:.2f}s audio)")

                # 3. Init muxer for this utterance
                muxer = FMP4Muxer(
                    width=256,
                    height=256,
                    fps=CFG.fps,
                    sample_rate=CFG.sample_rate,
                    video_bitrate=CFG.video_bitrate,
                    audio_bitrate=CFG.audio_bitrate,
                )
                init_seg = muxer.start()
                if init_seg:
                    await ws.send_bytes(init_seg)

                # 4. Walk audio in segment-sized chunks; for each chunk
                #    generate frames + emit fmp4 segment.
                seg_id = 0
                inference_total = 0.0
                mux_total = 0.0
                first_byte_logged = False

                for n_frames, audio_chunk in chunk_audio_for_segments(
                    pcm, CFG.segment_duration_s, CFG.fps, CFG.sample_rate
                ):
                    seg_t0 = time.time()
                    await ws.send_json({"type": "segment_start", "id": seg_id})

                    # ── Inference: get N frames synchronised to audio_chunk
                    inf_t0 = time.time()
                    # We need EXACTLY n_frames; the engine yields a generator.
                    frames = []
                    for i, frame in enumerate(
                        ENGINE.stream_frames(audio_chunk, avatar_name, fps=CFG.fps)
                    ):
                        if i >= n_frames:
                            break
                        frames.append(frame)
                    # Pad with last frame if engine produced fewer than expected
                    while len(frames) < n_frames and frames:
                        frames.append(frames[-1])
                    if not frames:
                        # No frames at all — synth a black frame
                        frames = [np.zeros((256, 256, 3), dtype=np.uint8)] * n_frames
                    frames_np = np.stack(frames, axis=0)
                    inference_total += time.time() - inf_t0

                    # ── Mux this segment
                    mux_t0 = time.time()
                    seg_bytes = muxer.write_segment(frames_np, audio_chunk)
                    mux_total += time.time() - mux_t0

                    if seg_bytes:
                        await ws.send_bytes(seg_bytes)
                        if not first_byte_logged:
                            logger.info(
                                f"[ws {sid}] first byte to client at "
                                f"{(time.time()-gen_t0)*1000:.0f}ms"
                            )
                            first_byte_logged = True

                    seg_ms = int((time.time() - seg_t0) * 1000)
                    await ws.send_json({
                        "type": "segment_end", "id": seg_id, "duration_ms": seg_ms,
                    })
                    seg_id += 1

                # 5. Trailer (final flush)
                tail = muxer.close()
                if tail:
                    await ws.send_bytes(tail)

                total_ms = int((time.time() - gen_t0) * 1000)
                await ws.send_json({
                    "type": "done",
                    "total_ms": total_ms,
                    "segments": seg_id,
                    "inference_ms": int(inference_total * 1000),
                    "mux_ms": int(mux_total * 1000),
                })
                logger.info(
                    f"[ws {sid}] done in {total_ms}ms "
                    f"(inf={int(inference_total*1000)}ms, mux={int(mux_total*1000)}ms, "
                    f"segs={seg_id})"
                )

            except asyncio.CancelledError:
                logger.info(f"[ws {sid}] generation cancelled (barge-in)")
                raise
            except Exception as e:
                logger.exception(f"[ws {sid}] generation failed: {e}")
                try:
                    await ws.send_json({"type": "error", "message": str(e)[:200]})
                except Exception:
                    pass

    # ── Main message loop ────────────────────────────────────────────
    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await ws.send_json({"type": "error", "message": "invalid json"})
                continue
            mtype = msg.get("type")

            if mtype == "ping":
                await ws.send_json({"type": "pong"})
                continue

            if mtype == "stop":
                if current_task and not current_task.done():
                    current_task.cancel()
                continue

            if mtype == "speak":
                text = (msg.get("text") or "").strip()
                if not text:
                    await ws.send_json({"type": "error", "message": "empty text"})
                    continue
                if current_task and not current_task.done():
                    current_task.cancel()
                    try:
                        await current_task
                    except (asyncio.CancelledError, Exception):
                        pass
                voice_id = msg.get("voice_id")
                current_task = asyncio.create_task(_generate(text, voice_id))
                continue

            await ws.send_json({"type": "error", "message": f"unknown type: {mtype}"})
    except WebSocketDisconnect:
        logger.info(f"[ws {sid}] client disconnected")
    except Exception as e:
        logger.exception(f"[ws {sid}] socket error: {e}")
    finally:
        if current_task and not current_task.done():
            current_task.cancel()
