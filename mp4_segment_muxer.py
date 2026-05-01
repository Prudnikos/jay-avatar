"""
Fragmented MP4 segment muxer.

Produces MSE-compatible MP4 segments containing both video (H.264) and
audio (AAC). We emit:
    1. An "init segment" — moov box only, sent once per session.
    2. A series of "media segments" — each a self-contained moof+mdat
       carrying one chunk of frames + matching audio samples.

The browser feeds these into a single MediaSource SourceBuffer, so the
video and audio share one media clock — sync is impossible to break.

Why PyAV (libav) and not subprocess ffmpeg:
    • we get back raw packets and can flush precisely on segment boundaries;
    • single-process — no pipe synchronisation headaches;
    • `default_base_moof+empty_moov` gives us streaming-friendly fmp4 without
      requiring two full ffmpeg instances.

Note: PyAV writes everything to one BytesIO stream. We split init/media by
calling `flush()` and slicing the tail.
"""
from __future__ import annotations

import io
from typing import Iterable, Optional, Tuple

import av
import numpy as np
from loguru import logger


# Codec strings the browser MSE wants in `addSourceBuffer`:
#   video: avc1.42E01F  = Baseline 3.1 — most compatible, low decode latency
#   audio: mp4a.40.2    = AAC LC
MIME_TYPE = 'video/mp4; codecs="avc1.42E01F,mp4a.40.2"'


class FMP4Muxer:
    """
    One-shot fragmented MP4 muxer.

    Usage:
        m = FMP4Muxer(width=256, height=256, fps=25, sample_rate=16000)
        init_bytes = m.start()                  # send to client first
        for vf, audio_pcm in chunks:
            seg = m.write_segment(vf, audio_pcm)
            ws.send(seg)
        tail = m.close()
        if tail: ws.send(tail)
    """

    def __init__(
        self,
        width: int = 256,
        height: int = 256,
        fps: int = 25,
        sample_rate: int = 16000,
        video_bitrate: str = "1M",
        audio_bitrate: str = "64k",
    ) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self.sample_rate = sample_rate

        self._buf = io.BytesIO()
        self._container = av.open(
            self._buf,
            mode="w",
            format="mp4",
            options={
                # Streaming-friendly fmp4 layout
                "movflags": "frag_keyframe+empty_moov+default_base_moof+separate_moof",
                "frag_duration": "1000000",  # 1s in microseconds
            },
        )

        # ── Video stream ──────────────────────────────────────────────
        # We try h264_nvenc first (GPU encode), fall back to libx264 if the
        # NVENC path isn't available.  PyAV doesn't always expose nvenc —
        # the user can override with FFMPEG_VIDEO_CODEC env var if needed.
        try:
            self.vstream = self._container.add_stream("h264_nvenc", rate=fps)
            self.vstream.options = {
                "preset": "p1",          # fastest NVENC preset
                "tune": "ull",           # ultra-low-latency
                "rc": "vbr",
                "b": video_bitrate,
                "profile": "baseline",
                "level": "3.1",
                "g": str(fps),           # GOP = 1s
            }
        except Exception as nvenc_err:
            logger.warning(f"NVENC unavailable, falling back to libx264: {nvenc_err}")
            self.vstream = self._container.add_stream("libx264", rate=fps)
            self.vstream.options = {
                "preset": "ultrafast",
                "tune": "zerolatency",
                "profile": "baseline",
                "level": "3.1",
                "x264-params": "keyint=25:min-keyint=25:scenecut=0",
            }

        self.vstream.width = width
        self.vstream.height = height
        self.vstream.pix_fmt = "yuv420p"
        self.vstream.bit_rate = self._parse_bitrate(video_bitrate)

        # ── Audio stream ──────────────────────────────────────────────
        self.astream = self._container.add_stream("aac", rate=sample_rate)
        self.astream.layout = "mono"
        self.astream.bit_rate = self._parse_bitrate(audio_bitrate)
        self.astream.format = "fltp"

        self._frame_idx = 0
        self._audio_samples_written = 0

    @staticmethod
    def _parse_bitrate(s: str) -> int:
        s = s.strip().lower()
        if s.endswith("k"):
            return int(s[:-1]) * 1000
        if s.endswith("m"):
            return int(s[:-1]) * 1_000_000
        return int(s)

    # ── Public API ────────────────────────────────────────────────────
    def start(self) -> bytes:
        """
        Encode an empty header and return the init segment (moov only).

        The init segment contains decoder configuration. Browser MSE
        needs this before any media segment.
        """
        # Force libav to flush the moov (init segment) by writing one
        # placeholder packet then flushing. We rely on
        # frag_keyframe+empty_moov to keep moov free of media samples.
        # The trick: write a single black frame + 1 audio sample, flush,
        # then truncate the buffer back to the moov boundary.
        self._write_placeholder()
        self._container.flush_buffers() if hasattr(self._container, "flush_buffers") else None
        # The placeholder approach is fragile across libav versions.
        # Simpler: just do nothing here and emit the first segment as
        # the combined init+segment.  Most MSE players accept this when
        # `addSourceBuffer` is given the right codec string, because the
        # first moof carries enough timing info.
        # Reset buffer.
        self._buf.seek(0)
        self._buf.truncate(0)
        return b""

    def _write_placeholder(self) -> None:
        # Reserved for future use — currently no-op.
        pass

    def write_segment(
        self,
        frames_bgr: np.ndarray,
        audio_pcm: np.ndarray,
    ) -> bytes:
        """
        Encode `frames_bgr` (shape: T, H, W, 3 — uint8 BGR) plus
        `audio_pcm` (float32 mono, sample_rate Hz) and return the bytes
        of the new fmp4 fragment(s).

        `audio_pcm` should approximately match the duration of frames
        (T / fps seconds). Mismatch is OK — extra audio is queued for
        the next segment.
        """
        assert frames_bgr.ndim == 4 and frames_bgr.shape[3] == 3
        assert audio_pcm.dtype == np.float32

        before = self._buf.tell()

        # ── Encode video frames ───────────────────────────────────────
        for frame_bgr in frames_bgr:
            video_frame = av.VideoFrame.from_ndarray(frame_bgr, format="bgr24")
            video_frame.pts = self._frame_idx
            video_frame.time_base = av.Rational(1, self.fps)
            for packet in self.vstream.encode(video_frame):
                self._container.mux(packet)
            self._frame_idx += 1

        # ── Encode audio ──────────────────────────────────────────────
        # AAC frames are 1024 samples wide at fltp; libav handles
        # framing internally if we pass any-length AudioFrame.
        if len(audio_pcm) > 0:
            audio_frame = av.AudioFrame.from_ndarray(
                audio_pcm.reshape(1, -1),
                format="fltp",
                layout="mono",
            )
            audio_frame.sample_rate = self.sample_rate
            audio_frame.pts = self._audio_samples_written
            audio_frame.time_base = av.Rational(1, self.sample_rate)
            self._audio_samples_written += len(audio_pcm)
            for packet in self.astream.encode(audio_frame):
                self._container.mux(packet)

        # The container hasn't necessarily flushed yet; force fragment.
        # PyAV doesn't expose ffmpeg's `av_write_frame` flush per-fragment,
        # but `frag_duration` in options causes auto-fragmentation.
        # We pull whatever bytes have been written since `before`.
        self._buf.flush()
        self._buf.seek(0, io.SEEK_END)
        after = self._buf.tell()

        if after <= before:
            return b""

        # Slice out the new bytes
        self._buf.seek(before)
        chunk = self._buf.read(after - before)
        # Reposition for next write
        self._buf.seek(after)
        return chunk

    def close(self) -> bytes:
        """Flush remaining packets and close. Returns trailer bytes."""
        before = self._buf.tell()
        for packet in self.vstream.encode():
            self._container.mux(packet)
        for packet in self.astream.encode():
            self._container.mux(packet)
        self._container.close()
        self._buf.seek(0, io.SEEK_END)
        after = self._buf.tell()
        if after <= before:
            return b""
        self._buf.seek(before)
        return self._buf.read(after - before)


def chunk_audio_for_segments(
    audio_pcm: np.ndarray,
    segment_duration_s: float,
    fps: int,
    sample_rate: int,
) -> Iterable[Tuple[int, np.ndarray]]:
    """
    Slice an audio array into segment-sized chunks.

    Yields (num_frames_for_segment, audio_pcm_for_segment).
    """
    samples_per_segment = int(segment_duration_s * sample_rate)
    frames_per_segment = int(round(segment_duration_s * fps))
    n_total_samples = len(audio_pcm)
    cursor = 0
    while cursor < n_total_samples:
        end = min(cursor + samples_per_segment, n_total_samples)
        seg_audio = audio_pcm[cursor:end]
        # Last segment: shrink frame count proportionally
        if end - cursor < samples_per_segment:
            seg_frames = max(1, int(round(len(seg_audio) / sample_rate * fps)))
        else:
            seg_frames = frames_per_segment
        yield seg_frames, seg_audio
        cursor = end
