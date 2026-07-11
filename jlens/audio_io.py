# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Small, dependency-light audio decoding helpers for the local explorer."""

from __future__ import annotations

import io
import math
import shutil
import subprocess
from dataclasses import dataclass

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

AUDIO_PREPROCESSING_VERSION = "mono-polyphase-16khz-v1"


@dataclass(frozen=True)
class DecodedAudio:
    waveform: np.ndarray
    sampling_rate: int
    duration_seconds: float


def _resample_bandlimited(
    waveform: np.ndarray, source_rate: int, target_rate: int
) -> np.ndarray:
    """Resample mono audio without folding out-of-band energy into speech."""
    if source_rate == target_rate:
        return np.asarray(waveform, dtype=np.float32)
    divisor = math.gcd(source_rate, target_rate)
    output = resample_poly(
        np.asarray(waveform, dtype=np.float32),
        target_rate // divisor,
        source_rate // divisor,
    ).astype(np.float32, copy=False)
    target_length = max(1, round(len(waveform) * target_rate / source_rate))
    if output.size > target_length:
        output = output[:target_length]
    elif output.size < target_length:
        output = np.pad(output, (0, target_length - output.size))
    return output


def _decode_with_soundfile(
    payload: bytes, *, max_duration_seconds: float
) -> tuple[np.ndarray, int]:
    with sf.SoundFile(io.BytesIO(payload)) as handle:
        sampling_rate = int(handle.samplerate)
        if sampling_rate <= 0 or sampling_rate > 384_000:
            raise ValueError(f"unsupported audio sample rate: {sampling_rate}")
        if handle.channels <= 0 or handle.channels > 32:
            raise ValueError(f"unsupported audio channel count: {handle.channels}")
        max_frames = math.ceil(max_duration_seconds * sampling_rate)
        if handle.frames > max_frames:
            duration = handle.frames / sampling_rate
            raise ValueError(
                f"audio is {duration:.1f}s; this explorer accepts at most "
                f"{max_duration_seconds:.0f}s per analysis"
            )
        # Reading at most max_frames + 1 also bounds formats whose headers do not
        # advertise an exact frame count.
        data = handle.read(
            frames=max_frames + 1, dtype="float32", always_2d=True
        )
    if data.shape[0] > max_frames:
        raise ValueError(
            f"audio exceeds the {max_duration_seconds:.0f}s analysis limit"
        )
    return data.mean(axis=1), sampling_rate


def _decode_with_ffmpeg(
    payload: bytes, target_rate: int, *, max_duration_seconds: float
) -> tuple[np.ndarray, int]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise ValueError(
            "audio format is not supported by libsndfile and ffmpeg is not installed"
        )
    try:
        result = subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                "pipe:0",
                "-t",
                f"{max_duration_seconds + 0.1:.3f}",
                "-f",
                "f32le",
                "-ac",
                "1",
                "-ar",
                str(target_rate),
                "pipe:1",
            ],
            input=payload,
            capture_output=True,
            timeout=60,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise ValueError("ffmpeg timed out while decoding the audio") from exc
    if result.returncode != 0 or not result.stdout:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"could not decode audio: {detail or 'unknown format'}")
    waveform = np.frombuffer(result.stdout, dtype="<f4").copy()
    max_samples = math.ceil(max_duration_seconds * target_rate)
    if waveform.size > max_samples:
        raise ValueError(
            f"audio exceeds the {max_duration_seconds:.0f}s analysis limit"
        )
    return waveform, target_rate


def decode_audio_bytes(
    payload: bytes,
    *,
    target_rate: int = 16_000,
    max_duration_seconds: float = 30.0,
) -> DecodedAudio:
    """Decode uploaded audio to a mono float32 waveform at ``target_rate``."""
    if not payload:
        raise ValueError("uploaded audio file is empty")
    try:
        waveform, sampling_rate = _decode_with_soundfile(
            payload, max_duration_seconds=max_duration_seconds
        )
    except (RuntimeError, sf.LibsndfileError):
        waveform, sampling_rate = _decode_with_ffmpeg(
            payload,
            target_rate,
            max_duration_seconds=max_duration_seconds,
        )
    if sampling_rate <= 0 or waveform.size == 0:
        raise ValueError("decoded audio contains no samples")
    if not np.isfinite(waveform).all():
        raise ValueError("decoded audio contains non-finite samples")

    duration = float(waveform.size / sampling_rate)
    if duration > max_duration_seconds + 1e-3:
        raise ValueError(
            f"audio is {duration:.1f}s; this explorer accepts at most "
            f"{max_duration_seconds:.0f}s per analysis"
        )
    waveform = _resample_bandlimited(waveform, sampling_rate, target_rate)
    return DecodedAudio(
        waveform=waveform,
        sampling_rate=target_rate,
        duration_seconds=duration,
    )
