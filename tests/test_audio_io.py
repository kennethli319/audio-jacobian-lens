from __future__ import annotations

import io

import numpy as np
import pytest
import soundfile as sf

from jlens.audio_io import decode_audio_bytes


def wav_bytes(samples: np.ndarray, sampling_rate: int) -> bytes:
    stream = io.BytesIO()
    sf.write(stream, samples, sampling_rate, format="WAV", subtype="PCM_16")
    return stream.getvalue()


def test_decode_audio_resamples_and_mixes_stereo():
    sampling_rate = 8_000
    time = np.arange(sampling_rate // 4) / sampling_rate
    stereo = np.stack(
        [np.sin(2 * np.pi * 220 * time), np.sin(2 * np.pi * 330 * time)],
        axis=1,
    ).astype(np.float32)
    decoded = decode_audio_bytes(wav_bytes(stereo, sampling_rate))
    assert decoded.sampling_rate == 16_000
    assert decoded.waveform.dtype == np.float32
    assert decoded.waveform.shape == (4_000,)
    assert decoded.duration_seconds == pytest.approx(0.25)


def test_decode_audio_rejects_empty_and_overlong_files():
    with pytest.raises(ValueError, match="empty"):
        decode_audio_bytes(b"")
    payload = wav_bytes(np.zeros(2_000, dtype=np.float32), 1_000)
    with pytest.raises(ValueError, match="this explorer accepts at most 1s"):
        decode_audio_bytes(payload, max_duration_seconds=1.0)


def test_downsampling_suppresses_out_of_band_tone():
    sampling_rate = 48_000
    time = np.arange(sampling_rate) / sampling_rate
    # 12 kHz is above the 8 kHz Nyquist limit of the target rate. A linear
    # interpolator aliases it to 4 kHz at almost full amplitude; the production
    # resampler must low-pass it first.
    tone = np.sin(2 * np.pi * 12_000 * time).astype(np.float32)
    decoded = decode_audio_bytes(wav_bytes(tone, sampling_rate))
    rms = float(np.sqrt(np.mean(np.square(decoded.waveform))))
    assert rms < 0.02
