# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""FastAPI localhost application for the Whisper Jacobian Lens explorer."""

from __future__ import annotations

import argparse
import ipaddress
import json
import logging
import math
import re
import sys
import threading
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Protocol
from urllib.parse import urlsplit

import torch
from fastapi import Body, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from jlens.audio_io import (
    AUDIO_PREPROCESSING_VERSION,
    DecodedAudio,
    decode_audio_bytes,
)
from jlens.phonetic_signatures import PhoneSignaturePrototypes
from jlens.whisper import HFWhisperLensModel
from jlens.whisper_analysis import analyze_whisper_run
from jlens.whisper_lens import WhisperJacobianLens

MAX_UPLOAD_BYTES = 64 * 1024 * 1024
MAX_MULTIPART_OVERHEAD = 1024 * 1024
SAMPLE_MANIFEST_NAME = "samples.json"
SAMPLE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
SAMPLE_MEDIA_TYPES = {
    ".flac": "audio/flac",
    ".m4a": "audio/mp4",
    ".mp3": "audio/mpeg",
    ".ogg": "audio/ogg",
    ".opus": "audio/ogg",
    ".wav": "audio/wav",
    ".webm": "audio/webm",
}
LOGGER = logging.getLogger(__name__)
DEFAULT_LFM_SERVING_MAX_NEW_TOKENS = 512


@dataclass(frozen=True)
class BundledAudioSample:
    """One trusted audio file and its public sample-picker metadata."""

    sample_id: str
    path: Path
    title: str
    description: str | None
    transcript: str | None
    duration_seconds: float | None
    media_type: str
    badge: str | None = None
    recommended_for: str | None = None

    def public_metadata(self) -> dict[str, Any]:
        payload = {
            "id": self.sample_id,
            "title": self.title,
            "description": self.description,
            "transcript": self.transcript,
            "duration_seconds": self.duration_seconds,
            "filename": self.path.name,
            "media_type": self.media_type,
            "audio_url": f"/api/samples/{self.sample_id}",
        }
        if self.badge is not None:
            payload["badge"] = self.badge
        if self.recommended_for is not None:
            payload["recommended_for"] = self.recommended_for
        return payload


def _sample_directory(samples_dir: str | Path | None) -> Path | None:
    if samples_dir is not None:
        configured = Path(samples_dir).expanduser()
        if not configured.is_dir():
            raise ValueError(
                f"configured sample audio directory does not exist: {configured}"
            )
        return configured.resolve()

    source_checkout = Path(__file__).resolve().parent.parent / "samples"
    installed_data = Path(sys.prefix) / "share" / "jlens" / "samples"
    for candidate in (source_checkout, installed_data):
        if candidate.is_dir() and (
            (candidate / SAMPLE_MANIFEST_NAME).is_file()
            or any(
                child.is_file() and child.suffix.lower() in SAMPLE_MEDIA_TYPES
                for child in candidate.iterdir()
            )
        ):
            return candidate.resolve()
    return None


def _phonetic_experiment_directory(
    directory: str | Path | None,
) -> Path | None:
    """Resolve an explicitly enabled, local-only phonetic report microsite."""

    if directory is None:
        return None
    resolved = Path(directory).expanduser().resolve()
    if not resolved.is_dir():
        raise ValueError(
            f"configured phonetic experiment directory does not exist: {resolved}"
        )
    if not (resolved / "index.html").is_file():
        raise ValueError(
            "configured phonetic experiment directory must contain index.html: "
            f"{resolved}"
        )
    return resolved


def _sample_path(directory: Path, filename: str) -> Path:
    if not filename or Path(filename).name != filename:
        raise ValueError(f"sample filename must be a basename: {filename!r}")
    path = (directory / filename).resolve()
    if not path.is_relative_to(directory) or not path.is_file():
        raise ValueError(f"sample audio file does not exist: {filename!r}")
    if path.suffix.lower() not in SAMPLE_MEDIA_TYPES:
        raise ValueError(f"sample has an unsupported audio format: {filename!r}")
    return path


def _optional_manifest_text(entry: dict[str, Any], key: str) -> str | None:
    value = entry.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"sample {key!r} must be a string or null")
    return value


def _manifest_samples(directory: Path, payload: Any) -> list[BundledAudioSample]:
    if not isinstance(payload, dict) or not isinstance(payload.get("samples"), list):
        raise ValueError(
            f"{SAMPLE_MANIFEST_NAME} must be an object with a 'samples' list"
        )

    samples: list[BundledAudioSample] = []
    for index, raw_entry in enumerate(payload["samples"]):
        if not isinstance(raw_entry, dict):
            raise ValueError(f"sample manifest entry {index} must be an object")
        sample_id = raw_entry.get("id")
        filename = raw_entry.get("file")
        title = raw_entry.get("title")
        if not isinstance(sample_id, str) or not SAMPLE_ID_PATTERN.fullmatch(
            sample_id
        ):
            raise ValueError(
                f"sample manifest entry {index} has an invalid id: {sample_id!r}"
            )
        if not isinstance(filename, str):
            raise ValueError(f"sample {sample_id!r} must name an audio file")
        if not isinstance(title, str) or not title.strip():
            raise ValueError(f"sample {sample_id!r} must have a non-empty title")

        duration = raw_entry.get("duration_seconds")
        if duration is not None:
            if isinstance(duration, bool) or not isinstance(duration, (int, float)):
                raise ValueError(
                    f"sample {sample_id!r} duration_seconds must be numeric or null"
                )
            duration = float(duration)
            if not 0 < duration <= 30:
                raise ValueError(
                    f"sample {sample_id!r} duration_seconds must be in (0, 30]"
                )

        path = _sample_path(directory, filename)
        samples.append(
            BundledAudioSample(
                sample_id=sample_id,
                path=path,
                title=title.strip(),
                description=_optional_manifest_text(raw_entry, "description"),
                transcript=_optional_manifest_text(raw_entry, "transcript"),
                duration_seconds=duration,
                media_type=SAMPLE_MEDIA_TYPES[path.suffix.lower()],
                badge=_optional_manifest_text(raw_entry, "badge"),
                recommended_for=_optional_manifest_text(
                    raw_entry, "recommended_for"
                ),
            )
        )
    return samples


def _inferred_samples(directory: Path) -> list[BundledAudioSample]:
    samples: list[BundledAudioSample] = []
    for path in sorted(directory.iterdir(), key=lambda candidate: candidate.name):
        if not path.is_file() or path.suffix.lower() not in SAMPLE_MEDIA_TYPES:
            continue
        safe_path = _sample_path(directory, path.name)
        sample_id = re.sub(r"[^A-Za-z0-9_-]+", "-", path.stem).strip("-")[:64]
        if not sample_id:
            continue
        samples.append(
            BundledAudioSample(
                sample_id=sample_id,
                path=safe_path,
                title=path.stem.replace("_", " ").replace("-", " ").title(),
                description=None,
                transcript=None,
                duration_seconds=None,
                media_type=SAMPLE_MEDIA_TYPES[path.suffix.lower()],
            )
        )
    return samples


def load_sample_catalog(
    samples_dir: str | Path | None = None,
) -> dict[str, BundledAudioSample]:
    """Load only trusted, root-contained files into a stable ID catalog."""
    directory = _sample_directory(samples_dir)
    if directory is None:
        return {}
    manifest_path = directory / SAMPLE_MANIFEST_NAME
    if manifest_path.is_file():
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"could not read {manifest_path}: {exc}") from exc
        samples = _manifest_samples(directory, payload)
    else:
        samples = _inferred_samples(directory)

    catalog: dict[str, BundledAudioSample] = {}
    for sample in samples:
        if sample.sample_id in catalog:
            raise ValueError(f"duplicate bundled sample id: {sample.sample_id!r}")
        catalog[sample.sample_id] = sample
    return catalog


class AnalysisBusyError(RuntimeError):
    """Raised rather than queueing unbounded concurrent model runs."""


class AnalysisBackend(Protocol):
    """Small runtime contract shared by the Whisper and local MLX backends."""

    def status(self) -> dict[str, Any]: ...

    def analyze(
        self,
        payload: bytes,
        *,
        time_bin_overlap_seconds: float | None = None,
    ) -> dict[str, Any]: ...


class ChatterboxBackend(Protocol):
    """Runtime contract for the separate text-to-speech frame trace."""

    def status(self) -> dict[str, Any]: ...

    def synthesize(self, text: str) -> dict[str, Any]: ...

    def trace(self, analysis_id: str, speech_code_index: int) -> dict[str, Any]: ...

    def branch(
        self,
        analysis_id: str,
        speech_code_index: int,
        replacement_code_id: int,
    ) -> dict[str, Any]: ...

    def residual_branch(
        self,
        analysis_id: str,
        speech_code_index: int,
        target_code_id: int,
        layers: list[int],
        forward_span: int,
        max_relative_residual_norm: float,
    ) -> dict[str, Any]: ...


def choose_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class WhisperAnalysisBackend:
    """Own one model/lens pair and reject overlapping local inference runs."""

    def __init__(
        self,
        model: HFWhisperLensModel,
        lens: WhisperJacobianLens,
        *,
        device: torch.device,
        top_k: int = 5,
        time_bin_seconds: float = 0.1,
        time_bin_overlap_seconds: float = 0.02,
        phone_signature_prototypes: PhoneSignaturePrototypes | None = None,
    ) -> None:
        lens.validate_model(model)
        preprocessing_versions = {
            stream.metadata.get("audio_preprocessing_version")
            for stream in (lens.encoder, lens.decoder)
            if stream is not None
            and stream.metadata.get("audio_preprocessing_version") is not None
        }
        if preprocessing_versions and preprocessing_versions != {
            AUDIO_PREPROCESSING_VERSION
        }:
            raise ValueError(
                "lens audio preprocessing is incompatible with this server: "
                f"lens={sorted(preprocessing_versions)!r}, "
                f"server={AUDIO_PREPROCESSING_VERSION!r}"
            )
        self.model = model
        self.lens = lens
        self.device = device
        self.top_k = top_k
        self.time_bin_seconds = time_bin_seconds
        self.time_bin_overlap_seconds = time_bin_overlap_seconds
        if phone_signature_prototypes is not None:
            if lens.encoder is None:
                raise ValueError("phone signatures require an encoder lens")
            phone_signature_prototypes.validate(
                model=model,
                encoder_lens=lens.encoder,
            )
        self.phone_signature_prototypes = phone_signature_prototypes
        self._lock = threading.Lock()

    @classmethod
    def load(
        cls,
        *,
        model_id: str,
        lens_path: str | None,
        encoder_lens_path: str | None,
        decoder_lens_path: str | None,
        revision: str | None,
        device: torch.device,
        top_k: int = 5,
        time_bin_seconds: float = 0.1,
        time_bin_overlap_seconds: float = 0.02,
        phone_signatures_path: str | None = None,
    ) -> WhisperAnalysisBackend:
        from transformers import AutoProcessor, WhisperForConditionalGeneration

        if lens_path is not None and (
            encoder_lens_path is not None or decoder_lens_path is not None
        ):
            raise ValueError(
                "pass either --lens or the per-stream lens flags, not both"
            )
        if lens_path is not None:
            lens = WhisperJacobianLens.load(lens_path)
        else:
            encoder_bundle = (
                None
                if encoder_lens_path is None
                else WhisperJacobianLens.load(encoder_lens_path)
            )
            decoder_bundle = (
                None
                if decoder_lens_path is None
                else WhisperJacobianLens.load(decoder_lens_path)
            )
            lens = WhisperJacobianLens.combine_streams(
                encoder_bundle=encoder_bundle,
                decoder_bundle=decoder_bundle,
            )

        processor = AutoProcessor.from_pretrained(model_id, revision=revision)
        hf_model = WhisperForConditionalGeneration.from_pretrained(
            model_id, revision=revision
        )
        model = HFWhisperLensModel(hf_model, processor, model_id=model_id)
        hf_model.to(device)
        phone_signature_prototypes = (
            None
            if phone_signatures_path is None
            else PhoneSignaturePrototypes.load(phone_signatures_path)
        )
        return cls(
            model,
            lens,
            device=device,
            top_k=top_k,
            time_bin_seconds=time_bin_seconds,
            time_bin_overlap_seconds=time_bin_overlap_seconds,
            phone_signature_prototypes=phone_signature_prototypes,
        )

    def status(self) -> dict[str, Any]:
        streams = [
            name
            for name, value in (
                ("encoder", self.lens.encoder),
                ("decoder", self.lens.decoder),
            )
            if value is not None
        ]
        return {
            "ready": True,
            "model_id": self.model.model_id,
            "device": str(self.device),
            "streams": streams,
            "phone_signatures": self.phone_signature_prototypes is not None,
            "message": f"{', '.join(streams)} lens ready on {self.device}",
        }

    def analyze(
        self,
        payload: bytes,
        *,
        time_bin_overlap_seconds: float | None = None,
    ) -> dict[str, Any]:
        decoded = decode_audio_bytes(payload)
        if not self._lock.acquire(blocking=False):
            raise AnalysisBusyError(
                "another Whisper analysis is already running; retry shortly"
            )
        try:
            with torch.inference_mode():
                return self._analyze_decoded(
                    decoded,
                    time_bin_overlap_seconds=(
                        self.time_bin_overlap_seconds
                        if time_bin_overlap_seconds is None
                        else time_bin_overlap_seconds
                    ),
                )
        finally:
            self._lock.release()

    def _analyze_decoded(
        self,
        decoded: DecodedAudio,
        *,
        time_bin_overlap_seconds: float,
    ) -> dict[str, Any]:
        feature_batch = self.model.processor.feature_extractor(
            decoded.waveform,
            sampling_rate=decoded.sampling_rate,
            return_tensors="pt",
            return_attention_mask=True,
        )
        input_features = feature_batch.input_features.to(self.device)
        attention_mask = feature_batch.attention_mask.to(self.device)
        generated = self.model.generate(
            input_features,
            attention_mask=attention_mask,
            return_dict_in_generate=True,
            return_token_timestamps=True,
        )
        sequence_ids = generated["sequences"].cpu()
        token_timestamps = generated.get("token_timestamps")
        if token_timestamps is not None:
            token_timestamps = token_timestamps.cpu()
        special_ids = set(int(token_id) for token_id in self.model.tokenizer.all_special_ids)
        has_ordinary_target = any(
            int(token_id) not in special_ids for token_id in sequence_ids[0, 1:]
        )
        inputs = self.model.prepare_audio(
            decoded.waveform,
            sampling_rate=decoded.sampling_rate,
            sequence_ids=sequence_ids,
            include_eos_target=not has_ordinary_target,
            duration_seconds=decoded.duration_seconds,
        )
        return analyze_whisper_run(
            self.model,
            self.lens,
            inputs,
            decoded.waveform,
            token_timestamps=token_timestamps,
            top_k=self.top_k,
            time_bin_seconds=self.time_bin_seconds,
            time_bin_overlap_seconds=time_bin_overlap_seconds,
            phone_signature_prototypes=self.phone_signature_prototypes,
        )


def create_app(
    backend: AnalysisBackend | None = None,
    *,
    web_dir: str | Path | None = None,
    samples_dir: str | Path | None = None,
    chatterbox_backend: ChatterboxBackend | None = None,
    phonetic_experiment_dir: str | Path | None = None,
) -> FastAPI:
    app = FastAPI(title="Audio Jacobian Lens", docs_url="/api/docs")
    sample_catalog = load_sample_catalog(samples_dir)
    private_phonetic_site = _phonetic_experiment_directory(
        phonetic_experiment_dir
    )

    @app.middleware("http")
    async def protect_local_analysis(request: Request, call_next):
        protected_paths = {
            "/api/analyze",
            "/api/chatterbox/generate",
            "/api/chatterbox/trace",
            "/api/chatterbox/branch",
            "/api/chatterbox/residual-branch",
        }
        if request.url.path in protected_paths:
            origin = request.headers.get("origin")
            host = request.headers.get("host", "")
            if origin and urlsplit(origin).netloc != host:
                return JSONResponse(
                    {"detail": "cross-origin analysis requests are not allowed"},
                    status_code=403,
                )
            content_length = request.headers.get("content-length")
            if content_length is not None:
                try:
                    too_large = int(content_length) > (
                        MAX_UPLOAD_BYTES + MAX_MULTIPART_OVERHEAD
                    )
                except ValueError:
                    too_large = True
                if too_large:
                    return JSONResponse(
                        {"detail": "analysis request exceeds the 64 MB limit"},
                        status_code=413,
                    )
        return await call_next(request)

    @app.get("/api/status")
    def status() -> dict[str, Any]:
        if backend is None:
            return {
                "ready": False,
                "model_id": None,
                "message": "No fitted audio lens was supplied; demo mode is available.",
            }
        return backend.status()

    @app.get("/api/samples")
    def list_samples() -> dict[str, Any]:
        return {
            "samples": [
                sample.public_metadata() for sample in sample_catalog.values()
            ]
        }

    @app.get("/api/samples/{sample_id:path}")
    def get_sample(sample_id: str) -> FileResponse:
        sample = sample_catalog.get(sample_id)
        if sample is None:
            raise HTTPException(status_code=404, detail="Bundled audio sample not found")
        return FileResponse(
            sample.path,
            media_type=sample.media_type,
            headers={"Cache-Control": "public, max-age=3600"},
        )

    @app.get("/api/chatterbox/status")
    def chatterbox_status() -> dict[str, Any]:
        if chatterbox_backend is None:
            return {
                "ready": False,
                "backend": None,
                "message": (
                    "No local Chatterbox backend is loaded; the synthetic "
                    "frame-trace demo remains available."
                ),
            }
        return chatterbox_backend.status()

    @app.post("/api/chatterbox/generate")
    async def chatterbox_generate(
        text: Annotated[str, Form()],
    ) -> dict[str, Any]:
        if chatterbox_backend is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "No local Chatterbox backend is loaded. Start the dedicated "
                    "audio-jlens-chatterbox server."
                ),
            )
        try:
            return chatterbox_backend.synthesize(text)
        except AnalysisBusyError as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            LOGGER.exception("Chatterbox generation failed")
            raise HTTPException(
                status_code=500, detail=f"Chatterbox generation failed: {exc}"
            ) from exc

    @app.post("/api/chatterbox/trace")
    async def chatterbox_trace(
        payload: Annotated[dict[str, Any], Body()],
    ) -> dict[str, Any]:
        if chatterbox_backend is None:
            raise HTTPException(
                status_code=503,
                detail="No local Chatterbox backend is loaded.",
            )
        analysis_id = payload.get("analysis_id")
        speech_code_index = payload.get("speech_code_index")
        if not isinstance(analysis_id, str) or not analysis_id:
            raise HTTPException(
                status_code=400, detail="analysis_id must be a non-empty string"
            )
        if isinstance(speech_code_index, bool) or not isinstance(
            speech_code_index, int
        ):
            raise HTTPException(
                status_code=400, detail="speech_code_index must be an integer"
            )
        try:
            return chatterbox_backend.trace(analysis_id, speech_code_index)
        except AnalysisBusyError as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(
                status_code=404,
                detail="Chatterbox analysis expired; generate the speech again.",
            ) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            LOGGER.exception("Chatterbox trace failed")
            raise HTTPException(
                status_code=500, detail=f"Chatterbox trace failed: {exc}"
            ) from exc

    @app.post("/api/chatterbox/branch")
    async def chatterbox_branch(
        payload: Annotated[dict[str, Any], Body()],
    ) -> dict[str, Any]:
        if chatterbox_backend is None:
            raise HTTPException(
                status_code=503,
                detail="No local Chatterbox backend is loaded.",
            )
        expected_fields = {
            "analysis_id",
            "speech_code_index",
            "replacement_code_id",
        }
        unknown_fields = sorted(set(payload) - expected_fields)
        if unknown_fields:
            raise HTTPException(
                status_code=400,
                detail=(
                    "unknown Chatterbox branch field(s): "
                    + ", ".join(unknown_fields)
                ),
            )
        analysis_id = payload.get("analysis_id")
        speech_code_index = payload.get("speech_code_index")
        replacement_code_id = payload.get("replacement_code_id")
        if (
            not isinstance(analysis_id, str)
            or not analysis_id
            or not analysis_id.strip()
        ):
            raise HTTPException(
                status_code=400, detail="analysis_id must be a non-empty string"
            )
        if isinstance(speech_code_index, bool) or not isinstance(
            speech_code_index, int
        ):
            raise HTTPException(
                status_code=400, detail="speech_code_index must be an integer"
            )
        if isinstance(replacement_code_id, bool) or not isinstance(
            replacement_code_id, int
        ):
            raise HTTPException(
                status_code=400, detail="replacement_code_id must be an integer"
            )
        try:
            return chatterbox_backend.branch(
                analysis_id,
                speech_code_index,
                replacement_code_id,
            )
        except AnalysisBusyError as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(
                status_code=404,
                detail="Chatterbox analysis expired; generate the speech again.",
            ) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            LOGGER.exception("Chatterbox forced-code branch failed")
            raise HTTPException(
                status_code=500,
                detail=f"Chatterbox forced-code branch failed: {exc}",
            ) from exc

    @app.post("/api/chatterbox/residual-branch")
    async def chatterbox_residual_branch(
        payload: Annotated[dict[str, Any], Body()],
    ) -> dict[str, Any]:
        if chatterbox_backend is None:
            raise HTTPException(
                status_code=503,
                detail="No local Chatterbox backend is loaded.",
            )
        expected_fields = {
            "analysis_id",
            "speech_code_index",
            "target_code_id",
            "layers",
            "forward_span",
            "max_relative_residual_norm",
        }
        unknown_fields = sorted(set(payload) - expected_fields)
        if unknown_fields:
            raise HTTPException(
                status_code=400,
                detail=(
                    "unknown Chatterbox residual-branch field(s): "
                    + ", ".join(unknown_fields)
                ),
            )
        analysis_id = payload.get("analysis_id")
        speech_code_index = payload.get("speech_code_index")
        target_code_id = payload.get("target_code_id")
        layers = payload.get("layers")
        forward_span = payload.get("forward_span")
        max_relative_residual_norm = payload.get(
            "max_relative_residual_norm"
        )
        if (
            not isinstance(analysis_id, str)
            or not analysis_id
            or not analysis_id.strip()
        ):
            raise HTTPException(
                status_code=400, detail="analysis_id must be a non-empty string"
            )
        for field_name, value in (
            ("speech_code_index", speech_code_index),
            ("target_code_id", target_code_id),
            ("forward_span", forward_span),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise HTTPException(
                    status_code=400, detail=f"{field_name} must be an integer"
                )
        if not isinstance(layers, list) or not layers:
            raise HTTPException(
                status_code=400,
                detail="layers must be a non-empty list of integers",
            )
        if any(
            isinstance(layer, bool) or not isinstance(layer, int)
            for layer in layers
        ):
            raise HTTPException(
                status_code=400, detail="layers must contain only integers"
            )
        if len(set(layers)) != len(layers):
            raise HTTPException(
                status_code=400, detail="layers must not contain duplicates"
            )
        if not 1 <= forward_span <= 8:
            raise HTTPException(
                status_code=400, detail="forward_span must be in [1, 8]"
            )
        if (
            isinstance(max_relative_residual_norm, bool)
            or not isinstance(max_relative_residual_norm, (int, float))
            or not math.isfinite(float(max_relative_residual_norm))
            or not 0 < float(max_relative_residual_norm) <= 2.0
        ):
            raise HTTPException(
                status_code=400,
                detail=(
                    "max_relative_residual_norm must be finite and in (0, 2.0]"
                ),
            )
        try:
            return chatterbox_backend.residual_branch(
                analysis_id,
                speech_code_index,
                target_code_id,
                layers,
                forward_span,
                float(max_relative_residual_norm),
            )
        except AnalysisBusyError as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(
                status_code=404,
                detail="Chatterbox analysis expired; generate the speech again.",
            ) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            LOGGER.exception("Chatterbox residual branch failed")
            raise HTTPException(
                status_code=500,
                detail=f"Chatterbox residual branch failed: {exc}",
            ) from exc

    def analyze_request(
        audio: Annotated[UploadFile, File()],
        time_bin_overlap_seconds: float | None,
    ) -> dict[str, Any]:
        if backend is None:
            raise HTTPException(
                status_code=503,
                detail="No fitted audio lens is loaded. Start the server with a backend and compatible lens artifact.",
            )
        try:
            payload = audio.file.read(MAX_UPLOAD_BYTES + 1)
            if len(payload) > MAX_UPLOAD_BYTES:
                raise ValueError("audio upload exceeds the 64 MB limit")
            return backend.analyze(
                payload,
                time_bin_overlap_seconds=time_bin_overlap_seconds,
            )
        except AnalysisBusyError as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            LOGGER.exception("Audio analysis failed")
            raise HTTPException(
                status_code=500, detail=f"Audio analysis failed: {exc}"
            ) from exc

    if backend is not None and getattr(backend, "requires_server_thread", False):

        @app.post("/api/analyze")
        async def analyze_on_server_thread(
            audio: Annotated[UploadFile, File()],
            time_bin_overlap_seconds: Annotated[float | None, Form()] = None,
        ) -> dict[str, Any]:
            # Declaring this route async keeps MLX on the Uvicorn thread that
            # loaded its compiled model. The backend lock still serializes the
            # intentionally local, single-user inference path.
            return analyze_request(audio, time_bin_overlap_seconds)

    else:

        @app.post("/api/analyze")
        def analyze_in_worker(
            audio: Annotated[UploadFile, File()],
            time_bin_overlap_seconds: Annotated[float | None, Form()] = None,
        ) -> dict[str, Any]:
            return analyze_request(audio, time_bin_overlap_seconds)

    if web_dir is not None:
        static_dir = Path(web_dir)
    else:
        source_checkout = Path(__file__).resolve().parent.parent / "web"
        installed_data = Path(sys.prefix) / "share" / "jlens" / "web"
        static_dir = (
            source_checkout if source_checkout.is_dir() else installed_data
        )
    if private_phonetic_site is not None:

        @app.get("/experiments/phonetic-signatures", include_in_schema=False)
        def private_phonetic_experiment_redirect() -> RedirectResponse:
            return RedirectResponse(
                url="/experiments/phonetic-signatures/",
                status_code=307,
            )

        app.mount(
            "/experiments/phonetic-signatures",
            StaticFiles(directory=private_phonetic_site, html=True),
            name="private-phonetic-experiment",
        )

    if static_dir.is_dir():

        @app.get("/")
        def index() -> FileResponse:
            return FileResponse(static_dir / "index.html")

        showcase_page = static_dir / "showcase.html"
        legacy_causal_page = static_dir / "causal.html"
        if showcase_page.is_file():

            @app.get("/showcase")
            def showcase() -> FileResponse:
                return FileResponse(showcase_page)

            @app.get("/causal")
            def legacy_causal_study() -> FileResponse:
                return FileResponse(showcase_page)

        elif legacy_causal_page.is_file():

            @app.get("/causal")
            def legacy_causal_study_fallback() -> FileResponse:
                return FileResponse(legacy_causal_page)

        steering_page = static_dir / "steering.html"
        if steering_page.is_file():

            @app.get("/steering")
            def phone_steering_replay() -> FileResponse:
                return FileResponse(steering_page)

        if (static_dir / "chatterbox.html").is_file():

            @app.get("/chatterbox")
            def chatterbox_study() -> FileResponse:
                return FileResponse(static_dir / "chatterbox.html")

        app.mount("/", StaticFiles(directory=static_dir), name="web")
    return app


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve the local Audio Jacobian Lens explorer"
    )
    parser.add_argument(
        "--backend",
        choices=("whisper-hf", "mlx-lfm"),
        default="whisper-hf",
        help="model runtime; mlx-lfm requires the optional MLX extra on Apple silicon",
    )
    parser.add_argument(
        "--model",
        help="model repository; defaults depend on --backend",
    )
    parser.add_argument("--revision", help="Hugging Face model revision or commit")
    parser.add_argument("--lens", help="combined WhisperJacobianLens artifact")
    parser.add_argument("--encoder-lens", help="encoder-only lens artifact")
    parser.add_argument("--decoder-lens", help="decoder-only lens artifact")
    parser.add_argument(
        "--phone-signatures",
        help=(
            "optional frozen phone-prototype artifact for an encoder-only "
            "phone-signature display mode"
        ),
    )
    parser.add_argument("--device", default="auto", help="auto, mps, cuda, or cpu")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--lfm-max-new-tokens",
        type=_positive_int,
        default=DEFAULT_LFM_SERVING_MAX_NEW_TOKENS,
        help=(
            "serving-only emergency interleaved generation cap for "
            "--backend mlx-lfm; generation normally stops earlier at final "
            "audio EOS"
        ),
    )
    parser.add_argument("--time-bin-seconds", type=float, default=0.1)
    parser.add_argument("--time-bin-overlap-seconds", type=float, default=0.02)
    parser.add_argument("--web-dir")
    parser.add_argument(
        "--phonetic-experiment-dir",
        help=(
            "opt-in local directory containing the private phonetic-signature "
            "microsite; mounted at /experiments/phonetic-signatures/"
        ),
    )
    parser.add_argument(
        "--samples-dir",
        help="directory containing bundled audio and optional samples.json manifest",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    try:
        is_loopback = ipaddress.ip_address(args.host).is_loopback
    except ValueError:
        is_loopback = args.host == "localhost"
    if args.phonetic_experiment_dir is not None and not is_loopback:
        raise SystemExit(
            "--phonetic-experiment-dir is private development output and may "
            "only be served on a loopback host"
        )
    if (
        args.phone_signatures is not None
        and args.lens is None
        and args.encoder_lens is None
    ):
        raise SystemExit("--phone-signatures requires an encoder lens")

    if args.lens is None and args.encoder_lens is None and args.decoder_lens is None:
        backend = None
    elif args.backend == "mlx-lfm":
        if args.lens is None:
            raise SystemExit("--backend mlx-lfm requires one --lens artifact")
        if args.encoder_lens is not None or args.decoder_lens is not None:
            raise SystemExit(
                "--encoder-lens/--decoder-lens are Whisper-only; use --lens for mlx-lfm"
            )
        if args.phone_signatures is not None:
            raise SystemExit("--phone-signatures is supported only by Whisper")
        from jlens.mlx_lfm import DEFAULT_LFM_MODEL_ID, DEFAULT_LFM_MODEL_REVISION
        from jlens.mlx_webapp import MLXLFMAnalysisBackend

        backend = MLXLFMAnalysisBackend.load(
            model_id=args.model or DEFAULT_LFM_MODEL_ID,
            lens_path=args.lens,
            revision=args.revision or DEFAULT_LFM_MODEL_REVISION,
            top_k=args.top_k,
            serving_max_new_tokens=args.lfm_max_new_tokens,
        )
    else:
        backend = WhisperAnalysisBackend.load(
            model_id=args.model or "openai/whisper-tiny.en",
            lens_path=args.lens,
            encoder_lens_path=args.encoder_lens,
            decoder_lens_path=args.decoder_lens,
            revision=args.revision,
            device=choose_device(args.device),
            top_k=args.top_k,
            time_bin_seconds=args.time_bin_seconds,
            time_bin_overlap_seconds=args.time_bin_overlap_seconds,
            phone_signatures_path=args.phone_signatures,
        )
    app = create_app(
        backend,
        web_dir=args.web_dir,
        samples_dir=args.samples_dir,
        phonetic_experiment_dir=args.phonetic_experiment_dir,
    )
    import uvicorn

    if not is_loopback:
        warnings.warn(
            "binding Audio Jacobian Lens beyond loopback exposes uploaded audio "
            "and expensive local inference; add an authenticated reverse proxy",
            stacklevel=1,
        )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
