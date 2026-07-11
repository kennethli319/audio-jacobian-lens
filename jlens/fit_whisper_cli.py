# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Command-line fitting workflow for Whisper Jacobian lenses."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path

import torch

from jlens._logging import configure_logging
from jlens.audio_io import AUDIO_PREPROCESSING_VERSION, decode_audio_bytes
from jlens.whisper import HFWhisperLensModel, WhisperLensInputs
from jlens.whisper_fitting import fit_whisper
from jlens.whisper_lens import WhisperJacobianLens


@dataclass(frozen=True)
class ManifestRecord:
    audio_path: Path
    text: str
    metadata: dict


def record_fingerprint(record: ManifestRecord) -> str:
    """Hash the exact audio bytes and reference text for overlap checks."""
    digest = hashlib.sha256()
    audio_bytes = record.audio_path.read_bytes()
    digest.update(len(audio_bytes).to_bytes(8, "big"))
    digest.update(audio_bytes)
    digest.update(record.text.encode("utf-8"))
    return digest.hexdigest()[:20]


def _parse_layers(value: str | None) -> list[int] | None:
    if value is None:
        return None
    if not value.strip():
        return []
    return [int(item.strip()) for item in value.split(",")]


def load_manifest(
    path: Path, *, limit: int | None = None
) -> tuple[list[ManifestRecord], str]:
    """Load JSONL records and hash transcript plus exact audio bytes."""
    records: list[ManifestRecord] = []
    digest = hashlib.sha256()
    base = path.resolve().parent
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            payload = json.loads(raw_line)
            if "audio" not in payload or "text" not in payload:
                raise ValueError(
                    f"{path}:{line_number} needs 'audio' and 'text' fields"
                )
            audio_path = Path(payload["audio"])
            if not audio_path.is_absolute():
                audio_path = base / audio_path
            if not audio_path.is_file():
                raise ValueError(f"audio file not found: {audio_path}")
            text = str(payload["text"])
            audio_bytes = audio_path.read_bytes()
            digest.update(len(audio_bytes).to_bytes(8, "big"))
            digest.update(audio_bytes)
            digest.update(text.encode("utf-8"))
            metadata = {
                key: value
                for key, value in payload.items()
                if key not in ("audio", "text")
            }
            records.append(ManifestRecord(audio_path, text, metadata))
            if limit is not None and len(records) >= limit:
                break
    if not records:
        raise ValueError(f"manifest has no records: {path}")
    return records, digest.hexdigest()[:20]


def _device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def prepare_examples(
    model: HFWhisperLensModel,
    records: list[ManifestRecord],
) -> list[WhisperLensInputs]:
    examples: list[WhisperLensInputs] = []
    for index, record in enumerate(records):
        decoded = decode_audio_bytes(record.audio_path.read_bytes())
        sequence_ids = model.tokenizer(
            record.text, return_tensors="pt"
        ).input_ids
        examples.append(
            model.prepare_audio(
                decoded.waveform,
                sampling_rate=decoded.sampling_rate,
                sequence_ids=sequence_ids,
                duration_seconds=decoded.duration_seconds,
                metadata={
                    "manifest_index": index,
                    "audio_name": record.audio_path.name,
                    **record.metadata,
                },
            )
        )
    return examples


def prepare_aligned_examples(
    model: HFWhisperLensModel,
    records: list[ManifestRecord],
    *,
    window_padding_seconds: float,
) -> list[WhisperLensInputs]:
    """Prepare one generated-token/DTW-aligned audio window per clip."""
    if window_padding_seconds < 0:
        raise ValueError("window_padding_seconds must be non-negative")
    examples: list[WhisperLensInputs] = []
    for record_index, record in enumerate(records):
        decoded = decode_audio_bytes(record.audio_path.read_bytes())
        feature_batch = model.processor.feature_extractor(
            decoded.waveform,
            sampling_rate=decoded.sampling_rate,
            return_tensors="pt",
            return_attention_mask=True,
        )
        with torch.inference_mode():
            generated = model.generate(
                feature_batch.input_features.to(model.input_device),
                attention_mask=feature_batch.attention_mask.to(model.input_device),
                return_dict_in_generate=True,
                return_token_timestamps=True,
            )
        sequence_ids = generated["sequences"].cpu()
        timestamps = generated["token_timestamps"].reshape(-1).float().cpu()
        prepared = model.prepare_audio(
            decoded.waveform,
            sampling_rate=decoded.sampling_rate,
            sequence_ids=sequence_ids,
            duration_seconds=decoded.duration_seconds,
            metadata={
                "manifest_index": record_index,
                "audio_name": record.audio_path.name,
                "alignment": "whisper-cross-attention-dtw",
                **record.metadata,
            },
        )
        valid_decoder = prepared.decoder_position_mask[0].nonzero(
            as_tuple=True
        )[0]
        if valid_decoder.numel() == 0:
            continue
        # Cycle through relative token positions instead of always selecting the
        # middle word, which would bias a small corpus toward sentence-medial
        # syntax. The choice is deterministic for resume/reproducibility.
        selection = (record_index * 7 + 1) % int(valid_decoder.numel())
        decoder_position = int(valid_decoder[selection])
        sequence_position = decoder_position + 1
        start = float(timestamps[sequence_position])
        end = decoded.duration_seconds
        for later in timestamps[sequence_position + 1 :].tolist():
            if later > start + 1e-4:
                end = min(decoded.duration_seconds, float(later))
                break

        valid_encoder = int(prepared.encoder_position_mask[0].sum())
        start_frame = max(
            0, math.floor((start - window_padding_seconds) / 0.02)
        )
        end_frame = min(
            valid_encoder,
            math.ceil((end + window_padding_seconds) / 0.02),
        )
        if end_frame <= start_frame:
            end_frame = min(valid_encoder, start_frame + 1)
        encoder_mask = torch.zeros_like(prepared.encoder_position_mask)
        encoder_mask[:, start_frame:end_frame] = True
        decoder_mask = torch.zeros_like(prepared.decoder_position_mask)
        decoder_mask[:, decoder_position] = True
        target_id = int(prepared.decoder_target_ids[0, decoder_position])
        examples.append(
            replace(
                prepared,
                encoder_position_mask=encoder_mask,
                decoder_position_mask=decoder_mask,
                metadata={
                    **prepared.metadata,
                    "aligned_target_id": target_id,
                    "aligned_target_text": model.tokenizer.decode(
                        [target_id], clean_up_tokenization_spaces=False
                    ),
                    "aligned_start_seconds": start,
                    "aligned_end_seconds": end,
                    "aligned_start_frame": start_frame,
                    "aligned_end_frame": end_frame,
                },
            )
        )
    if not examples:
        raise ValueError("generation produced no alignable text tokens")
    return examples


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit decoder and encoder-to-decoder J-lenses for Whisper"
    )
    parser.add_argument("manifest", type=Path, help="JSONL with audio/text fields")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="openai/whisper-tiny.en")
    parser.add_argument("--revision", help="Hugging Face model revision or commit")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--encoder-layers", default=None, help="e.g. 0,1,2,3")
    parser.add_argument("--decoder-layers", default=None, help="e.g. 0,1,2")
    parser.add_argument("--target-layer", type=int)
    parser.add_argument("--encoder-dim-batch", type=int, default=4)
    parser.add_argument("--decoder-dim-batch", type=int, default=8)
    parser.add_argument(
        "--target-reduction", choices=("sum", "mean"), default="sum"
    )
    parser.add_argument("--skip-encoder", action="store_true")
    parser.add_argument("--skip-decoder", action="store_true")
    parser.add_argument(
        "--encoder-estimator",
        choices=("aligned", "global"),
        default="aligned",
        help="aligned samples one generated token and its DTW audio window per clip",
    )
    parser.add_argument("--alignment-window-padding", type=float, default=0.1)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--no-resume", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.skip_encoder and args.skip_decoder:
        raise SystemExit("cannot skip both encoder and decoder lenses")
    configure_logging()
    records, corpus_fingerprint = load_manifest(args.manifest, limit=args.limit)
    device = _device(args.device)

    from transformers import AutoProcessor, WhisperForConditionalGeneration

    processor = AutoProcessor.from_pretrained(args.model, revision=args.revision)
    hf_model = WhisperForConditionalGeneration.from_pretrained(
        args.model, revision=args.revision
    )
    model = HFWhisperLensModel(hf_model, processor, model_id=args.model)
    hf_model.to(device)
    decoder_examples = prepare_examples(model, records)

    checkpoint_dir = args.checkpoint_dir or args.output.parent / ".checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    provenance = {
        "corpus_fingerprint": corpus_fingerprint,
        "manifest_name": args.manifest.name,
        "requested_examples": len(records),
        "audio_preprocessing_version": AUDIO_PREPROCESSING_VERSION,
        "fit_example_fingerprints": [
            record_fingerprint(record) for record in records
        ],
    }

    decoder_bundle = None
    if not args.skip_decoder:
        decoder_bundle = fit_whisper(
            model,
            decoder_examples,
            encoder_source_layers=[],
            decoder_source_layers=_parse_layers(args.decoder_layers),
            target_decoder_layer=args.target_layer,
            dim_batch=args.decoder_dim_batch,
            target_reduction=args.target_reduction,
            estimator_name="decoder-causal-all-targets",
            artifact_metadata=provenance,
            checkpoint_path=str(checkpoint_dir / "decoder_fit.pt"),
            resume=not args.no_resume,
        )

    encoder_bundle = None
    if not args.skip_encoder:
        encoder_examples = (
            decoder_examples
            if args.encoder_estimator == "global"
            else prepare_aligned_examples(
                model,
                records,
                window_padding_seconds=args.alignment_window_padding,
            )
        )
        encoder_estimator_name = (
            "encoder-global-all-targets"
            if args.encoder_estimator == "global"
            else "encoder-dtw-aligned-token"
        )
        encoder_provenance = {
            **provenance,
            "alignment_window_padding_seconds": (
                None
                if args.encoder_estimator == "global"
                else args.alignment_window_padding
            ),
        }
        encoder_bundle = fit_whisper(
            model,
            encoder_examples,
            encoder_source_layers=_parse_layers(args.encoder_layers),
            decoder_source_layers=[],
            target_decoder_layer=args.target_layer,
            dim_batch=args.encoder_dim_batch,
            target_reduction=args.target_reduction,
            estimator_name=encoder_estimator_name,
            artifact_metadata=encoder_provenance,
            checkpoint_path=str(
                checkpoint_dir / f"encoder_{args.encoder_estimator}_fit.pt"
            ),
            resume=not args.no_resume,
        )

    combined = WhisperJacobianLens.combine_streams(
        encoder_bundle=encoder_bundle,
        decoder_bundle=decoder_bundle,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined.save(str(args.output))
    print(
        f"saved {args.output} ({len(records)} examples, "
        f"corpus {corpus_fingerprint}, device {device})"
    )


if __name__ == "__main__":
    main()
