# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Small held-out evaluation harness for Whisper Jacobian lenses."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from jlens.audio_io import AUDIO_PREPROCESSING_VERSION, decode_audio_bytes
from jlens.fit_whisper_cli import (
    ManifestRecord,
    _device,
    load_manifest,
    record_fingerprint,
)
from jlens.whisper import HFWhisperLensModel
from jlens.whisper_lens import WhisperJacobianLens


def _ranks(logits: torch.Tensor, target_ids: torch.Tensor) -> list[int]:
    target_scores = logits.gather(1, target_ids[:, None])
    return (logits.gt(target_scores).sum(dim=1) + 1).cpu().tolist()


def _summarize(ranks: list[int]) -> dict[str, float | int]:
    if not ranks:
        return {"n": 0}
    return {
        "n": len(ranks),
        "median_rank": float(statistics.median(ranks)),
        "mean_reciprocal_rank": sum(1.0 / rank for rank in ranks) / len(ranks),
        "top_1_rate": sum(rank <= 1 for rank in ranks) / len(ranks),
        "top_10_rate": sum(rank <= 10 for rank in ranks) / len(ranks),
        "top_100_rate": sum(rank <= 100 for rank in ranks) / len(ranks),
    }


def _edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for reference_index, reference_item in enumerate(reference, start=1):
        current = [reference_index]
        for hypothesis_index, hypothesis_item in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[hypothesis_index] + 1,
                    previous[hypothesis_index - 1]
                    + (reference_item != hypothesis_item),
                )
            )
        previous = current
    return previous[-1]


def _normalized_words(text: str) -> list[str]:
    """Lowercase and remove punctuation for a transparent pilot WER/CER."""
    return re.findall(r"[^\W_]+(?:['’][^\W_]+)?", text.casefold())


def _fit_example_fingerprints(lens: WhisperJacobianLens) -> set[str]:
    fingerprints: set[str] = set()
    for stream in (lens.encoder, lens.decoder):
        if stream is not None:
            fingerprints.update(stream.metadata.get("fit_example_fingerprints", []))
    return fingerprints


def _semantic_positions(model: HFWhisperLensModel, inputs) -> list[int]:
    positions = inputs.decoder_position_mask[0].nonzero(as_tuple=True)[0].tolist()
    return [
        position
        for position in positions
        if any(
            character.isalnum()
            for character in model.tokenizer.decode(
                [int(inputs.decoder_target_ids[0, position])],
                clean_up_tokenization_spaces=False,
            )
        )
    ]


def _aligned_frames(
    decoder_positions: list[int],
    timestamps: torch.Tensor,
    *,
    duration: float,
    valid_encoder_positions: int,
) -> list[int]:
    values = timestamps.reshape(-1).float().cpu().tolist()
    frames: list[int] = []
    for decoder_position in decoder_positions:
        sequence_position = decoder_position + 1
        start = values[sequence_position]
        end = duration
        for later in values[sequence_position + 1 :]:
            if later > start + 1e-4:
                end = min(duration, later)
                break
        center = (start + end) / 2
        frames.append(
            max(0, min(valid_encoder_positions - 1, round(center / 0.02)))
        )
    return frames


@torch.no_grad()
def evaluate_records(
    model: HFWhisperLensModel,
    lens: WhisperJacobianLens,
    records: list[ManifestRecord],
) -> dict[str, Any]:
    lens.validate_model(model)
    ranks_by_metric: dict[str, list[int]] = defaultdict(list)
    examples: list[dict[str, str]] = []
    word_errors = 0
    reference_words = 0
    character_errors = 0
    reference_characters = 0

    for record in records:
        decoded = decode_audio_bytes(record.audio_path.read_bytes())
        feature_batch = model.processor.feature_extractor(
            decoded.waveform,
            sampling_rate=decoded.sampling_rate,
            return_tensors="pt",
            return_attention_mask=True,
        )
        generated = model.generate(
            feature_batch.input_features.to(model.input_device),
            attention_mask=feature_batch.attention_mask.to(model.input_device),
            return_dict_in_generate=True,
            return_token_timestamps=True,
        )
        sequence = generated["sequences"].cpu()
        timestamps = generated["token_timestamps"].cpu()
        special_ids = set(int(token_id) for token_id in model.tokenizer.all_special_ids)
        has_ordinary_target = any(
            int(token_id) not in special_ids for token_id in sequence[0, 1:]
        )
        inputs = model.prepare_audio(
            decoded.waveform,
            sampling_rate=decoded.sampling_rate,
            sequence_ids=sequence,
            include_eos_target=not has_ordinary_target,
            duration_seconds=decoded.duration_seconds,
        )
        encoder_layers = [] if lens.encoder is None else lens.encoder.source_layers
        decoder_layers = [] if lens.decoder is None else lens.decoder.source_layers
        encoder, decoder, actual_logits = model.capture(
            inputs,
            encoder_layers=encoder_layers,
            decoder_layers=decoder_layers,
        )
        positions = _semantic_positions(model, inputs)
        targets = inputs.decoder_target_ids[0, positions].to(model.input_device)
        generated_text = model.tokenizer.decode(
            sequence[0], skip_special_tokens=True
        )
        reference_word_items = _normalized_words(record.text)
        generated_word_items = _normalized_words(generated_text)
        reference_character_items = list(" ".join(reference_word_items))
        generated_character_items = list(" ".join(generated_word_items))
        word_errors += _edit_distance(reference_word_items, generated_word_items)
        reference_words += len(reference_word_items)
        character_errors += _edit_distance(
            reference_character_items, generated_character_items
        )
        reference_characters += len(reference_character_items)
        examples.append(
            {
                "audio": record.audio_path.name,
                "reference": record.text,
                "generated": generated_text,
            }
        )

        ranks_by_metric["actual_output_generated_token_self_check"] += _ranks(
            actual_logits[0, positions], targets
        )
        if lens.decoder is not None:
            for layer in decoder_layers:
                residuals = decoder[layer][0, positions].float()
                lens_logits = model.unembed(
                    lens.decoder.transport(residuals, layer)
                ).float()
                direct_logits = model.unembed(residuals).float()
                ranks_by_metric[f"decoder_jlens_L{layer}"] += _ranks(
                    lens_logits, targets
                )
                ranks_by_metric[f"decoder_direct_L{layer}"] += _ranks(
                    direct_logits, targets
                )

        if lens.encoder is not None:
            valid_encoder = int(inputs.encoder_position_mask[0].sum())
            aligned = _aligned_frames(
                positions,
                timestamps,
                duration=decoded.duration_seconds,
                valid_encoder_positions=valid_encoder,
            )
            remote = [
                (frame + valid_encoder // 2) % valid_encoder for frame in aligned
            ]
            for layer in encoder_layers:
                aligned_residuals = encoder[layer][0, aligned].float()
                remote_residuals = encoder[layer][0, remote].float()
                aligned_logits = model.unembed(
                    lens.encoder.transport(aligned_residuals, layer)
                ).float()
                remote_logits = model.unembed(
                    lens.encoder.transport(remote_residuals, layer)
                ).float()
                ranks_by_metric[f"encoder_affine_aligned_L{layer}"] += _ranks(
                    aligned_logits, targets
                )
                ranks_by_metric[f"encoder_affine_remote_L{layer}"] += _ranks(
                    remote_logits, targets
                )
                if lens.encoder.target_mean is not None:
                    target_mean = lens.encoder.target_mean.to(
                        model.input_device
                    ).float()[None]
                    baseline_logits = model.unembed(target_mean).float()
                    ranks_by_metric[f"encoder_delta_aligned_L{layer}"] += _ranks(
                        aligned_logits - baseline_logits, targets
                    )
                    ranks_by_metric[f"encoder_delta_remote_L{layer}"] += _ranks(
                        remote_logits - baseline_logits, targets
                    )

    return {
        "model": lens.model_metadata,
        "estimator": lens.estimator_metadata,
        "n_examples": len(records),
        "evaluation_target": "model_generated_tokens",
        "asr": {
            "word_errors": word_errors,
            "reference_words": reference_words,
            "word_error_rate": (
                None if reference_words == 0 else word_errors / reference_words
            ),
            "character_errors": character_errors,
            "reference_characters": reference_characters,
            "character_error_rate": (
                None
                if reference_characters == 0
                else character_errors / reference_characters
            ),
        },
        "examples": examples,
        "metrics": {
            name: _summarize(values)
            for name, values in sorted(ranks_by_metric.items())
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--lens", type=Path, required=True)
    parser.add_argument("--model", default="openai/whisper-tiny.en")
    parser.add_argument("--revision", help="Hugging Face model revision or commit")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--allow-overlap",
        action="store_true",
        help="allow evaluation records that were used to fit the lens",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    records, _ = load_manifest(args.manifest)
    records = records[args.start :]
    if args.limit is not None:
        records = records[: args.limit]
    if not records:
        raise SystemExit("no evaluation records selected")

    lens = WhisperJacobianLens.load(str(args.lens))
    preprocessing_versions = {
        stream.metadata.get("audio_preprocessing_version")
        for stream in (lens.encoder, lens.decoder)
        if stream is not None
        and stream.metadata.get("audio_preprocessing_version") is not None
    }
    if preprocessing_versions and preprocessing_versions != {
        AUDIO_PREPROCESSING_VERSION
    }:
        raise SystemExit(
            "lens/evaluator audio preprocessing mismatch: "
            f"lens={sorted(preprocessing_versions)!r}, "
            f"evaluator={AUDIO_PREPROCESSING_VERSION!r}"
        )
    overlap = _fit_example_fingerprints(lens).intersection(
        record_fingerprint(record) for record in records
    )
    if overlap and not args.allow_overlap:
        raise SystemExit(
            f"refusing to evaluate on {len(overlap)} fitting example(s); "
            "select a disjoint split or pass --allow-overlap for a smoke test"
        )

    from transformers import AutoProcessor, WhisperForConditionalGeneration

    device = _device(args.device)
    processor = AutoProcessor.from_pretrained(args.model, revision=args.revision)
    hf_model = WhisperForConditionalGeneration.from_pretrained(
        args.model, revision=args.revision
    )
    model = HFWhisperLensModel(hf_model, processor, model_id=args.model)
    hf_model.to(device)
    report = evaluate_records(model, lens, records)
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output is None:
        print(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
