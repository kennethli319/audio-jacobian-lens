# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Fit a projected average-Jacobian lens over Chatterbox T3 speech positions."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import torch

from jlens.chatterbox_fitting import (
    ChatterboxLensExample,
    example_manifest_record,
    fit_mlx_chatterbox_speech_lens,
)
from jlens.mlx_chatterbox import (
    DEFAULT_CHATTERBOX_LAYERS,
    DEFAULT_CHATTERBOX_MODEL_ID,
    DEFAULT_CHATTERBOX_MODEL_REVISION,
    DEFAULT_S3_TOKENIZER_ID,
    DEFAULT_S3_TOKENIZER_REVISION,
    ChatterboxGenerationConfig,
    MLXChatterboxModel,
)


def _layer_list(value: str) -> list[int]:
    try:
        layers = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    except ValueError as exc:
        raise argparse.ArgumentTypeError("layers must be comma-separated integers") from exc
    if not layers:
        raise argparse.ArgumentTypeError("select at least one source layer")
    return layers


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit a corpus-averaged projected J-lens from intermediate T3 "
            "speech-prediction residuals to the final T3 speech-code head"
        )
    )
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/chatterbox_fit_prompts.jsonl"),
        help="JSONL records with id, split, and text fields",
    )
    parser.add_argument("--split", default="fit")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model", default=DEFAULT_CHATTERBOX_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_CHATTERBOX_MODEL_REVISION)
    parser.add_argument("--s3-tokenizer", default=DEFAULT_S3_TOKENIZER_ID)
    parser.add_argument(
        "--s3-tokenizer-revision", default=DEFAULT_S3_TOKENIZER_REVISION
    )
    parser.add_argument("--generation-seed", type=int, default=7)
    parser.add_argument("--max-speech-tokens", type=int, default=64)
    parser.add_argument("--max-speech-positions", type=int, default=48)
    parser.add_argument(
        "--source-layers",
        type=_layer_list,
        default=list(DEFAULT_CHATTERBOX_LAYERS),
    )
    parser.add_argument("--target-layer", type=int, default=-1)
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--projection-seed", type=int, default=29)
    parser.add_argument(
        "--target-reduction", choices=("sum", "mean"), default="sum"
    )
    parser.add_argument(
        "--center",
        action="store_true",
        help="Fit an affine centered variant rather than the paper-style linear map",
    )
    parser.add_argument(
        "--prepared-output",
        type=Path,
        default=None,
        help="Write exact generated token trajectories (default: OUTPUT.examples.jsonl)",
    )
    return parser


def _manifest_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_records(
    path: Path, *, split: str, limit: int | None
) -> list[dict[str, str]]:
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    records: list[dict[str, str]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON on {path}:{line_number}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"record on {path}:{line_number} must be an object")
        if payload.get("split", "fit") != split:
            continue
        text = payload.get("text")
        record_id = payload.get("id", f"line-{line_number}")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"record on {path}:{line_number} has no non-empty text")
        if not isinstance(record_id, str) or not record_id:
            raise ValueError(f"record on {path}:{line_number} has an invalid id")
        records.append({"id": record_id, "text": text})
        if limit is not None and len(records) >= limit:
            break
    if not records:
        raise ValueError(f"manifest contains no records for split {split!r}")
    return records


def _write_prepared_examples(
    path: Path, examples: list[ChatterboxLensExample]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = "".join(
        json.dumps(example_manifest_record(example), sort_keys=True) + "\n"
        for example in examples
    )
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(rendered, encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = _parser().parse_args()
    if args.max_speech_positions <= 0:
        raise ValueError("max_speech_positions must be positive")
    records = _load_records(args.manifest, split=args.split, limit=args.limit)
    model = MLXChatterboxModel.from_pretrained(
        args.model,
        revision=args.revision,
        s3_tokenizer_id=args.s3_tokenizer,
        s3_tokenizer_revision=args.s3_tokenizer_revision,
        generation_config=ChatterboxGenerationConfig(
            seed=args.generation_seed,
            max_speech_tokens=args.max_speech_tokens,
        ),
    )

    import mlx.core as mx

    examples: list[ChatterboxLensExample] = []
    capture_started = time.perf_counter()
    for index, record in enumerate(records, 1):
        print(
            f"capture {index}/{len(records)} · {record['id']} · {record['text']}",
            flush=True,
        )
        run = model.capture_for_fitting(record["text"])
        examples.append(
            ChatterboxLensExample.from_captured_run(
                run,
                max_speech_positions=args.max_speech_positions,
                record_id=record["id"],
            )
        )
        del run
        mx.clear_cache()

    prepared_path = args.prepared_output or args.output.with_suffix(
        args.output.suffix + ".examples.jsonl"
    )
    _write_prepared_examples(prepared_path, examples)

    last_probe = -1

    def report(event: dict[str, Any]) -> None:
        nonlocal last_probe
        if event["event"] == "example_start":
            print(
                f"fit {event['example_index'] + 1}/{event['example_count']} · starting",
                flush=True,
            )
        elif event["event"] == "probe":
            probe = int(event["probe_index"]) + 1
            count = int(event["probe_count"])
            if probe == count or probe == 1 or probe - last_probe >= max(1, count // 8):
                print(f"  probe {probe}/{count}", flush=True)
                last_probe = probe
        elif event["event"] == "example_complete":
            print(
                f"  complete in {event['projection_seconds']:.2f}s · "
                f"{event['target_positions']} positions",
                flush=True,
            )

    lens = fit_mlx_chatterbox_speech_lens(
        model,
        examples,
        source_layers=args.source_layers,
        target_layer=args.target_layer,
        projection_dim=args.rank,
        projection_seed=args.projection_seed,
        target_reduction=args.target_reduction,
        center=args.center,
        artifact_metadata={
            "manifest_name": args.manifest.name,
            "manifest_sha256": _manifest_sha256(args.manifest),
            "split": args.split,
            "prepared_examples_name": prepared_path.name,
            "capture_seconds": time.perf_counter() - capture_started,
        },
        progress=report,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    lens.save(args.output, dtype=torch.float16)
    size = args.output.stat().st_size
    print(
        json.dumps(
            {
                "output": str(args.output),
                "bytes": size,
                "examples": lens.n_examples,
                "positions": lens.metadata.get("target_positions"),
                "layers": lens.source_layers,
                "rank": lens.projection_dim,
                "target_dim": lens.target_dim,
                "projection_seconds": lens.metadata.get("projection_seconds"),
                "examples_fingerprint": lens.metadata.get("examples_fingerprint"),
                "prepared_examples": str(prepared_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
