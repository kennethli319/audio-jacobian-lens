# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Fit a projected text-head Jacobian lens for local MLX LFM2.5 Audio."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from jlens.audio_io import decode_audio_bytes
from jlens.mlx_fitting import fit_mlx_lfm_language_lens
from jlens.mlx_lfm import (
    DEFAULT_LFM_MODEL_ID,
    DEFAULT_LFM_MODEL_REVISION,
    DEFAULT_LFM_SYSTEM_PROMPT,
    LFMGenerationConfig,
    MLXLFMModel,
)


def _layer_list(value: str) -> list[int]:
    try:
        layers = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "layers must be a comma-separated list of integers"
        ) from exc
    if not layers:
        raise argparse.ArgumentTypeError("select at least one source layer")
    return layers


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit a projected language-backbone J-lens for the pinned local "
            "LFM2.5 speech-to-speech model"
        )
    )
    parser.add_argument("audio", nargs="+", type=Path, help="short fitting clips")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_LFM_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_LFM_MODEL_REVISION)
    parser.add_argument(
        "--source-layers", type=_layer_list, default=[0, 4, 8, 12, 14]
    )
    parser.add_argument("--target-layer", type=int, default=15)
    parser.add_argument("--projection-dim", type=int, default=512)
    parser.add_argument("--projection-seed", type=int, default=0)
    parser.add_argument(
        "--target-reduction", choices=("sum", "mean"), default="sum"
    )
    parser.add_argument("--max-new-tokens", type=int, default=72)
    parser.add_argument("--system-prompt", default=DEFAULT_LFM_SYSTEM_PROMPT)
    parser.add_argument(
        "--center",
        action="store_true",
        help="fit affine activation means; off by default for same-stream replication",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    for path in args.audio:
        if not path.is_file():
            raise SystemExit(f"audio file does not exist: {path}")
    generation_config = LFMGenerationConfig(
        system_prompt=args.system_prompt,
        max_new_tokens=args.max_new_tokens,
    )
    model = MLXLFMModel.from_pretrained(
        args.model,
        revision=args.revision,
        generation_config=generation_config,
    )
    examples = []
    for index, path in enumerate(args.audio):
        decoded = decode_audio_bytes(
            path.read_bytes(), target_rate=model.input_sample_rate
        )
        example = model.prepare_audio(
            decoded.waveform,
            sampling_rate=decoded.sampling_rate,
            duration_seconds=decoded.duration_seconds,
            decode_output_audio=False,
        )
        examples.append(example)
        print(
            json.dumps(
                {
                    "event": "prepared",
                    "index": index,
                    "file": str(path),
                    "generated_text": example.generated_text,
                    "target_tokens": len(example.target_token_ids),
                }
            ),
            file=sys.stderr,
            flush=True,
        )

    def report(event):
        if event["event"] != "probe" or event["probe_index"] in {
            0,
            event["probe_count"] - 1,
        }:
            print(json.dumps(event), file=sys.stderr, flush=True)

    lens = fit_mlx_lfm_language_lens(
        model,
        examples,
        source_layers=args.source_layers,
        target_layer=args.target_layer,
        projection_dim=args.projection_dim,
        projection_seed=args.projection_seed,
        target_reduction=args.target_reduction,
        center=args.center,
        progress=report,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    lens.save(args.output)
    print(
        json.dumps(
            {
                "artifact": str(args.output),
                "bytes": args.output.stat().st_size,
                "model_id": model.model_id,
                "model_revision": model.model_revision,
                "model_fingerprint": model.fingerprint,
                "source_layers": lens.source_layers,
                "target_layer": lens.metadata["target_layer"],
                "projection_method": lens.projection_method,
                "projection_dim": lens.projection_dim,
                "examples": lens.n_examples,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
