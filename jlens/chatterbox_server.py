# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""CLI for the dedicated local MLX Chatterbox Frame Trace server."""

from __future__ import annotations

import argparse
import ipaddress
import warnings

from jlens.mlx_chatterbox import (
    DEFAULT_CHATTERBOX_MODEL_ID,
    DEFAULT_CHATTERBOX_MODEL_REVISION,
    DEFAULT_S3_TOKENIZER_ID,
    DEFAULT_S3_TOKENIZER_REVISION,
    ChatterboxGenerationConfig,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve the local MLX Chatterbox Frame Trace explorer"
    )
    parser.add_argument("--model", default=DEFAULT_CHATTERBOX_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_CHATTERBOX_MODEL_REVISION)
    parser.add_argument("--s3-tokenizer", default=DEFAULT_S3_TOKENIZER_ID)
    parser.add_argument(
        "--s3-tokenizer-revision", default=DEFAULT_S3_TOKENIZER_REVISION
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-speech-tokens", type=int, default=160)
    parser.add_argument("--max-cached-runs", type=int, default=2)
    parser.add_argument(
        "--lens",
        default=None,
        help="Optional fitted projected T3 speech-code J-lens artifact",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--web-dir", default="web")
    return parser


def main() -> None:
    args = _parser().parse_args()
    from jlens.chatterbox_webapp import MLXChatterboxBackend
    from jlens.webapp import create_app

    backend = MLXChatterboxBackend.load(
        model_id=args.model,
        revision=args.revision,
        s3_tokenizer_id=args.s3_tokenizer,
        s3_tokenizer_revision=args.s3_tokenizer_revision,
        generation_config=ChatterboxGenerationConfig(
            seed=args.seed,
            max_speech_tokens=args.max_speech_tokens,
        ),
        max_cached_runs=args.max_cached_runs,
        lens_path=args.lens,
        top_k=args.top_k,
    )
    app = create_app(
        None,
        web_dir=args.web_dir,
        chatterbox_backend=backend,
    )
    import uvicorn

    try:
        is_loopback = ipaddress.ip_address(args.host).is_loopback
    except ValueError:
        is_loopback = args.host == "localhost"
    if not is_loopback:
        warnings.warn(
            "binding Chatterbox Frame Trace beyond loopback exposes expensive "
            "local synthesis; add authentication and rate limits",
            stacklevel=1,
        )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
