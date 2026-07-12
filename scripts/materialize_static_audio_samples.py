#!/usr/bin/env python3
"""Materialize the pinned static-explorer LibriSpeech FLAC files.

The catalog pins both the Hugging Face dataset revision and every output-file
hash.  Existing matching files are retained; mismatches fail closed unless
``--force`` is explicitly supplied.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download

try:
    from jlens.static_explorer_catalog import (
        StaticAudioSample,
        StaticExplorerCatalog,
        load_static_explorer_catalog,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from jlens.static_explorer_catalog import (  # noqa: E402
        StaticAudioSample,
        StaticExplorerCatalog,
        load_static_explorer_catalog,
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _flac_stream_info(value: bytes) -> tuple[int, int, int, int]:
    """Return sample rate, channels, bits per sample, and total samples."""

    if len(value) < 42 or value[:4] != b"fLaC":
        raise ValueError("payload is not a FLAC file")
    metadata_header = value[4:8]
    block_type = metadata_header[0] & 0x7F
    block_length = int.from_bytes(metadata_header[1:4], "big")
    if block_type != 0 or block_length != 34 or len(value) < 8 + block_length:
        raise ValueError("FLAC has no standard 34-byte STREAMINFO block")
    stream_info = value[8:42]
    packed = int.from_bytes(stream_info[10:18], "big")
    sample_rate = (packed >> 44) & 0xFFFFF
    channels = ((packed >> 41) & 0x7) + 1
    bits_per_sample = ((packed >> 36) & 0x1F) + 1
    total_samples = packed & ((1 << 36) - 1)
    if sample_rate <= 0 or total_samples <= 0:
        raise ValueError("FLAC STREAMINFO has invalid sample coordinates")
    return sample_rate, channels, bits_per_sample, total_samples


def _validated_audio_bytes(value: Any, sample: StaticAudioSample) -> bytes:
    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise ValueError(f"{sample.utterance_id} has no embedded audio bytes")
    audio = bytes(value)
    digest = _sha256_bytes(audio)
    if digest != sample.sha256:
        raise ValueError(
            f"{sample.utterance_id} hash mismatch: expected {sample.sha256}, "
            f"found {digest}"
        )
    sample_rate, channels, bits_per_sample, total_samples = _flac_stream_info(audio)
    if (sample_rate, channels, bits_per_sample) != (16_000, 1, 16):
        raise ValueError(f"{sample.utterance_id} is not 16 kHz mono PCM16 FLAC")
    duration = total_samples / sample_rate
    if not math.isclose(
        duration, sample.duration_seconds, rel_tol=0.0, abs_tol=1 / sample_rate
    ):
        raise ValueError(
            f"{sample.utterance_id} duration mismatch: expected "
            f"{sample.duration_seconds}, found {duration}"
        )
    return audio


def _parquet_rows(path: Path) -> dict[str, Mapping[str, Any]]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError(
            "materializing the static audio requires pyarrow; install the "
            "project's dev dependencies"
        ) from error
    table = parquet.read_table(path, columns=["id", "audio"])
    rows: dict[str, Mapping[str, Any]] = {}
    for value in table.to_pylist():
        if not isinstance(value, Mapping):
            raise ValueError("Parquet row must be an object")
        utterance_id = value.get("id")
        if not isinstance(utterance_id, str) or not utterance_id:
            raise ValueError("Parquet row has no utterance ID")
        if utterance_id in rows:
            raise ValueError(f"duplicate Parquet utterance ID: {utterance_id}")
        rows[utterance_id] = value
    return rows


def _atomic_write(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        path.chmod(0o644)
    finally:
        temporary_path.unlink(missing_ok=True)


def materialize_samples(
    *,
    catalog: StaticExplorerCatalog,
    parquet_path: Path,
    output_dir: Path,
    selected_slugs: set[str],
    force: bool,
) -> tuple[Path, ...]:
    rows = _parquet_rows(parquet_path)
    written: list[Path] = []
    for sample in catalog.audio_samples:
        if sample.slug not in selected_slugs:
            continue
        row = rows.get(sample.utterance_id)
        if row is None:
            raise ValueError(f"pinned Parquet has no utterance {sample.utterance_id}")
        audio_field = row.get("audio")
        if not isinstance(audio_field, Mapping):
            raise ValueError(f"{sample.utterance_id} has no audio object")
        audio = _validated_audio_bytes(audio_field.get("bytes"), sample)
        target = output_dir / sample.filename
        if target.exists():
            existing_hash = _sha256_file(target)
            if existing_hash == sample.sha256:
                target.chmod(0o644)
                print(f"kept {target} ({existing_hash})")
                continue
            if not force:
                raise ValueError(
                    f"refusing to replace mismatched {target}; pass --force"
                )
        _atomic_write(target, audio)
        written.append(target)
        print(f"wrote {target} ({len(audio):,} bytes; {sample.sha256})")
    return tuple(written)


def _selected_slugs(
    catalog: StaticExplorerCatalog, requested: list[str] | None
) -> set[str]:
    available = {sample.slug for sample in catalog.audio_samples}
    if not requested:
        return available
    selected: set[str] = set()
    for group in requested:
        for raw_value in group.split(","):
            value = raw_value.strip()
            if value not in available:
                raise ValueError(f"unknown static audio sample: {value}")
            selected.add(value)
    if not selected:
        raise ValueError("--only did not select any static audio samples")
    return selected


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("data/static_explorer_catalog_v2.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("samples"))
    parser.add_argument(
        "--parquet",
        type=Path,
        help="Use this pinned Parquet instead of the Hugging Face cache/download.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Fail rather than downloading when the pinned Parquet is not cached.",
    )
    parser.add_argument(
        "--only",
        action="append",
        metavar="SLUG",
        help="Materialize only these comma-separated slugs; repeat as needed.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace a destination file whose hash differs from the catalog.",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    catalog = load_static_explorer_catalog(args.catalog)
    parquet_path = args.parquet
    if parquet_path is None:
        parquet_path = Path(
            hf_hub_download(
                repo_id=catalog.audio_source.dataset_id,
                repo_type="dataset",
                filename=catalog.audio_source.parquet_path,
                revision=catalog.audio_source.dataset_revision,
                local_files_only=args.local_files_only,
            )
        )
    if not parquet_path.is_file():
        raise FileNotFoundError(f"pinned Parquet not found: {parquet_path}")
    materialize_samples(
        catalog=catalog,
        parquet_path=parquet_path,
        output_dir=args.output_dir,
        selected_slugs=_selected_slugs(catalog, args.only),
        force=args.force,
    )


if __name__ == "__main__":
    main()
