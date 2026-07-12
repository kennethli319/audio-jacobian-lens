from __future__ import annotations

import stat
from pathlib import Path

import pytest

from jlens.static_explorer_catalog import load_static_explorer_catalog
from scripts import materialize_static_audio_samples as materializer

ROOT = Path(__file__).resolve().parents[1]


def test_materializer_validates_real_flac_stream_info_and_hash() -> None:
    catalog = load_static_explorer_catalog(
        ROOT / "data" / "static_explorer_catalog_v2.json"
    )
    sample = catalog.audio_samples[0]
    value = (ROOT / "samples" / sample.filename).read_bytes()

    assert materializer._flac_stream_info(value) == (16_000, 1, 16, 32_640)
    assert materializer._validated_audio_bytes(value, sample) == value


def test_materializer_extracts_and_reuses_a_pinned_parquet(
    tmp_path: Path,
) -> None:
    pyarrow = pytest.importorskip("pyarrow")
    parquet = pytest.importorskip("pyarrow.parquet")
    catalog = load_static_explorer_catalog(
        ROOT / "data" / "static_explorer_catalog_v2.json"
    )
    sample = catalog.audio_samples[0]
    value = (ROOT / "samples" / sample.filename).read_bytes()
    table = pyarrow.Table.from_pylist(
        [
            {
                "id": sample.utterance_id,
                "audio": {"bytes": value, "path": sample.filename},
            }
        ]
    )
    parquet_path = tmp_path / "fixture.parquet"
    parquet.write_table(table, parquet_path)
    output_dir = tmp_path / "audio"

    written = materializer.materialize_samples(
        catalog=catalog,
        parquet_path=parquet_path,
        output_dir=output_dir,
        selected_slugs={sample.slug},
        force=False,
    )
    reused = materializer.materialize_samples(
        catalog=catalog,
        parquet_path=parquet_path,
        output_dir=output_dir,
        selected_slugs={sample.slug},
        force=False,
    )

    assert written == (output_dir / sample.filename,)
    assert reused == ()
    assert (output_dir / sample.filename).read_bytes() == value
    assert stat.S_IMODE((output_dir / sample.filename).stat().st_mode) == 0o644
