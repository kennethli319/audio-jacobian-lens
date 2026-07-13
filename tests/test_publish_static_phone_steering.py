from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.publish_static_phone_steering import ASSET_VERSION, PUBLIC_ROUTE, publish

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_publish_static_phone_steering_builds_safe_public_route(tmp_path: Path) -> None:
    manifest = {
        "routes": {"detailed_cached_explorers": ["/audio-jacobian-lens/"]},
        "sha256": {},
        "interaction_boundary": "Cached reports only.",
        "media_policy": "Cleared inputs only.",
        "payload_policy": "Reduced reports only.",
    }
    (tmp_path / "site-manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    publish(source_root=ROOT, site_root=tmp_path, published_on="2026-07-13")

    page = (tmp_path / "steering" / "index.html").read_text(encoding="utf-8")
    assert 'name="robots" content="noindex,nofollow"' in page
    assert 'href="https://kennethli319.github.io/audio-jacobian-lens/steering/"' in page
    assert f'../assets/steering.js?v={ASSET_VERSION}' in page
    assert f'../assets/steering.css?v={ASSET_VERSION}' in page
    assert 'data-results-url="../data/phone-steering-results.json"' in page
    assert '<nav class="site-nav" aria-label="Model explorers">' in page
    assert "PREVIEW · STATIC REPLAY" in page
    assert 'href="../">ASR</a>' in page
    assert 'href="../speech/"' in page
    assert 'href="../tts/"' in page
    assert 'class="active" href="./" aria-current="page">Steering</a>' in page
    assert "Showcase</a>" not in page
    assert "<audio" not in page

    data = json.loads(
        (tmp_path / "data" / "phone-steering-results.json").read_text(
            encoding="utf-8"
        )
    )
    assert data["mode"] == "static_recorded_checkpoints"
    assert data["source"]["media_included"] is False

    published_manifest = json.loads(
        (tmp_path / "site-manifest.json").read_text(encoding="utf-8")
    )
    assert published_manifest["routes"]["recorded_interventions"] == [PUBLIC_ROUTE]
    assert published_manifest["publication_mode"] == "public_linked_noindex_review"
    for relative_path in (
        "assets/steering.css",
        "assets/steering.js",
        "data/phone-steering-results.json",
        "steering/index.html",
    ):
        assert published_manifest["sha256"][relative_path] == _sha256(
            tmp_path / relative_path
        )
