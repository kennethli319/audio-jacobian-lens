from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.publish_static_phone_steering import (
    INTEGRATED_ROUTE,
    PUBLIC_CANONICAL,
    PUBLIC_ROUTE,
    publish,
)

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_publish_static_phone_steering_builds_safe_public_route(tmp_path: Path) -> None:
    manifest = {
        "routes": {"detailed_cached_explorers": ["/audio-jacobian-lens/"]},
        "sha256": {},
        "interaction_boundary": (
            "Cached reports only. The steering route replays only saved checkpoints "
            "and never performs inference or interpolates an unmeasured intervention."
        ),
        "media_policy": "Cleared inputs only.",
        "payload_policy": "Reduced reports only.",
    }
    (tmp_path / "site-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    publish(source_root=ROOT, site_root=tmp_path, published_on="2026-07-13")

    page = (tmp_path / "steering" / "index.html").read_text(encoding="utf-8")
    assert 'name="robots" content="noindex,nofollow"' in page
    assert f'href="{PUBLIC_CANONICAL}"' in page
    assert "window.location.replace" in page
    assert 'get("target")' in page
    assert "`&condition=${target}`" in page
    assert 'href="../?sample=asr-laurel-yanny"' in page
    assert "assets/steering" not in page
    assert "phone-steering-results.json" not in page
    assert 'class="site-nav"' not in page
    assert "<audio" not in page

    data = json.loads(
        (tmp_path / "data" / "phone-steering-results.json").read_text(encoding="utf-8")
    )
    assert data["mode"] == "static_recorded_checkpoints"
    assert data["source"]["media_included"] is False

    published_manifest = json.loads(
        (tmp_path / "site-manifest.json").read_text(encoding="utf-8")
    )
    assert published_manifest["routes"]["recorded_interventions"] == [INTEGRATED_ROUTE]
    assert published_manifest["routes"]["retired_redirects"] == [PUBLIC_ROUTE]
    assert published_manifest["publication_mode"] == "public_static_cached_explorers"
    assert "TTS" not in published_manifest["description"]
    assert "ASR Audio 10" in published_manifest["description"]
    assert "The steering route" not in published_manifest["interaction_boundary"]
    assert "The archived steering payload" in published_manifest["payload_policy"]
    assert "Audio S7 MP3" in published_manifest["media_policy"]
    assert "linked but not embedded" not in published_manifest["media_policy"]
    for relative_path in (
        "assets/steering.css",
        "assets/steering.js",
        "data/phone-steering-results.json",
        "steering/index.html",
    ):
        assert published_manifest["sha256"][relative_path] == _sha256(
            tmp_path / relative_path
        )
