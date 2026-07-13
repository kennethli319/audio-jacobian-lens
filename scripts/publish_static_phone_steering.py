#!/usr/bin/env python3
"""Publish the sanitized phone-steering replay into the personal static site."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

PUBLIC_ROUTE = "/audio-jacobian-lens/steering/"
PUBLIC_CANONICAL = "https://kennethli319.github.io/audio-jacobian-lens/steering/"
ASSET_VERSION = "20260713-2"
PUBLISHED_FILES = {
    "web/steering.css": "assets/steering.css",
    "web/steering.js": "assets/steering.js",
    "data/static_phone_steering_v1.json": "data/phone-steering-results.json",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _replace_once(body: str, old: str, new: str) -> str:
    if body.count(old) != 1:
        raise ValueError(f"expected exactly one source marker: {old!r}")
    return body.replace(old, new, 1)


def _public_html(source: str) -> str:
    body = _replace_once(
        source,
        '<meta name="description" content="Interactive replay of recorded Whisper phone-signature steering experiments." />',
        (
            '<meta name="description" content="Interactive replay of recorded '
            'Whisper phone-signature steering experiments." />\n'
            '    <meta name="robots" content="noindex,nofollow" />\n'
            f'    <link rel="canonical" href="{PUBLIC_CANONICAL}" />'
        ),
    )
    body = _replace_once(
        body,
        '    <link rel="icon" href="./favicon.svg" type="image/svg+xml" />\n',
        "",
    )
    body = _replace_once(
        body,
        '<link rel="stylesheet" href="./steering.css?v=1" />',
        f'<link rel="stylesheet" href="../assets/steering.css?v={ASSET_VERSION}" />',
    )
    body = _replace_once(
        body,
        '<script src="./steering.js?v=1" defer></script>',
        f'<script src="../assets/steering.js?v={ASSET_VERSION}" defer></script>',
    )
    body = _replace_once(
        body,
        'data-results-url="./data/phone-steering-results.json"',
        'data-results-url="../data/phone-steering-results.json"',
    )
    body = _replace_once(
        body,
        '<a class="brand" href="./" aria-label="Audio Jacobian Lens home">',
        '<a class="brand" href="../" aria-label="Audio Jacobian Lens home">',
    )
    body = _replace_once(
        body,
        (
            '<nav class="site-nav" aria-label="Model explorers">\n'
            '        <a href="./">ASR</a>\n'
            '        <a href="./">Speech</a>\n'
            '        <a href="./chatterbox">TTS</a>\n'
            '        <a class="active" href="./steering.html" aria-current="page">Steering</a>\n'
            '      </nav>'
        ),
        (
            '<nav class="site-nav" aria-label="Model explorers">\n'
            '        <a href="../">ASR</a>\n'
            '        <a href="../speech/">Speech</a>\n'
            '        <a class="active" href="./" aria-current="page">Steering</a>\n'
            '      </nav>'
        ),
    )
    return body


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def publish(*, source_root: Path, site_root: Path, published_on: str) -> None:
    source_root = source_root.resolve()
    site_root = site_root.resolve()
    canonical_data = source_root / "data" / "static_phone_steering_v1.json"
    runtime_data = source_root / "web" / "data" / "phone-steering-results.json"
    if canonical_data.read_bytes() != runtime_data.read_bytes():
        raise ValueError("canonical and local-runtime steering data disagree")

    published_paths: list[str] = []
    for source_relative, destination_relative in PUBLISHED_FILES.items():
        source_path = source_root / source_relative
        destination_path = site_root / destination_relative
        if not source_path.is_file():
            raise ValueError(f"missing steering source file: {source_path}")
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, destination_path)
        published_paths.append(destination_relative)

    source_html = (source_root / "web" / "steering.html").read_text(
        encoding="utf-8"
    )
    destination_html = site_root / "steering" / "index.html"
    destination_html.parent.mkdir(parents=True, exist_ok=True)
    destination_html.write_text(_public_html(source_html), encoding="utf-8")
    published_paths.append("steering/index.html")

    manifest_path = site_root / "site-manifest.json"
    manifest = _load_object(manifest_path)
    routes = manifest.setdefault("routes", {})
    if not isinstance(routes, dict):
        raise ValueError("site manifest routes must be an object")
    routes["recorded_interventions"] = [PUBLIC_ROUTE]
    manifest["published_on"] = published_on
    manifest["publication_mode"] = "public_linked_noindex_review"
    manifest["description"] = (
        "Public static ASR, speech-to-speech, and recorded phone-steering "
        "Audio Jacobian Lens explorers."
    )
    interaction = str(manifest.get("interaction_boundary") or "").rstrip()
    addition = (
        " The steering route replays only saved checkpoints and never performs "
        "inference or interpolates an unmeasured intervention."
    )
    if addition.strip() not in interaction:
        manifest["interaction_boundary"] = interaction + addition
    media = str(manifest.get("media_policy") or "").rstrip()
    media_addition = (
        " The external Laurel/Yanny source recording is linked but not embedded "
        "or redistributed."
    )
    if media_addition.strip() not in media:
        manifest["media_policy"] = media + media_addition
    payload = str(manifest.get("payload_policy") or "").rstrip()
    payload_addition = (
        " The steering payload contains only recorded schedules, scalar "
        "coefficients, bounded candidates, output metrics, and compact controls; "
        "it excludes residuals, prototype tensors, private paths, and audio."
    )
    if payload_addition.strip() not in payload:
        manifest["payload_policy"] = payload + payload_addition
    recorded_hashes = manifest.setdefault("sha256", {})
    if not isinstance(recorded_hashes, dict):
        raise ValueError("site manifest sha256 must be an object")
    for relative_path in published_paths:
        recorded_hashes[relative_path] = _sha256(site_root / relative_path)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--site-root",
        required=True,
        type=Path,
        help="The audio-jacobian-lens directory inside the personal-site checkout.",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--published-on", default="2026-07-13")
    return parser


def main() -> None:
    args = _parser().parse_args()
    publish(
        source_root=args.source_root,
        site_root=args.site_root,
        published_on=args.published_on,
    )
    print(f"published recorded steering replay to {args.site_root / 'steering'}")


if __name__ == "__main__":
    main()
