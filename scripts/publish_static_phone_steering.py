#!/usr/bin/env python3
"""Archive the sanitized steering payload and retire its standalone public page."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

PUBLIC_ROUTE = "/audio-jacobian-lens/steering/"
INTEGRATED_ROUTE = "/audio-jacobian-lens/?sample=asr-laurel-yanny"
PUBLIC_CANONICAL = (
    "https://kennethli319.github.io/audio-jacobian-lens/?sample=asr-laurel-yanny"
)
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


def _redirect_html() -> str:
    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <meta name="robots" content="noindex,nofollow">
    <meta name="description" content="This retired checkpoint page now opens the integrated Laurel/Yanny ASR replay.">
    <link rel="canonical" href="{PUBLIC_CANONICAL}">
    <title>Opening the Laurel/Yanny ASR replay</title>
    <script>
      (() => {{
        const target = new URLSearchParams(window.location.search).get("target");
        const condition = ["yanny", "laurel"].includes(target) ? `&condition=${{target}}` : "";
        window.location.replace(`../?sample=asr-laurel-yanny${{condition}}`);
      }})();
    </script>
    <noscript><meta http-equiv="refresh" content="0; url=../?sample=asr-laurel-yanny"></noscript>
  </head>
  <body>
    <p>The standalone checkpoint page has moved to the
      <a href="../?sample=asr-laurel-yanny">integrated Laurel/Yanny ASR replay</a>.
    </p>
  </body>
</html>
"""


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

    destination_html = site_root / "steering" / "index.html"
    destination_html.parent.mkdir(parents=True, exist_ok=True)
    destination_html.write_text(_redirect_html(), encoding="utf-8")
    published_paths.append("steering/index.html")

    manifest_path = site_root / "site-manifest.json"
    manifest = _load_object(manifest_path)
    routes = manifest.setdefault("routes", {})
    if not isinstance(routes, dict):
        raise ValueError("site manifest routes must be an object")
    routes["recorded_interventions"] = [INTEGRATED_ROUTE]
    routes["retired_redirects"] = [PUBLIC_ROUTE]
    manifest["published_on"] = published_on
    manifest["publication_mode"] = "public_static_cached_explorers"
    manifest["description"] = (
        "Public static ASR and speech-to-speech Audio Jacobian Lens explorers, "
        "with the recorded phone-steering experiment integrated into ASR Audio 10."
    )
    interaction = str(manifest.get("interaction_boundary") or "").rstrip()
    retired_copy = (
        " The steering route replays only saved checkpoints and never performs"
        " inference or interpolates an unmeasured intervention."
    )
    manifest["interaction_boundary"] = interaction.replace(retired_copy, "")
    manifest["media_policy"] = (
        "The detailed explorers contain ten ASR reports and ten speech-to-speech "
        "reports. Their referenced media comprises ten CC BY 4.0 LibriSpeech "
        "FLACs plus the unchanged Laurel/Yanny Audio S7 MP3 reproduced from "
        "Hans Rutger Bosker's demo using that page's CC BY 4.0 notice. Generated "
        "LFM and Chatterbox audio remains excluded pending derived-output review."
    )
    payload = str(manifest.get("payload_policy") or "").rstrip()
    active_payload_copy = (
        " The steering payload contains only recorded schedules, scalar "
        "coefficients, bounded candidates, output metrics, and compact controls; "
        "it excludes residuals, prototype tensors, private paths, and audio."
    )
    payload = payload.replace(active_payload_copy, "")
    payload_addition = (
        " The archived steering payload contains only recorded schedules, scalar "
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
    print(f"published retired steering redirect to {args.site_root / 'steering'}")


if __name__ == "__main__":
    main()
