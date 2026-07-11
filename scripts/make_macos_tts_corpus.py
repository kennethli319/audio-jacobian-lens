#!/usr/bin/env python3
"""Generate a small local ASR fitting corpus with macOS `say`.

The audio and manifest are intended for plumbing and convergence smoke tests,
not for making scientific claims about natural-speech generalization.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

SENTENCES = [
    "A bright red kite drifted above the quiet park.",
    "Please leave the blue notebook beside the kitchen clock.",
    "Seven small boats crossed the harbor before sunrise.",
    "The museum closes early on cold winter evenings.",
    "Fresh coffee and warm bread filled the little cafe.",
    "Morgan counted every silver coin twice.",
    "A sudden rainstorm interrupted the outdoor concert.",
    "The curious child asked why the moon changes shape.",
    "Our train arrives at the central station around noon.",
    "Three green apples rolled out of the paper bag.",
    "The old radio played a familiar jazz melody.",
    "Someone left a striped umbrella near the front door.",
    "Soft footsteps echoed through the empty hallway.",
    "The baker carefully measured sugar, flour, and salt.",
    "Clouds gathered slowly above the distant mountains.",
    "Her final answer surprised everyone in the room.",
]


def available_voices(say: str) -> set[str]:
    result = subprocess.run(
        [say, "-v", "?"], capture_output=True, text=True, check=True
    )
    return {
        line.split()[0]
        for line in result.stdout.splitlines()
        if line.strip() and len(line.split()) >= 2
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/tts_corpus"))
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--voices", default="Samantha,Alex,Moira")
    parser.add_argument("--rate", type=int, default=175)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    say = shutil.which("say")
    if say is None:
        raise SystemExit("macOS `say` was not found")
    if not 1 <= args.count <= len(SENTENCES):
        raise SystemExit(f"--count must be between 1 and {len(SENTENCES)}")
    installed = available_voices(say)
    requested = [voice.strip() for voice in args.voices.split(",") if voice.strip()]
    voices = [voice for voice in requested if voice in installed]
    if not voices:
        voices = [""]

    audio_dir = args.output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.jsonl"
    records = []
    for index, text in enumerate(SENTENCES[: args.count]):
        voice = voices[index % len(voices)]
        slug = f"{index:03d}_{(voice or 'default').lower()}.aiff"
        output = audio_dir / slug
        command = [say]
        if voice:
            command.extend(["-v", voice])
        command.extend(["-r", str(args.rate), "-o", str(output), text])
        if args.overwrite or not output.exists():
            subprocess.run(command, check=True)
        records.append(
            {
                "audio": str(Path("audio") / slug),
                "text": text,
                "voice": voice or "system-default",
                "source": "macos-say-synthetic",
            }
        )

    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"wrote {len(records)} clips and {manifest_path}")


if __name__ == "__main__":
    main()
