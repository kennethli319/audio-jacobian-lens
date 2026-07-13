"""Regression tests for the Hugging Face Docker Space entrypoint."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = ROOT / "docker" / "entrypoint.sh"
DOCKERFILE = ROOT / "Dockerfile"
MODEL_REVISION = "87c7102498dcde7456f24cfd30239ca606ed9063"


def _make_executable(path: Path, source: str) -> None:
    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)


def _run_entrypoint(
    tmp_path: Path, **variables: str
) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    args_file = tmp_path / "audio-jlens-args.txt"
    _make_executable(
        tmp_path / "audio-jlens",
        "#!/bin/sh\n"
        ': > "$ARGS_FILE"\n'
        'for arg in "$@"; do printf "%s\\n" "$arg" >> "$ARGS_FILE"; done\n',
    )
    environment = {
        "ARGS_FILE": str(args_file),
        "HOME": str(tmp_path),
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "PORT": "7860",
        **variables,
    }
    result = subprocess.run(
        ["/bin/sh", str(ENTRYPOINT)],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    arguments = (
        args_file.read_text(encoding="utf-8").splitlines()
        if args_file.exists()
        else []
    )
    return result, arguments


def test_space_image_installs_cpu_only_torch() -> None:
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")

    assert "--torch-backend cpu" in dockerfile
    assert "--prune torch" in dockerfile
    assert "nvidia-" not in dockerfile


def test_entrypoint_passes_pinned_model_and_local_lens(tmp_path: Path) -> None:
    lens_path = tmp_path / "combined lens.pt"
    lens_path.touch()

    result, arguments = _run_entrypoint(
        tmp_path,
        JLENS_DEVICE="cpu",
        JLENS_LENS_PATH=str(lens_path),
        JLENS_MODEL_REVISION=MODEL_REVISION,
    )

    assert result.returncode == 0, result.stderr
    assert arguments == [
        "--host",
        "0.0.0.0",
        "--port",
        "7860",
        "--model",
        "openai/whisper-tiny.en",
        "--device",
        "cpu",
        "--revision",
        MODEL_REVISION,
        "--lens",
        str(lens_path),
    ]
    assert "model-backed analysis" in result.stdout


def test_entrypoint_downloads_private_hub_lens_before_start(tmp_path: Path) -> None:
    downloaded_lens = tmp_path / "downloaded.pt"
    downloaded_lens.touch()
    hf_args_file = tmp_path / "hf-args.txt"
    _make_executable(
        tmp_path / "hf",
        "#!/bin/sh\n"
        'printf "%s\\n" "$@" > "$HF_ARGS_FILE"\n'
        'printf "%s\\n" "$DOWNLOADED_LENS_PATH"\n',
    )

    result, arguments = _run_entrypoint(
        tmp_path,
        DOWNLOADED_LENS_PATH=str(downloaded_lens),
        HF_ARGS_FILE=str(hf_args_file),
        JLENS_LENS_FILENAME="whisper.pt",
        JLENS_LENS_REPO_ID="example/private-lens",
        JLENS_LENS_REVISION="abc123",
        JLENS_MODEL_REVISION=MODEL_REVISION,
    )

    assert result.returncode == 0, result.stderr
    assert hf_args_file.read_text(encoding="utf-8").splitlines() == [
        "download",
        "example/private-lens",
        "whisper.pt",
        "--repo-type",
        "model",
        "--revision",
        "abc123",
        "--quiet",
    ]
    assert arguments[-2:] == ["--lens", str(downloaded_lens)]


def test_entrypoint_downloads_phone_signatures_and_enables_asr_only(
    tmp_path: Path,
) -> None:
    downloaded_lens = tmp_path / "combined.pt"
    downloaded_phones = tmp_path / "phones.pt"
    downloaded_lens.touch()
    downloaded_phones.touch()
    hf_args_file = tmp_path / "hf-args.txt"
    _make_executable(
        tmp_path / "hf",
        "#!/bin/sh\n"
        'printf "%s\\n" "$@" >> "$HF_ARGS_FILE"\n'
        'case "$3" in\n'
        '  combined.pt) printf "%s\\n" "$DOWNLOADED_LENS_PATH" ;;\n'
        '  phones.pt) printf "%s\\n" "$DOWNLOADED_PHONES_PATH" ;;\n'
        'esac\n',
    )

    result, arguments = _run_entrypoint(
        tmp_path,
        DOWNLOADED_LENS_PATH=str(downloaded_lens),
        DOWNLOADED_PHONES_PATH=str(downloaded_phones),
        HF_ARGS_FILE=str(hf_args_file),
        JLENS_ASR_ONLY="true",
        JLENS_LENS_FILENAME="combined.pt",
        JLENS_LENS_REPO_ID="example/private-lens",
        JLENS_LENS_REVISION="abc123",
        JLENS_PHONE_SIGNATURES_FILENAME="phones.pt",
    )

    assert result.returncode == 0, result.stderr
    assert arguments[-5:] == [
        "--lens",
        str(downloaded_lens),
        "--phone-signatures",
        str(downloaded_phones),
        "--asr-only",
    ]
    assert hf_args_file.read_text(encoding="utf-8").splitlines().count(
        "download"
    ) == 2


def test_entrypoint_rejects_invalid_asr_only_value(tmp_path: Path) -> None:
    result, arguments = _run_entrypoint(tmp_path, JLENS_ASR_ONLY="sometimes")

    assert result.returncode == 64
    assert not arguments
    assert "JLENS_ASR_ONLY must be true or false" in result.stderr


def test_entrypoint_enables_bounded_analysis_queue(tmp_path: Path) -> None:
    result, arguments = _run_entrypoint(
        tmp_path,
        JLENS_ANALYSIS_QUEUE_CAPACITY="4",
        JLENS_ANALYSIS_QUEUE_INITIAL_SECONDS="4.0",
    )

    assert result.returncode == 0, result.stderr
    assert arguments[-4:] == [
        "--analysis-queue-capacity",
        "4",
        "--analysis-queue-initial-seconds",
        "4.0",
    ]


def test_entrypoint_rejects_partial_hub_lens_configuration(tmp_path: Path) -> None:
    result, arguments = _run_entrypoint(
        tmp_path,
        JLENS_LENS_REPO_ID="example/private-lens",
    )

    assert result.returncode == 64
    assert not arguments
    assert "must be set together" in result.stderr
