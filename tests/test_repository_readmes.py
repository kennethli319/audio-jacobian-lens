from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_github_readme_starts_with_project_content() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert readme.startswith("# Audio Jacobian Lens\n")
    assert not readme.startswith("---\n")
    assert "sdk: docker" not in readme


def test_hugging_face_readme_retains_space_configuration() -> None:
    readme = (ROOT / "deploy/huggingface/README.md").read_text(encoding="utf-8")

    assert readme.startswith("---\ntitle: Audio Jacobian Lens\n")
    assert "\nsdk: docker\n" in readme
    assert "\napp_port: 7860\n" in readme
    assert "\n  - openai/whisper-tiny.en\n" in readme
