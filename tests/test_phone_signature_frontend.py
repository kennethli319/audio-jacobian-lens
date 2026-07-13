from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HTML = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
SCRIPT = (ROOT / "web" / "app.js").read_text(encoding="utf-8")


def test_phone_signature_view_is_opt_in_and_explains_score_semantics():
    assert 'id="encoder-phone-signature-toggle"' in HTML
    assert 'aria-pressed="false" disabled' in HTML
    assert "ranked by cosine similarity" in HTML
    assert "not probability, a framewise vote, a phoneme boundary" in HTML
    assert "local-only receptive field" in HTML


def test_phone_signature_view_requires_matching_metadata_and_complete_cells():
    availability = SCRIPT[
        SCRIPT.index("function phoneSignatureAvailable()") :
        SCRIPT.index("function phoneSignatureCandidateCount")
    ]
    assert "metadata.available === false" in availability
    assert "flattened.every" in availability
    assert "phoneSignaturesForCell(cell).length > 0" in availability


def test_phone_signature_view_pauses_only_the_encoder_character_filter():
    mode = SCRIPT[
        SCRIPT.index("function updateEncoderPhoneSignatureMode") :
        SCRIPT.index("function updateEncoderTokenLengthFilter")
    ]
    assert "encoderTokenLengthFilter.disabled" in mode
    assert "encoderMaxTokenLength.disabled" in mode
    assert "token-length reranking is paused" in mode
    assert "decoderTokenLengthFilter" not in mode


def test_phone_signature_view_labels_similarity_without_probability_language():
    inspector = SCRIPT[
        SCRIPT.index("function inspectTimelineCell") :
        SCRIPT.index("function inspectOutputHeadPosition")
    ]
    assert '"Cosine similarity to frozen phone prototype"' in inspector
    assert "It is not a model probability, phoneme confidence, framewise vote" in inspector
    assert "pooled 100 ms window" in inspector
    assert "renderPhoneSignatures(phoneSignatures, denominator)" in inspector


def test_synthetic_demo_exercises_phone_signature_toggle():
    demo = SCRIPT[SCRIPT.index("function buildDemoData") : SCRIPT.index("function createDemoAudio")]
    assert "phone_signatures:" in demo
    assert "phone_signature:" in demo
    assert 'score_kind: "cosine_similarity"' in demo
