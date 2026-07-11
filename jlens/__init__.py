# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Jacobian lens: fit and apply the average input-output Jacobian as a readout
of decoder-transformer residuals."""

from jlens._logging import configure_logging
from jlens.cross_lens import CrossJacobianLens
from jlens.fitting import fit, jacobian_for_prompt
from jlens.hf import HFLensModel, Layout, from_hf
from jlens.hooks import ActivationRecorder, DecoderResidualScheduleAdder, ResidualAdder
from jlens.lens import JacobianLens
from jlens.projected_lens import ProjectedCrossJacobianLens
from jlens.protocol import LensModel
from jlens.whisper import HFWhisperLensModel, WhisperLensInputs
from jlens.whisper_causal import (
    CandidateSequenceScore,
    DecoderIntervention,
    DecoderInterventionSchedule,
    EncoderIntervention,
    EncoderInterventionSchedule,
    WhisperCausalTrace,
    WhisperDecoderCausalTrace,
    candidate_text_token_ids,
    decoder_lens_contrast_direction,
    encoder_lens_contrast_direction,
    prepare_candidate_inputs,
    random_decoder_direction,
    random_encoder_direction,
    run_decoder_intervention_schedule,
    run_encoder_intervention,
    run_encoder_intervention_schedule,
    score_candidate_text,
    vocabulary_token_ids_starting_with,
)
from jlens.whisper_fitting import fit_whisper, jacobians_for_whisper_example
from jlens.whisper_lens import WhisperJacobianLens

__all__ = [
    "ActivationRecorder",
    "CandidateSequenceScore",
    "DecoderIntervention",
    "DecoderInterventionSchedule",
    "DecoderResidualScheduleAdder",
    "EncoderIntervention",
    "EncoderInterventionSchedule",
    "CrossJacobianLens",
    "HFLensModel",
    "HFWhisperLensModel",
    "JacobianLens",
    "Layout",
    "LensModel",
    "ProjectedCrossJacobianLens",
    "ResidualAdder",
    "WhisperCausalTrace",
    "WhisperDecoderCausalTrace",
    "WhisperJacobianLens",
    "WhisperLensInputs",
    "configure_logging",
    "candidate_text_token_ids",
    "decoder_lens_contrast_direction",
    "fit",
    "fit_whisper",
    "from_hf",
    "jacobian_for_prompt",
    "jacobians_for_whisper_example",
    "encoder_lens_contrast_direction",
    "prepare_candidate_inputs",
    "random_decoder_direction",
    "run_encoder_intervention",
    "run_encoder_intervention_schedule",
    "run_decoder_intervention_schedule",
    "random_encoder_direction",
    "score_candidate_text",
    "vocabulary_token_ids_starting_with",
]
