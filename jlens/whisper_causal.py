"""Causal encoder- and decoder-residual interventions for Whisper studies.

The encoder has no vocabulary distribution to edit directly. This module adds
matched-norm vectors to actual post-block residuals and reruns every downstream
computation. J-lens directions may propose a vector, but the measured result is
always the resulting model computation.
"""

from __future__ import annotations

from contextlib import ExitStack, nullcontext
from dataclasses import dataclass, replace
from typing import Any

import torch

from jlens.hooks import (
    ActivationRecorder,
    DecoderResidualScheduleAdder,
    ResidualAdder,
)
from jlens.whisper import WhisperLensInputs
from jlens.whisper_lens import WhisperJacobianLens


@dataclass(frozen=True)
class EncoderIntervention:
    """A matched-norm additive edit over a contiguous raw encoder span.

    ``strength`` is the per-position perturbation norm as a fraction of the
    baseline residual norm averaged over ``start_position:end_position``.
    """

    layer: int
    start_position: int
    end_position: int
    direction: torch.Tensor
    strength: float

    def make_delta(self, residual: torch.Tensor) -> torch.Tensor:
        """Construct a full-shape delta calibrated against ``residual``."""
        if residual.ndim != 3:
            raise ValueError("residual must have shape [batch, positions, width]")
        if not 0 <= self.start_position < self.end_position <= residual.shape[1]:
            raise ValueError("intervention positions are outside the residual span")
        if self.direction.ndim != 1 or self.direction.numel() != residual.shape[-1]:
            raise ValueError("direction must have one value per residual width")
        if self.strength < 0:
            raise ValueError("strength cannot be negative")

        direction = self.direction.to(device=residual.device, dtype=torch.float32)
        direction_norm = direction.norm()
        if not torch.isfinite(direction_norm) or direction_norm <= 0:
            raise ValueError("direction must have a finite, nonzero norm")
        reference = residual[:, self.start_position : self.end_position].float()
        reference_norm = reference.norm(dim=-1).mean()
        vector = direction / direction_norm * reference_norm * self.strength
        delta = torch.zeros_like(residual)
        delta[:, self.start_position : self.end_position] = vector.to(delta.dtype)
        return delta


@dataclass(frozen=True)
class EncoderInterventionSchedule:
    """One or more post-block encoder edits applied in forward order."""

    interventions: tuple[EncoderIntervention, ...]

    def __post_init__(self) -> None:
        if not self.interventions:
            raise ValueError("an intervention schedule cannot be empty")
        layers = [intervention.layer for intervention in self.interventions]
        if layers != sorted(layers):
            raise ValueError("schedule layers must be ordered")

    def make_deltas(
        self, baseline_encoder: dict[int, torch.Tensor]
    ) -> dict[int, torch.Tensor]:
        deltas: dict[int, torch.Tensor] = {}
        for intervention in self.interventions:
            delta = intervention.make_delta(baseline_encoder[intervention.layer])
            deltas[intervention.layer] = (
                delta
                if intervention.layer not in deltas
                else deltas[intervention.layer] + delta
            )
        return deltas


@dataclass(frozen=True)
class DecoderIntervention:
    """A matched-norm edit at one absolute decoder residual position."""

    layer: int
    position: int
    direction: torch.Tensor
    strength: float

    def make_vector(self, residual: torch.Tensor) -> torch.Tensor:
        """Scale ``direction`` against the baseline norm at ``position``."""
        if residual.ndim != 3:
            raise ValueError("residual must have shape [batch, positions, width]")
        if not 0 <= self.position < residual.shape[1]:
            raise ValueError("intervention position is outside the residual span")
        if self.direction.ndim != 1 or self.direction.numel() != residual.shape[-1]:
            raise ValueError("direction must have one value per residual width")
        if self.strength < 0:
            raise ValueError("strength cannot be negative")

        direction = self.direction.to(device=residual.device, dtype=torch.float32)
        direction_norm = direction.norm()
        if not torch.isfinite(direction_norm) or direction_norm <= 0:
            raise ValueError("direction must have a finite, nonzero norm")
        reference_norm = residual[:, self.position].float().norm(dim=-1).mean()
        return (direction / direction_norm * reference_norm * self.strength).detach()


@dataclass(frozen=True)
class DecoderInterventionSchedule:
    """One or more post-block decoder edits applied in forward order."""

    interventions: tuple[DecoderIntervention, ...]

    def __post_init__(self) -> None:
        if not self.interventions:
            raise ValueError("a decoder intervention schedule cannot be empty")
        layers = [intervention.layer for intervention in self.interventions]
        if layers != sorted(layers):
            raise ValueError("decoder schedule layers must be ordered")

    def make_vectors(
        self, baseline_decoder: dict[int, torch.Tensor]
    ) -> dict[int, dict[int, torch.Tensor]]:
        """Combine same-layer, same-position edits into one hook payload."""
        vectors: dict[int, dict[int, torch.Tensor]] = {}
        for intervention in self.interventions:
            vector = intervention.make_vector(
                baseline_decoder[intervention.layer]
            )
            layer_vectors = vectors.setdefault(intervention.layer, {})
            layer_vectors[intervention.position] = (
                vector
                if intervention.position not in layer_vectors
                else layer_vectors[intervention.position] + vector
            )
        return vectors


@dataclass(frozen=True)
class WhisperCausalTrace:
    """Baseline and intervened states from one full Whisper recomputation."""

    schedule: EncoderInterventionSchedule
    deltas: dict[int, torch.Tensor]
    baseline_encoder: dict[int, torch.Tensor]
    steered_encoder: dict[int, torch.Tensor]
    baseline_decoder: dict[int, torch.Tensor]
    steered_decoder: dict[int, torch.Tensor]
    baseline_logits: torch.Tensor
    steered_logits: torch.Tensor

    @property
    def intervention(self) -> EncoderIntervention:
        """Single edit compatibility accessor for existing callers."""
        if len(self.schedule.interventions) != 1:
            raise ValueError("this trace contains multiple interventions")
        return self.schedule.interventions[0]

    @property
    def delta(self) -> torch.Tensor:
        """Single edit compatibility accessor for existing callers."""
        return self.deltas[self.intervention.layer]

    def encoder_change_norms(self) -> dict[int, torch.Tensor]:
        """Per-position L2 changes for each captured encoder layer."""
        return {
            layer: (self.steered_encoder[layer] - baseline).float().norm(dim=-1)
            for layer, baseline in self.baseline_encoder.items()
        }

    def decoder_change_norms(self) -> dict[int, torch.Tensor]:
        """Per-position L2 changes for each captured decoder layer."""
        return {
            layer: (self.steered_decoder[layer] - baseline).float().norm(dim=-1)
            for layer, baseline in self.baseline_decoder.items()
        }


@dataclass(frozen=True)
class WhisperDecoderCausalTrace:
    """Baseline and intervened decoder states for one teacher-forced path."""

    schedule: DecoderInterventionSchedule
    vectors: dict[int, dict[int, torch.Tensor]]
    baseline_decoder: dict[int, torch.Tensor]
    steered_decoder: dict[int, torch.Tensor]
    baseline_logits: torch.Tensor
    steered_logits: torch.Tensor

    def decoder_change_norms(self) -> dict[int, torch.Tensor]:
        """Per-position L2 changes for each captured decoder layer."""
        return {
            layer: (self.steered_decoder[layer] - baseline).float().norm(dim=-1)
            for layer, baseline in self.baseline_decoder.items()
        }


@dataclass(frozen=True)
class CandidateSequenceScore:
    """Teacher-forced probability summary for one complete text candidate."""

    text: str
    token_ids: tuple[int, ...]
    token_log_probabilities: tuple[float, ...]
    total_log_probability: float
    mean_log_probability: float


def encoder_lens_contrast_direction(
    model,
    lens: WhisperJacobianLens,
    *,
    layer: int,
    positive_token_ids: list[int],
    negative_token_ids: list[int],
) -> torch.Tensor:
    """Return a J-lens-proposed source direction for a token-set contrast."""
    if lens.encoder is None:
        raise ValueError("an encoder-to-decoder lens is required for this direction")
    if layer not in lens.encoder.source_layers:
        raise ValueError(f"encoder lens has no source layer {layer}")
    if not positive_token_ids or not negative_token_ids:
        raise ValueError("positive and negative token sets must both be nonempty")
    if sorted(positive_token_ids) == sorted(negative_token_ids):
        raise ValueError("positive and negative token sets must differ")
    token_ids = [*positive_token_ids, *negative_token_ids]
    if min(token_ids) < 0 or max(token_ids) >= model.vocab_size:
        raise ValueError("contrast token ID is outside the model vocabulary")

    directions = lens.encoder.vocabulary_directions(
        model.unembedding_weight.float(), layer
    )
    positive = directions[positive_token_ids].mean(dim=0)
    negative = directions[negative_token_ids].mean(dim=0)
    return (positive - negative).detach()


def decoder_lens_contrast_direction(
    model,
    lens: WhisperJacobianLens,
    *,
    layer: int,
    positive_token_ids: list[int],
    negative_token_ids: list[int],
) -> torch.Tensor:
    """Return a decoder J-lens source direction for a token-set contrast."""
    if lens.decoder is None:
        raise ValueError("a decoder-to-decoder lens is required for this direction")
    if layer not in lens.decoder.source_layers:
        raise ValueError(f"decoder lens has no source layer {layer}")
    if not positive_token_ids or not negative_token_ids:
        raise ValueError("positive and negative token sets must both be nonempty")
    if sorted(positive_token_ids) == sorted(negative_token_ids):
        raise ValueError("positive and negative token sets must differ")
    token_ids = [*positive_token_ids, *negative_token_ids]
    if min(token_ids) < 0 or max(token_ids) >= model.vocab_size:
        raise ValueError("contrast token ID is outside the model vocabulary")

    directions = lens.decoder.vocabulary_directions(
        model.unembedding_weight.float(), layer
    )
    positive = directions[positive_token_ids].mean(dim=0)
    negative = directions[negative_token_ids].mean(dim=0)
    return (positive - negative).detach()


def random_encoder_direction(
    width: int, *, seed: int, device: torch.device | str = "cpu"
) -> torch.Tensor:
    """Create a reproducible matched-norm control direction.

    :class:`EncoderIntervention` normalizes all directions before scaling, so a
    random vector from this function is matched in intervention norm to a
    J-lens-derived direction at the same layer and audio span.
    """
    if width <= 0:
        raise ValueError("width must be positive")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(width, generator=generator).to(device)


def random_decoder_direction(
    width: int, *, seed: int, device: torch.device | str = "cpu"
) -> torch.Tensor:
    """Create a reproducible decoder-space matched-norm control direction."""
    return random_encoder_direction(width, seed=seed, device=device)


def _residual_schedule_context(model, schedule, deltas):
    """Register schedule edits before recorders observe source states."""
    stack = ExitStack()
    for layer in sorted(deltas):
        stack.enter_context(
            ResidualAdder(
                model.encoder_layers,
                layer=layer,
                delta=deltas[layer],
            )
        )
    return stack


def _decoder_residual_schedule_context(model, vectors):
    """Register one position-aware decoder edit hook per edited layer."""
    return DecoderResidualScheduleAdder(
        getattr(model, "decoder", None),
        model.decoder_layers,
        vectors_by_layer=vectors,
    )


def candidate_text_token_ids(model: Any, text: str) -> list[int]:
    """Tokenize a candidate continuation without adding Whisper special tokens."""
    if not text:
        raise ValueError("candidate text cannot be empty")
    tokenizer = model.tokenizer
    encoded = tokenizer(text, add_special_tokens=False)
    token_ids = (
        encoded.input_ids if hasattr(encoded, "input_ids") else encoded["input_ids"]
    )
    if token_ids and isinstance(token_ids[0], list):
        if len(token_ids) != 1:
            raise ValueError("candidate tokenization unexpectedly returned a batch")
        token_ids = token_ids[0]
    token_ids = [int(token_id) for token_id in token_ids]
    if not token_ids:
        raise ValueError(f"candidate text {text!r} contains no ordinary tokens")
    return token_ids


def vocabulary_token_ids_starting_with(
    model: Any,
    prefix: str,
    *,
    strip_leading_whitespace: bool = True,
) -> list[int]:
    """Return ordinary token IDs whose decoded text starts with ``prefix``.

    GPT-style tokenizers encode word-boundary whitespace inside many tokens.
    The default strips only leading whitespace before matching so ``Y`` groups
    both ``Y`` and `` Y...`` tokens while preserving case and every other
    character.
    """
    normalized_prefix = prefix.lstrip() if strip_leading_whitespace else prefix
    if not normalized_prefix:
        raise ValueError("vocabulary prefix cannot be empty")
    special_ids = set(int(token_id) for token_id in model.tokenizer.all_special_ids)
    matches: list[int] = []
    for token_id in range(model.vocab_size):
        if token_id in special_ids:
            continue
        decoded = model.tokenizer.decode(
            [token_id], clean_up_tokenization_spaces=False
        )
        candidate = decoded.lstrip() if strip_leading_whitespace else decoded
        if candidate.startswith(normalized_prefix):
            matches.append(token_id)
    if not matches:
        raise ValueError(
            f"no ordinary vocabulary tokens start with {normalized_prefix!r}"
        )
    return matches


def prepare_candidate_inputs(
    model: Any, inputs: WhisperLensInputs, text: str
) -> tuple[WhisperLensInputs, tuple[int, ...], int]:
    """Build teacher-forced inputs and return the first prediction position."""
    prefix_ids = [int(token_id) for token_id in model.decoder_prefix_ids]
    if not prefix_ids:
        raise ValueError("model has no decoder prefix; cannot score a continuation")
    token_ids = candidate_text_token_ids(model, text)
    sequence = [*prefix_ids, *token_ids]
    if len(sequence) < 2:
        raise ValueError("candidate sequence needs a prefix and at least one token")

    inputs = inputs.to(model.input_device)
    decoder_input_ids = torch.tensor(
        [sequence[:-1]], device=model.input_device, dtype=torch.long
    )
    decoder_target_ids = torch.tensor(
        [sequence[1:]], device=model.input_device, dtype=torch.long
    )
    candidate_start = len(prefix_ids) - 1
    decoder_position_mask = torch.zeros_like(decoder_input_ids, dtype=torch.bool)
    decoder_position_mask[:, candidate_start:] = True
    return (
        replace(
            inputs,
            decoder_input_ids=decoder_input_ids,
            decoder_target_ids=decoder_target_ids,
            decoder_position_mask=decoder_position_mask,
        ),
        tuple(token_ids),
        candidate_start,
    )


@torch.no_grad()
def score_candidate_text(
    model: Any,
    inputs: WhisperLensInputs,
    text: str,
    *,
    intervention: EncoderIntervention | None = None,
    delta: torch.Tensor | None = None,
    schedule: EncoderInterventionSchedule | None = None,
    deltas: dict[int, torch.Tensor] | None = None,
    decoder_vectors: dict[int, dict[int, torch.Tensor]] | None = None,
) -> CandidateSequenceScore:
    """Score a complete candidate continuation on the same audio.

    The returned score covers candidate text tokens only, excluding Whisper's
    forced decoder prefix and EOS. Total token-path log probability is the
    primary causal comparison; mean token log probability is retained as a
    secondary diagnostic that removes the mechanical shorter-path advantage.
    """
    if decoder_vectors is not None and any(
        value is not None for value in (intervention, delta, schedule, deltas)
    ):
        raise ValueError("pass either encoder edits or decoder edit vectors")
    if schedule is not None or deltas is not None:
        if intervention is not None or delta is not None:
            raise ValueError("pass either a single edit or an edit schedule")
        if schedule is None or deltas is None:
            raise ValueError("schedule and deltas must be supplied together")
    elif (intervention is None) != (delta is None):
        raise ValueError("intervention and delta must be supplied together")
    elif intervention is not None:
        schedule = EncoderInterventionSchedule((intervention,))
        deltas = {intervention.layer: delta}
    candidate_inputs, token_ids, candidate_start = prepare_candidate_inputs(
        model, inputs, text
    )
    if decoder_vectors is not None:
        context = _decoder_residual_schedule_context(model, decoder_vectors)
    elif schedule is not None:
        context = _residual_schedule_context(model, schedule, deltas)
    else:
        context = nullcontext()
    with context:
        outputs = model.forward(candidate_inputs)
    log_probabilities = model.output_head(outputs.last_hidden_state).float().log_softmax(
        dim=-1
    )
    candidate_targets = candidate_inputs.decoder_target_ids[:, candidate_start:]
    selected = log_probabilities[:, candidate_start:].gather(
        -1, candidate_targets.unsqueeze(-1)
    ).squeeze(-1)
    token_log_probabilities = tuple(float(value) for value in selected[0].cpu())
    total = sum(token_log_probabilities)
    return CandidateSequenceScore(
        text=text,
        token_ids=token_ids,
        token_log_probabilities=token_log_probabilities,
        total_log_probability=total,
        mean_log_probability=total / len(token_ids),
    )


@torch.no_grad()
def run_encoder_intervention(
    model,
    inputs: WhisperLensInputs,
    intervention: EncoderIntervention,
) -> WhisperCausalTrace:
    """Run one additive encoder intervention with downstream capture."""
    return run_encoder_intervention_schedule(
        model,
        inputs,
        EncoderInterventionSchedule((intervention,)),
    )


@torch.no_grad()
def run_encoder_intervention_schedule(
    model,
    inputs: WhisperLensInputs,
    schedule: EncoderInterventionSchedule,
) -> WhisperCausalTrace:
    """Run an ordered encoder edit schedule with downstream capture."""
    if any(
        intervention.layer < 0 or intervention.layer >= model.n_encoder_layers
        for intervention in schedule.interventions
    ):
        raise ValueError("intervention layer is outside the model encoder")
    inputs = inputs.to(model.input_device)
    encoder_layers = list(range(model.n_encoder_layers))
    decoder_layers = list(range(model.n_decoder_layers))

    with (
        ActivationRecorder(model.encoder_layers, at=encoder_layers) as baseline_encoder,
        ActivationRecorder(model.decoder_layers, at=decoder_layers) as baseline_decoder,
    ):
        baseline_output = model.forward(inputs)
    baseline_encoder_states = {
        layer: baseline_encoder.activations[layer].detach() for layer in encoder_layers
    }
    baseline_decoder_states = {
        layer: baseline_decoder.activations[layer].detach() for layer in decoder_layers
    }
    deltas = schedule.make_deltas(baseline_encoder_states)

    # Register the intervention first so recorders see the modified source
    # residual as well as every genuinely downstream state.
    with (
        _residual_schedule_context(model, schedule, deltas),
        ActivationRecorder(model.encoder_layers, at=encoder_layers) as steered_encoder,
        ActivationRecorder(model.decoder_layers, at=decoder_layers) as steered_decoder,
    ):
        steered_output = model.forward(inputs)
    return WhisperCausalTrace(
        schedule=schedule,
        deltas={layer: delta.detach() for layer, delta in deltas.items()},
        baseline_encoder=baseline_encoder_states,
        steered_encoder={
            layer: steered_encoder.activations[layer].detach()
            for layer in encoder_layers
        },
        baseline_decoder=baseline_decoder_states,
        steered_decoder={
            layer: steered_decoder.activations[layer].detach()
            for layer in decoder_layers
        },
        baseline_logits=model.output_head(baseline_output.last_hidden_state).detach(),
        steered_logits=model.output_head(steered_output.last_hidden_state).detach(),
    )


@torch.no_grad()
def run_decoder_intervention_schedule(
    model,
    inputs: WhisperLensInputs,
    schedule: DecoderInterventionSchedule,
) -> WhisperDecoderCausalTrace:
    """Run ordered decoder edits on one teacher-forced sequence."""
    if any(
        intervention.layer < 0 or intervention.layer >= model.n_decoder_layers
        for intervention in schedule.interventions
    ):
        raise ValueError("intervention layer is outside the model decoder")
    inputs = inputs.to(model.input_device)
    decoder_layers = list(range(model.n_decoder_layers))

    with ActivationRecorder(
        model.decoder_layers, at=decoder_layers
    ) as baseline_decoder:
        baseline_output = model.forward(inputs)
    baseline_decoder_states = {
        layer: baseline_decoder.activations[layer].detach()
        for layer in decoder_layers
    }
    vectors = schedule.make_vectors(baseline_decoder_states)

    # Register edits before recorders so an edited source layer is captured in
    # its shifted state and every later decoder block sees that state.
    with (
        _decoder_residual_schedule_context(model, vectors),
        ActivationRecorder(model.decoder_layers, at=decoder_layers) as steered_decoder,
    ):
        steered_output = model.forward(inputs)
    return WhisperDecoderCausalTrace(
        schedule=schedule,
        vectors={
            layer: {
                position: vector.detach()
                for position, vector in layer_vectors.items()
            }
            for layer, layer_vectors in vectors.items()
        },
        baseline_decoder=baseline_decoder_states,
        steered_decoder={
            layer: steered_decoder.activations[layer].detach()
            for layer in decoder_layers
        },
        baseline_logits=model.output_head(baseline_output.last_hidden_state).detach(),
        steered_logits=model.output_head(steered_output.last_hidden_state).detach(),
    )
