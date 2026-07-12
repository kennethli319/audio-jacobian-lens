# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Saved Whisper lens bundles and memory-bounded readout helpers."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from jlens.cross_lens import CrossJacobianLens, _merge_shard_metadata


def _is_combined_estimator_metadata(metadata: Mapping[str, Any]) -> bool:
    return set(metadata) == {"encoder", "decoder"}


def _bundle_example_count(bundle: WhisperJacobianLens) -> int:
    counts = {
        lens.n_examples for lens in (bundle.encoder, bundle.decoder) if lens is not None
    }
    if len(counts) != 1:
        raise ValueError(
            "flat estimator metadata cannot describe streams fitted on "
            "different example counts"
        )
    return counts.pop()


def _merge_estimator_metadata(
    lenses: Sequence[WhisperJacobianLens],
) -> dict[str, Any]:
    """Merge flat or per-stream estimator metadata without losing shards."""
    combined = [
        _is_combined_estimator_metadata(lens.estimator_metadata) for lens in lenses
    ]
    if any(value != combined[0] for value in combined[1:]):
        raise ValueError("Whisper lens bundles use different metadata schemas")
    if not combined[0]:
        return _merge_shard_metadata(
            [(lens.estimator_metadata, _bundle_example_count(lens)) for lens in lenses]
        )

    merged: dict[str, Any] = {}
    for stream in ("encoder", "decoder"):
        stream_items: list[tuple[Mapping[str, Any], int]] = []
        for bundle in lenses:
            stream_lens = getattr(bundle, stream)
            stream_metadata = bundle.estimator_metadata[stream]
            if (stream_lens is None) != (stream_metadata is None):
                raise ValueError(
                    f"{stream} estimator metadata does not match stream presence"
                )
            if stream_lens is not None:
                if not isinstance(stream_metadata, Mapping):
                    raise ValueError(f"{stream} estimator metadata must be a mapping")
                stream_items.append((stream_metadata, stream_lens.n_examples))
        merged[stream] = (
            None if not stream_items else _merge_shard_metadata(stream_items)
        )
    return merged


class WhisperJacobianLens:
    """Decoder and/or encoder-to-decoder lenses fitted for one Whisper setup."""

    FORMAT_VERSION = 2

    def __init__(
        self,
        *,
        decoder: CrossJacobianLens | None = None,
        encoder: CrossJacobianLens | None = None,
        model_metadata: Mapping[str, Any],
        estimator_metadata: Mapping[str, Any],
    ) -> None:
        if decoder is None and encoder is None:
            raise ValueError("a WhisperJacobianLens needs decoder and/or encoder maps")
        if decoder is not None and (
            decoder.source_stream != "decoder" or decoder.target_stream != "decoder"
        ):
            raise ValueError("decoder lens streams must be decoder->decoder")
        if encoder is not None and (
            encoder.source_stream != "encoder" or encoder.target_stream != "decoder"
        ):
            raise ValueError("encoder lens streams must be encoder->decoder")
        if decoder is not None and encoder is not None:
            if decoder.target_dim != encoder.target_dim:
                raise ValueError("encoder and decoder lens target widths disagree")

        self.decoder = decoder
        self.encoder = encoder
        self.model_metadata = dict(model_metadata)
        self.estimator_metadata = dict(estimator_metadata)

    def __repr__(self) -> str:
        streams = [
            name
            for name, lens in (("encoder", self.encoder), ("decoder", self.decoder))
            if lens is not None
        ]
        return (
            f"WhisperJacobianLens(streams={streams}, "
            f"model={self.model_metadata.get('model_id')!r})"
        )

    def state_dict(self, *, dtype: torch.dtype = torch.float16) -> dict[str, Any]:
        return {
            "format": "whisper-jacobian-lens",
            "format_version": self.FORMAT_VERSION,
            "decoder": None
            if self.decoder is None
            else self.decoder.state_dict(dtype=dtype),
            "encoder": None
            if self.encoder is None
            else self.encoder.state_dict(dtype=dtype),
            "model_metadata": self.model_metadata,
            "estimator_metadata": self.estimator_metadata,
        }

    def save(self, path: str, *, dtype: torch.dtype = torch.float16) -> None:
        torch.save(self.state_dict(dtype=dtype), path)

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> WhisperJacobianLens:
        if state.get("format") != "whisper-jacobian-lens":
            raise ValueError("payload is not a WhisperJacobianLens")
        version = state.get("format_version")
        if version not in (1, cls.FORMAT_VERSION):
            raise ValueError(
                f"unsupported WhisperJacobianLens version {version!r}; "
                f"expected 1 or {cls.FORMAT_VERSION}"
            )
        return cls(
            decoder=(
                None
                if state.get("decoder") is None
                else CrossJacobianLens.from_state_dict(state["decoder"])
            ),
            encoder=(
                None
                if state.get("encoder") is None
                else CrossJacobianLens.from_state_dict(state["encoder"])
            ),
            model_metadata=state["model_metadata"],
            estimator_metadata=state["estimator_metadata"],
        )

    @classmethod
    def load(cls, path: str) -> WhisperJacobianLens:
        return cls.from_state_dict(
            torch.load(path, map_location="cpu", weights_only=True)
        )

    @classmethod
    def from_pretrained(
        cls,
        name_or_path: str,
        *,
        filename: str = "whisper_jacobian_lens.pt",
        revision: str | None = None,
    ) -> WhisperJacobianLens:
        if os.path.isfile(name_or_path):
            return cls.load(name_or_path)
        if not os.path.isdir(name_or_path):
            from huggingface_hub import snapshot_download

            name_or_path = snapshot_download(
                name_or_path, allow_patterns=[filename], revision=revision
            )
        return cls.load(os.path.join(name_or_path, filename))

    @classmethod
    def merge(cls, lenses: Sequence[WhisperJacobianLens]) -> WhisperJacobianLens:
        if not lenses:
            raise ValueError("merge() needs at least one lens")
        first = lenses[0]
        for other in lenses[1:]:
            if (
                other.model_metadata != first.model_metadata
                or (other.decoder is None) != (first.decoder is None)
                or (other.encoder is None) != (first.encoder is None)
            ):
                raise ValueError("Whisper lens bundles are not compatible")
        estimator_metadata = _merge_estimator_metadata(lenses)
        decoder = (
            None
            if first.decoder is None
            else CrossJacobianLens.merge(
                [lens.decoder for lens in lenses if lens.decoder is not None]
            )
        )
        encoder = (
            None
            if first.encoder is None
            else CrossJacobianLens.merge(
                [lens.encoder for lens in lenses if lens.encoder is not None]
            )
        )
        return cls(
            decoder=decoder,
            encoder=encoder,
            model_metadata=first.model_metadata,
            estimator_metadata=estimator_metadata,
        )

    @classmethod
    def combine_streams(
        cls,
        *,
        encoder_bundle: WhisperJacobianLens | None = None,
        decoder_bundle: WhisperJacobianLens | None = None,
    ) -> WhisperJacobianLens:
        """Combine independently fitted encoder and decoder artifacts.

        Encoder and decoder lenses often need different example masking (for
        example aligned audio windows versus all causal decoder positions), so
        fitting them independently is preferable to pretending they share one
        estimator configuration.
        """
        if encoder_bundle is None and decoder_bundle is None:
            raise ValueError("provide an encoder_bundle and/or decoder_bundle")
        chosen = encoder_bundle or decoder_bundle
        assert chosen is not None
        if (
            encoder_bundle is not None
            and decoder_bundle is not None
            and encoder_bundle.model_metadata != decoder_bundle.model_metadata
        ):
            raise ValueError("encoder and decoder bundles target different models")
        encoder = None if encoder_bundle is None else encoder_bundle.encoder
        decoder = None if decoder_bundle is None else decoder_bundle.decoder
        if encoder_bundle is not None and encoder is None:
            raise ValueError("encoder_bundle contains no encoder lens")
        if decoder_bundle is not None and decoder is None:
            raise ValueError("decoder_bundle contains no decoder lens")
        return cls(
            encoder=encoder,
            decoder=decoder,
            model_metadata=chosen.model_metadata,
            estimator_metadata={
                "encoder": (
                    None
                    if encoder_bundle is None
                    else encoder_bundle.estimator_metadata
                ),
                "decoder": (
                    None
                    if decoder_bundle is None
                    else decoder_bundle.estimator_metadata
                ),
            },
        )

    def validate_model(self, model: Any) -> None:
        """Reject accidental use with a different Whisper configuration."""
        expected = self.model_metadata.get("model_fingerprint")
        actual = getattr(model, "fingerprint", None)
        if expected is not None and actual != expected:
            raise ValueError(
                f"lens/model fingerprint mismatch: lens={expected!r}, model={actual!r}"
            )


@dataclass(frozen=True)
class PooledResiduals:
    residuals: torch.Tensor
    start_positions: list[int]
    end_positions: list[int]


def pool_sequence_residuals(
    residuals: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    positions_per_bin: int,
    stride_positions: int | None = None,
) -> PooledResiduals:
    """Mean-pool valid positions into display bins with an optional overlap."""
    if residuals.ndim != 2:
        raise ValueError("residuals must have shape [positions, width]")
    if valid_mask.ndim != 1 or valid_mask.shape[0] != residuals.shape[0]:
        raise ValueError("valid_mask must align with residual positions")
    if positions_per_bin <= 0:
        raise ValueError("positions_per_bin must be positive")
    if stride_positions is None:
        stride_positions = positions_per_bin
    if stride_positions <= 0:
        raise ValueError("stride_positions must be positive")
    if stride_positions > positions_per_bin:
        raise ValueError("stride_positions cannot exceed positions_per_bin")
    valid = valid_mask.nonzero(as_tuple=True)[0]
    if valid.numel() == 0:
        raise ValueError("valid_mask selects no residual positions")

    # The audio feature extractor produces a contiguous left-aligned valid span.
    # Reject holes rather than silently pooling unrelated pieces together.
    first, last = int(valid[0]), int(valid[-1]) + 1
    expected = torch.arange(first, last, device=valid.device)
    if not torch.equal(valid, expected):
        raise ValueError("valid_mask must select one contiguous position span")

    pooled: list[torch.Tensor] = []
    starts: list[int] = []
    ends: list[int] = []
    for start in range(first, last, stride_positions):
        end = min(last, start + positions_per_bin)
        # With overlapping windows, a trailing start can be wholly contained in
        # the preceding window and add no new audio coverage.
        if ends and end <= ends[-1]:
            break
        pooled.append(residuals[start:end].mean(dim=0))
        starts.append(start)
        ends.append(end)
    return PooledResiduals(torch.stack(pooled), starts, ends)


@dataclass(frozen=True)
class LensTopK:
    token_ids: torch.Tensor
    scores: torch.Tensor
    ranks: torch.Tensor
    rank_denominator: int
    display_vocabulary_ranks: torch.Tensor
    display_vocabulary_denominator: int
    full_vocabulary_ranks: torch.Tensor
    full_vocabulary_denominator: int
    selected_readouts: LensSelectedReadouts | None = None


@dataclass(frozen=True)
class LensSelectedReadouts:
    """Exact readouts for caller-selected token IDs at each position.

    ``display_vocabulary_ranks`` uses zero only as an internal sentinel when a
    selected token is outside ``token_mask``. Callers must consult
    ``display_vocabulary_eligible`` before serializing that rank. Grouped
    readouts additionally expose per-group strictly-greater counts, allowing
    exact ranks over unions of disjoint groups without retaining logits.
    """

    token_ids: torch.Tensor
    scores: torch.Tensor
    display_vocabulary_ranks: torch.Tensor
    display_vocabulary_eligible: torch.Tensor
    display_vocabulary_denominator: int
    full_vocabulary_ranks: torch.Tensor
    full_vocabulary_denominator: int
    group_strictly_greater_counts: dict[int, torch.Tensor] | None = None
    group_denominators: dict[int, int] | None = None


@dataclass(frozen=True)
class GroupedLensTopK:
    overall: LensTopK
    groups: dict[int, LensTopK]


def _strict_score_ranks(
    logits: torch.Tensor, selected_scores: torch.Tensor
) -> torch.Tensor:
    """Competition ranks: one plus the number of strictly greater scores."""
    return torch.stack(
        [
            (logits > selected_scores[:, index : index + 1]).sum(dim=-1) + 1
            for index in range(selected_scores.shape[1])
        ],
        dim=-1,
    )


def _competition_rank_map(logits: torch.Tensor) -> torch.Tensor:
    """Exact competition rank for every column, computed with one stable sort."""
    sorted_scores, sorted_indices = logits.sort(
        dim=-1, descending=True, stable=True
    )
    width = sorted_scores.shape[1]
    ordinals = torch.arange(
        1, width + 1, device=logits.device, dtype=torch.int32
    ).unsqueeze(0)
    group_starts = torch.cat(
        [
            torch.ones(
                (sorted_scores.shape[0], 1),
                device=logits.device,
                dtype=torch.bool,
            ),
            sorted_scores[:, 1:] < sorted_scores[:, :-1],
        ],
        dim=1,
    )
    sorted_ranks = torch.where(
        group_starts,
        ordinals.expand(sorted_scores.shape[0], -1),
        torch.ones((), device=logits.device, dtype=torch.int32),
    ).cummax(dim=-1).values
    ranks = torch.empty_like(sorted_ranks)
    ranks.scatter_(1, sorted_indices, sorted_ranks)
    return ranks


@torch.no_grad()
def lens_topk(
    model: Any,
    lens: CrossJacobianLens,
    residuals: torch.Tensor,
    *,
    layer: int,
    top_k: int = 10,
    position_chunk_size: int = 64,
    token_mask: torch.Tensor | None = None,
    selected_token_ids: torch.Tensor | None = None,
    subtract_target_baseline: bool = False,
) -> LensTopK:
    """Apply a cross-stream lens without retaining full-vocabulary logits.

    By default, returned ``scores`` are raw lens logits. For a centered
    cross-stream lens, ``subtract_target_baseline=True`` instead returns the
    change from the fitted target-mean readout. This removes the corpus language
    prior while retaining final-normalization effects; neither mode is a
    probability or a calibrated causal effect.
    """
    if residuals.ndim != 2:
        raise ValueError("residuals must have shape [positions, source_dim]")
    if top_k <= 0 or top_k > model.vocab_size:
        raise ValueError(f"top_k must be in [1, {model.vocab_size}]")
    if position_chunk_size <= 0:
        raise ValueError("position_chunk_size must be positive")
    if token_mask is not None and (
        token_mask.ndim != 1 or token_mask.numel() != model.vocab_size
    ):
        raise ValueError("token_mask must have shape [vocab_size]")
    if selected_token_ids is not None:
        if (
            selected_token_ids.ndim != 1
            or selected_token_ids.shape[0] != residuals.shape[0]
        ):
            raise ValueError("selected_token_ids must have shape [positions]")
        if selected_token_ids.dtype == torch.bool or torch.is_floating_point(
            selected_token_ids
        ):
            raise TypeError("selected_token_ids must be an integer tensor")
        if bool(
            ((selected_token_ids < 0) | (selected_token_ids >= model.vocab_size)).any()
        ):
            raise ValueError("selected_token_ids contains an out-of-range token ID")
    if subtract_target_baseline and lens.target_mean is None:
        raise ValueError("target-baseline subtraction needs a centered lens")

    baseline_logits = None
    if subtract_target_baseline:
        assert lens.target_mean is not None
        target_mean = lens.target_mean.to(
            device=residuals.device, dtype=torch.float32
        ).unsqueeze(0)
        baseline_logits = model.unembed(target_mean).float()

    token_mask_cpu = (
        None
        if token_mask is None
        else token_mask.detach().to(device="cpu", dtype=torch.bool)
    )
    eligible_cpu_ids = (
        torch.arange(model.vocab_size)
        if token_mask_cpu is None
        else token_mask_cpu.nonzero(as_tuple=True)[0]
    )
    if eligible_cpu_ids.numel() == 0:
        raise ValueError("token_mask selects no vocabulary entries")
    resolved_top_k = min(top_k, int(eligible_cpu_ids.numel()))
    ids: list[torch.Tensor] = []
    scores: list[torch.Tensor] = []
    ranks: list[torch.Tensor] = []
    full_ranks: list[torch.Tensor] = []
    selected_ids: list[torch.Tensor] = []
    selected_scores: list[torch.Tensor] = []
    selected_display_ranks: list[torch.Tensor] = []
    selected_display_eligible: list[torch.Tensor] = []
    selected_full_ranks: list[torch.Tensor] = []
    for start in range(0, residuals.shape[0], position_chunk_size):
        source = residuals[start : start + position_chunk_size].float()
        transported = lens.transport(source, layer)
        logits = model.unembed(transported).float()
        if baseline_logits is not None:
            logits = logits - baseline_logits.to(logits.device)
        eligible_ids = eligible_cpu_ids.to(logits.device)
        eligible_logits = logits.index_select(1, eligible_ids)
        values, local_indices = eligible_logits.topk(resolved_top_k, dim=-1)
        indices = eligible_ids[local_indices]
        ids.append(indices.cpu())
        scores.append(values.cpu())
        ranks.append(_strict_score_ranks(eligible_logits, values).cpu())
        full_ranks.append(_strict_score_ranks(logits, values).cpu())
        if selected_token_ids is not None:
            chunk_selected_ids = selected_token_ids[start : start + source.shape[0]].to(
                logits.device, dtype=torch.long
            )
            chunk_selected_scores = logits.gather(1, chunk_selected_ids.unsqueeze(1))
            chunk_selected_eligible = (
                token_mask_cpu[chunk_selected_ids.cpu()]
                if token_mask_cpu is not None
                else torch.ones(chunk_selected_ids.shape[0], dtype=torch.bool)
            )
            chunk_display_ranks = _strict_score_ranks(
                eligible_logits, chunk_selected_scores
            ).squeeze(1)
            chunk_display_ranks = torch.where(
                chunk_selected_eligible.to(chunk_display_ranks.device),
                chunk_display_ranks,
                torch.zeros_like(chunk_display_ranks),
            )
            selected_ids.append(chunk_selected_ids.cpu())
            selected_scores.append(chunk_selected_scores.squeeze(1).cpu())
            selected_display_ranks.append(chunk_display_ranks.cpu())
            selected_display_eligible.append(chunk_selected_eligible.cpu())
            selected_full_ranks.append(
                _strict_score_ranks(logits, chunk_selected_scores).squeeze(1).cpu()
            )
    active_ranks = torch.cat(ranks)
    denominator = int(eligible_cpu_ids.numel())
    return LensTopK(
        token_ids=torch.cat(ids),
        scores=torch.cat(scores),
        ranks=active_ranks,
        rank_denominator=denominator,
        display_vocabulary_ranks=active_ranks.clone(),
        display_vocabulary_denominator=denominator,
        full_vocabulary_ranks=torch.cat(full_ranks),
        full_vocabulary_denominator=model.vocab_size,
        selected_readouts=(
            None
            if selected_token_ids is None
            else LensSelectedReadouts(
                token_ids=torch.cat(selected_ids),
                scores=torch.cat(selected_scores),
                display_vocabulary_ranks=torch.cat(selected_display_ranks),
                display_vocabulary_eligible=torch.cat(selected_display_eligible),
                display_vocabulary_denominator=denominator,
                full_vocabulary_ranks=torch.cat(selected_full_ranks),
                full_vocabulary_denominator=model.vocab_size,
            )
        ),
    )


@torch.no_grad()
def lens_topk_grouped(
    model: Any,
    lens: CrossJacobianLens,
    residuals: torch.Tensor,
    *,
    layer: int,
    token_mask: torch.Tensor,
    token_groups: dict[int, torch.Tensor],
    top_k: int = 10,
    position_chunk_size: int = 64,
    selected_token_ids: torch.Tensor | None = None,
    subtract_target_baseline: bool = False,
) -> GroupedLensTopK:
    """Compute an overall and grouped top-k from one unembedding pass.

    Each group is ranked only over its supplied vocabulary mask. Returning the
    top-k for every exact token length is sufficient to reconstruct the exact
    top-k for any maximum length by merging the eligible groups and reranking.
    If ``selected_token_ids`` is supplied, the overall result also retains its
    exact score, display/full ranks, and strictly-greater count in each group.
    """
    if residuals.ndim != 2:
        raise ValueError("residuals must have shape [positions, source_dim]")
    if top_k <= 0 or top_k > model.vocab_size:
        raise ValueError(f"top_k must be in [1, {model.vocab_size}]")
    if position_chunk_size <= 0:
        raise ValueError("position_chunk_size must be positive")
    if selected_token_ids is not None:
        if (
            selected_token_ids.ndim != 1
            or selected_token_ids.shape[0] != residuals.shape[0]
        ):
            raise ValueError("selected_token_ids must have shape [positions]")
        if selected_token_ids.dtype == torch.bool or torch.is_floating_point(
            selected_token_ids
        ):
            raise TypeError("selected_token_ids must be an integer tensor")
        if bool(
            ((selected_token_ids < 0) | (selected_token_ids >= model.vocab_size)).any()
        ):
            raise ValueError("selected_token_ids contains an out-of-range token ID")
    masks = {"overall": token_mask, **token_groups}
    if any(mask.ndim != 1 or mask.numel() != model.vocab_size for mask in masks.values()):
        raise ValueError("all token masks must have shape [vocab_size]")
    if any(bool((mask & ~token_mask).any()) for mask in token_groups.values()):
        raise ValueError("token groups must be subsets of token_mask")
    if subtract_target_baseline and lens.target_mean is None:
        raise ValueError("target-baseline subtraction needs a centered lens")

    baseline_logits = None
    if subtract_target_baseline:
        assert lens.target_mean is not None
        target_mean = lens.target_mean.to(
            device=residuals.device, dtype=torch.float32
        ).unsqueeze(0)
        baseline_logits = model.unembed(target_mean).float()

    group_token_ids = {
        group: mask.nonzero(as_tuple=True)[0]
        for group, mask in masks.items()
        if bool(mask.any())
    }
    collected_ids: dict[int | str, list[torch.Tensor]] = {
        group: [] for group in group_token_ids
    }
    collected_scores: dict[int | str, list[torch.Tensor]] = {
        group: [] for group in group_token_ids
    }
    collected_ranks: dict[int | str, list[torch.Tensor]] = {
        group: [] for group in group_token_ids
    }
    collected_display_ranks: dict[int | str, list[torch.Tensor]] = {
        group: [] for group in group_token_ids
    }
    collected_full_ranks: dict[int | str, list[torch.Tensor]] = {
        group: [] for group in group_token_ids
    }
    selected_ids: list[torch.Tensor] = []
    selected_scores: list[torch.Tensor] = []
    selected_display_ranks: list[torch.Tensor] = []
    selected_display_eligible: list[torch.Tensor] = []
    selected_full_ranks: list[torch.Tensor] = []
    selected_group_greater_counts: dict[int, list[torch.Tensor]] = {
        int(group): [] for group in group_token_ids if group != "overall"
    }
    display_cpu_ids = group_token_ids["overall"]
    display_local_by_vocab = torch.full(
        (model.vocab_size,), -1, dtype=torch.long
    )
    display_local_by_vocab[display_cpu_ids] = torch.arange(
        display_cpu_ids.numel()
    )
    for start in range(0, residuals.shape[0], position_chunk_size):
        source = residuals[start : start + position_chunk_size].float()
        transported = lens.transport(source, layer)
        logits = model.unembed(transported).float()
        if baseline_logits is not None:
            logits = logits - baseline_logits.to(logits.device)
        # Stable full-vocabulary sorting is substantially faster on CPU than
        # MPS for Whisper-sized rows. Copy one bounded position chunk, compute
        # both rank maps once, then gather only the serialized candidates.
        cpu_logits = logits.cpu()
        full_rank_map = _competition_rank_map(cpu_logits)
        display_rank_map = _competition_rank_map(
            cpu_logits.index_select(1, display_cpu_ids)
        )
        if selected_token_ids is not None:
            chunk_selected_ids = selected_token_ids[start : start + source.shape[0]].to(
                device="cpu", dtype=torch.long
            )
            chunk_selected_scores = cpu_logits.gather(
                1, chunk_selected_ids.unsqueeze(1)
            ).squeeze(1)
            chunk_display_indices = display_local_by_vocab[chunk_selected_ids]
            chunk_display_eligible = chunk_display_indices >= 0
            chunk_display_ranks = display_rank_map.gather(
                1, chunk_display_indices.clamp_min(0).unsqueeze(1)
            ).squeeze(1)
            chunk_display_ranks = torch.where(
                chunk_display_eligible,
                chunk_display_ranks,
                torch.zeros_like(chunk_display_ranks),
            )
            selected_ids.append(chunk_selected_ids)
            selected_scores.append(chunk_selected_scores)
            selected_display_ranks.append(chunk_display_ranks)
            selected_display_eligible.append(chunk_display_eligible)
            selected_full_ranks.append(
                full_rank_map.gather(1, chunk_selected_ids.unsqueeze(1)).squeeze(1)
            )
            for group, cpu_ids in group_token_ids.items():
                if group == "overall":
                    continue
                selected_group_greater_counts[int(group)].append(
                    (
                        cpu_logits.index_select(1, cpu_ids)
                        > chunk_selected_scores.unsqueeze(1)
                    ).sum(dim=1)
                )
        for group, cpu_ids in group_token_ids.items():
            eligible_ids = cpu_ids.to(logits.device)
            eligible_logits = logits.index_select(1, eligible_ids)
            values, local_indices = eligible_logits.topk(
                min(top_k, eligible_ids.numel()), dim=-1
            )
            selected_cpu_ids = eligible_ids[local_indices].cpu()
            collected_ids[group].append(selected_cpu_ids)
            collected_scores[group].append(values.cpu())
            collected_ranks[group].append(
                _strict_score_ranks(eligible_logits, values).cpu()
            )
            collected_display_ranks[group].append(
                display_rank_map.gather(
                    1, display_local_by_vocab[selected_cpu_ids]
                ).cpu()
            )
            collected_full_ranks[group].append(
                full_rank_map.gather(1, selected_cpu_ids).cpu()
            )

    selected_readouts = (
        None
        if selected_token_ids is None
        else LensSelectedReadouts(
            token_ids=torch.cat(selected_ids),
            scores=torch.cat(selected_scores),
            display_vocabulary_ranks=torch.cat(selected_display_ranks),
            display_vocabulary_eligible=torch.cat(selected_display_eligible),
            display_vocabulary_denominator=int(display_cpu_ids.numel()),
            full_vocabulary_ranks=torch.cat(selected_full_ranks),
            full_vocabulary_denominator=model.vocab_size,
            group_strictly_greater_counts={
                group: torch.cat(chunks)
                for group, chunks in selected_group_greater_counts.items()
            },
            group_denominators={
                int(group): int(ids.numel())
                for group, ids in group_token_ids.items()
                if group != "overall"
            },
        )
    )
    results = {
        group: LensTopK(
            token_ids=torch.cat(collected_ids[group]),
            scores=torch.cat(collected_scores[group]),
            ranks=torch.cat(collected_ranks[group]),
            rank_denominator=int(group_token_ids[group].numel()),
            display_vocabulary_ranks=torch.cat(
                collected_display_ranks[group]
            ),
            display_vocabulary_denominator=int(
                group_token_ids["overall"].numel()
            ),
            full_vocabulary_ranks=torch.cat(collected_full_ranks[group]),
            full_vocabulary_denominator=model.vocab_size,
            selected_readouts=(
                selected_readouts if group == "overall" else None
            ),
        )
        for group in group_token_ids
    }
    overall = results.pop("overall")
    return GroupedLensTopK(overall=overall, groups=results)
