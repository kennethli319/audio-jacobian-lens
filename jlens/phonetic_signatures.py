# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Frozen phone-prototype readouts for Whisper encoder J-signatures.

The prototypes in this module are not model probabilities.  Each prototype is
the normalized mean of sparse, rank-thresholded top-k J-readout signatures for
one aligned training-set phone.  A new encoder state is assigned similarities
by constructing the same sparse signature and taking cosine products with the
frozen prototypes.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from jlens.cross_lens import CrossJacobianLens


@dataclass(frozen=True)
class PhonePrototypePullback:
    """One phone-prototype readout objective pulled into encoder space.

    ``direction`` is the local gradient of a fitted readout-logit contrast, not
    a phone probability or evidence that Whisper causally uses the prototype.
    Callers may propose it as an additive residual intervention and must measure
    the resulting model computation separately.
    """

    direction: torch.Tensor
    layer: int
    target_phone: str
    objective_value: float
    target_readout_score: float
    other_mean_readout_score: float
    competing_phone_count: int
    gradient_l2_norm: float
    objective_kind: str = (
        "target_phone_prototype_minus_other_phone_mean_readout_logit"
    )
    is_probability: bool = False


def _hash_tensor(digest: Any, tensor: torch.Tensor) -> None:
    value = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
    digest.update(str(tuple(value.shape)).encode())
    digest.update(np.asarray(value.numpy(), dtype="<f4").tobytes())


def encoder_lens_fingerprint(lens: CrossJacobianLens) -> str:
    """Return a deterministic content fingerprint for one encoder lens.

    This follows the in-memory float32 lens content instead of the container
    file bytes, so independently combined stream bundles can still prove that
    they use exactly the encoder transport that produced the prototypes.
    """
    digest = hashlib.sha256()
    stable = {
        "source_layers": lens.source_layers,
        "n_examples": lens.n_examples,
        "source_dim": lens.source_dim,
        "target_dim": lens.target_dim,
        "source_stream": lens.source_stream,
        "target_stream": lens.target_stream,
        "metadata": lens.metadata,
    }
    digest.update(
        json.dumps(
            stable,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode()
    )
    for layer in lens.source_layers:
        digest.update(f"J:{layer}".encode())
        _hash_tensor(digest, lens.jacobians[layer])
        if lens.source_means is not None:
            digest.update(f"source_mean:{layer}".encode())
            _hash_tensor(digest, lens.source_means[layer])
    if lens.target_mean is not None:
        digest.update(b"target_mean")
        _hash_tensor(digest, lens.target_mean)
    return digest.hexdigest()


class PhoneSignaturePrototypes:
    """A frozen bank of phone prototypes compatible with one encoder lens."""

    FORMAT = "jlens-phone-signature-prototypes"
    FORMAT_VERSION = 1
    SCORE_KIND = "phone_prototype_cosine_similarity"

    def __init__(
        self,
        prototypes: Mapping[int, torch.Tensor],
        *,
        labels: list[str],
        signature_top_k: int,
        vocab_size: int,
        model_fingerprint: str,
        encoder_lens_fingerprint_value: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if not labels or any(not isinstance(label, str) or not label for label in labels):
            raise ValueError("phone labels must be non-empty strings")
        if len(set(labels)) != len(labels):
            raise ValueError("phone labels must be unique")
        if signature_top_k <= 0 or signature_top_k >= vocab_size:
            raise ValueError("signature_top_k must be in [1, vocab_size)")
        if not prototypes:
            raise ValueError("phone prototypes must contain at least one layer")

        checked: dict[int, torch.Tensor] = {}
        expected = (len(labels), vocab_size)
        for raw_layer, raw_prototypes in prototypes.items():
            layer = int(raw_layer)
            value = raw_prototypes.detach().to(device="cpu", dtype=torch.float32)
            if value.ndim != 2 or tuple(value.shape) != expected:
                raise ValueError(
                    f"layer {layer} prototypes have shape {tuple(value.shape)}, "
                    f"expected {expected}"
                )
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"layer {layer} prototypes contain non-finite values")
            norms = torch.linalg.vector_norm(value, dim=1)
            if not torch.allclose(norms, torch.ones_like(norms), atol=2e-3, rtol=2e-3):
                raise ValueError(f"layer {layer} phone prototypes are not normalized")
            checked[layer] = value.contiguous()

        self.prototypes = checked
        self.source_layers = sorted(checked)
        self.labels = list(labels)
        self.signature_top_k = int(signature_top_k)
        self.vocab_size = int(vocab_size)
        self.model_fingerprint = str(model_fingerprint)
        self.encoder_lens_fingerprint = str(encoder_lens_fingerprint_value)
        self.metadata = dict(metadata or {})

    def state_dict(self, *, dtype: torch.dtype = torch.float16) -> dict[str, Any]:
        return {
            "format": self.FORMAT,
            "format_version": self.FORMAT_VERSION,
            "labels": self.labels,
            "source_layers": self.source_layers,
            "prototypes": {
                layer: value.to(dtype=dtype)
                for layer, value in self.prototypes.items()
            },
            "signature_top_k": self.signature_top_k,
            "vocab_size": self.vocab_size,
            "model_fingerprint": self.model_fingerprint,
            "encoder_lens_fingerprint": self.encoder_lens_fingerprint,
            "metadata": self.metadata,
        }

    @classmethod
    def from_state_dict(
        cls, state: Mapping[str, Any]
    ) -> PhoneSignaturePrototypes:
        if state.get("format") != cls.FORMAT:
            raise ValueError("payload is not a phone-signature prototype artifact")
        if state.get("format_version") != cls.FORMAT_VERSION:
            raise ValueError(
                "unsupported phone-signature prototype format version "
                f"{state.get('format_version')!r}"
            )
        source_layers = [int(layer) for layer in state.get("source_layers", [])]
        prototypes = {int(layer): value for layer, value in state["prototypes"].items()}
        if sorted(prototypes) != sorted(source_layers):
            raise ValueError("prototype layers do not match source_layers")
        return cls(
            prototypes,
            labels=list(state["labels"]),
            signature_top_k=int(state["signature_top_k"]),
            vocab_size=int(state["vocab_size"]),
            model_fingerprint=str(state["model_fingerprint"]),
            encoder_lens_fingerprint_value=str(state["encoder_lens_fingerprint"]),
            metadata=state.get("metadata", {}),
        )

    @classmethod
    def load(cls, path: str | Path) -> PhoneSignaturePrototypes:
        return cls.from_state_dict(
            torch.load(path, map_location="cpu", weights_only=True)
        )

    def save(
        self, path: str | Path, *, dtype: torch.dtype = torch.float16
    ) -> None:
        torch.save(self.state_dict(dtype=dtype), path)

    def validate(
        self,
        *,
        model: Any,
        encoder_lens: CrossJacobianLens,
    ) -> None:
        if encoder_lens.source_layers != self.source_layers:
            raise ValueError(
                "phone prototype layers do not match the active encoder lens"
            )
        if getattr(model, "vocab_size", None) != self.vocab_size:
            raise ValueError(
                "phone prototype vocabulary does not match the active model"
            )
        if getattr(model, "fingerprint", None) != self.model_fingerprint:
            raise ValueError(
                "phone prototype model fingerprint does not match the active model"
            )
        actual_lens_fingerprint = encoder_lens_fingerprint(encoder_lens)
        if actual_lens_fingerprint != self.encoder_lens_fingerprint:
            raise ValueError(
                "phone prototypes were fitted from a different encoder lens"
            )

    def prototype_logit_pullback(
        self,
        model: Any,
        encoder_lens: CrossJacobianLens,
        reference_residual: torch.Tensor,
        *,
        layer: int,
        target_phone: str,
    ) -> PhonePrototypePullback:
        """Pull a dense phone-prototype contrast into encoder coordinates.

        The scalar objective is the target prototype's dot product with the
        target-mean-relative J-readout logits minus the mean corresponding
        score of every other fitted phone prototype. Its gradient is taken with
        respect to the actual reference encoder residual through the centered
        encoder transport and the model's normal final norm and unembedding.

        This deliberately uses the complete fitted vocabulary-space prototype,
        rather than a decoded-token or BPE proxy. The result is only a proposed
        local causal intervention direction; it is not a probability, a
        calibrated phone score, or proof that the model uses this direction.
        """
        self.validate(model=model, encoder_lens=encoder_lens)
        if layer not in self.prototypes:
            raise ValueError(f"phone prototypes contain no layer {layer}")
        if target_phone not in self.labels:
            raise ValueError(f"phone prototypes contain no label {target_phone!r}")
        if len(self.labels) < 2:
            raise ValueError("phone prototype contrast requires at least two labels")
        if encoder_lens.source_means is None or encoder_lens.target_mean is None:
            raise ValueError("phone prototype pullback requires a centered encoder lens")
        if (
            reference_residual.ndim != 1
            or reference_residual.shape[0] != encoder_lens.source_dim
        ):
            raise ValueError(
                "reference_residual must have shape [encoder_lens.source_dim]"
            )
        if not bool(torch.isfinite(reference_residual).all()):
            raise ValueError("reference_residual contains non-finite values")

        target_index = self.labels.index(target_phone)
        with torch.enable_grad():
            source = (
                reference_residual.detach()
                .to(dtype=torch.float32)
                .clone()
                .requires_grad_(True)
            )
            target_mean = encoder_lens.target_mean.to(
                device=source.device, dtype=source.dtype
            )
            with torch.no_grad():
                baseline_logits = model.unembed(target_mean.unsqueeze(0)).float()
            transported = encoder_lens.transport(source.unsqueeze(0), layer)
            logits = model.unembed(transported).float()
            expected_shape = (1, self.vocab_size)
            if tuple(logits.shape) != expected_shape:
                raise ValueError(
                    f"model unembedding returned shape {tuple(logits.shape)}, "
                    f"expected {expected_shape}"
                )
            if tuple(baseline_logits.shape) != expected_shape:
                raise ValueError(
                    "model unembedding returned an invalid target-mean baseline shape"
                )
            if not bool(torch.isfinite(logits).all()) or not bool(
                torch.isfinite(baseline_logits).all()
            ):
                raise ValueError("phone prototype readout logits are non-finite")

            prototypes = self.prototypes[layer].to(
                device=logits.device, dtype=logits.dtype
            )
            readout_logits = logits - baseline_logits.to(logits.device)
            scores = (readout_logits @ prototypes.T)[0]
            other_mask = torch.ones(
                len(self.labels), device=scores.device, dtype=torch.bool
            )
            other_mask[target_index] = False
            target_score = scores[target_index]
            other_mean_score = scores[other_mask].mean()
            objective = target_score - other_mean_score
            if not bool(torch.isfinite(objective)):
                raise ValueError("phone prototype pullback objective is non-finite")
            try:
                (gradient,) = torch.autograd.grad(objective, source)
            except RuntimeError as error:
                raise ValueError(
                    "model unembedding must preserve gradients for phone pullback"
                ) from error

        direction = gradient.detach()
        if direction.ndim != 1 or direction.shape[0] != encoder_lens.source_dim:
            raise ValueError("phone prototype pullback returned an invalid direction")
        if not bool(torch.isfinite(direction).all()):
            raise ValueError("phone prototype pullback direction is non-finite")
        gradient_norm = float(direction.float().norm().cpu())
        if gradient_norm <= 1e-12:
            raise ValueError("phone prototype pullback direction has zero norm")
        return PhonePrototypePullback(
            direction=direction,
            layer=layer,
            target_phone=target_phone,
            objective_value=float(objective.detach().cpu()),
            target_readout_score=float(target_score.detach().cpu()),
            other_mean_readout_score=float(other_mean_score.detach().cpu()),
            competing_phone_count=len(self.labels) - 1,
            gradient_l2_norm=gradient_norm,
        )

    @torch.no_grad()
    def score_layer(
        self,
        model: Any,
        encoder_lens: CrossJacobianLens,
        residuals: torch.Tensor,
        *,
        layer: int,
        top_n: int = 5,
        position_chunk_size: int = 64,
    ) -> list[list[dict[str, Any]]]:
        """Score pooled encoder states against the frozen phone prototypes."""
        if layer not in self.prototypes:
            raise ValueError(f"phone prototypes contain no layer {layer}")
        if residuals.ndim != 2 or residuals.shape[1] != encoder_lens.source_dim:
            raise ValueError("residuals must have shape [positions, encoder_dim]")
        if top_n <= 0:
            raise ValueError("top_n must be positive")
        if position_chunk_size <= 0:
            raise ValueError("position_chunk_size must be positive")
        if encoder_lens.target_mean is None:
            raise ValueError("phone signatures require a centered encoder lens")

        baseline = model.unembed(
            encoder_lens.target_mean.to(
                device=residuals.device, dtype=torch.float32
            ).unsqueeze(0)
        ).float()
        prototype = self.prototypes[layer]
        rows: list[list[dict[str, Any]]] = []
        k = self.signature_top_k
        returned = min(top_n, len(self.labels))
        for start in range(0, residuals.shape[0], position_chunk_size):
            source = residuals[start : start + position_chunk_size].float()
            transported = encoder_lens.transport(source, layer)
            logits = model.unembed(transported).float() - baseline.to(source.device)
            values, token_ids = logits.topk(k + 1, dim=-1)
            weights = (values[:, :k] - values[:, k : k + 1]).clamp_min(0)
            norms = torch.linalg.vector_norm(weights, dim=1, keepdim=True)
            usable = norms[:, 0] > 1e-12
            weights = weights / norms.clamp_min(1e-12)

            ids_cpu = token_ids[:, :k].to(device="cpu", dtype=torch.long)
            weights_cpu = weights.to(device="cpu", dtype=torch.float32)
            gathered = prototype[:, ids_cpu]
            similarities = torch.einsum("cpk,pk->pc", gathered, weights_cpu)
            values_out, indices_out = similarities.topk(returned, dim=-1)
            for position in range(similarities.shape[0]):
                if not bool(usable[position]):
                    rows.append([])
                    continue
                position_scores = similarities[position]
                candidates: list[dict[str, Any]] = []
                for score, index in zip(
                    values_out[position].tolist(),
                    indices_out[position].tolist(),
                    strict=True,
                ):
                    rank = int((position_scores > score).sum()) + 1
                    candidates.append(
                        {
                            "phone": self.labels[index],
                            "similarity": float(score),
                            "rank": rank,
                            "rank_denominator": len(self.labels),
                            "score_kind": self.SCORE_KIND,
                            "is_probability": False,
                        }
                    )
                rows.append(candidates)
        return rows

    def public_metadata(self) -> dict[str, Any]:
        source = self.metadata.get("source", {})
        safe_source = (
            {
                key: source[key]
                for key in (
                    "split",
                    "development_or_test_opened",
                    "non_silence_rows",
                    "lens_examples",
                    "signature_name",
                )
                if key in source
            }
            if isinstance(source, Mapping)
            else {}
        )
        return {
            "available": True,
            "method": "nearest_frozen_top_k_j_signature_phone_prototype",
            "score_kind": self.SCORE_KIND,
            "signature_top_k": self.signature_top_k,
            "phone_inventory_size": len(self.labels),
            "phone_inventory": self.labels,
            "training_unit": "aligned_native_20_ms_phone_midpoint_state",
            "display_unit": "pooled_encoder_window",
            "silence_or_unknown_class_available": False,
            "prototype_source": safe_source,
            "interpretation": (
                "exploratory cosine similarity to frozen phone prototypes; "
                "not probability, model confidence, or causal attribution"
            ),
        }
