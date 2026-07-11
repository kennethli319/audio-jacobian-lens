# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Local server backend for the MLX Chatterbox Frame Trace page."""

from __future__ import annotations

import hashlib
import json
import threading
import uuid
from collections import OrderedDict
from typing import Any

import torch

from jlens.chatterbox_fitting import (
    chatterbox_speech_lens_logits,
    validate_chatterbox_speech_lens,
)
from jlens.mlx_chatterbox import (
    DEFAULT_CHATTERBOX_MODEL_ID,
    DEFAULT_CHATTERBOX_MODEL_REVISION,
    DEFAULT_S3_TOKENIZER_ID,
    DEFAULT_S3_TOKENIZER_REVISION,
    ChatterboxCapturedRun,
    ChatterboxGenerationConfig,
    MLXChatterboxModel,
)
from jlens.projected_lens import ProjectedCrossJacobianLens
from jlens.webapp import AnalysisBusyError


class UnknownChatterboxRunError(KeyError):
    """Raised when an evicted or unknown in-memory analysis is requested."""


class MLXChatterboxBackend:
    """Own one local Chatterbox model and a bounded in-memory run cache."""

    # MLX-Audio's compiled generation graph must remain on the Uvicorn thread
    # that loaded the model. create_app registers async routes for this backend.
    requires_server_thread = True

    def __init__(
        self,
        model: MLXChatterboxModel,
        lens: ProjectedCrossJacobianLens | None = None,
        *,
        max_cached_runs: int = 2,
        top_k: int = 5,
    ) -> None:
        if max_cached_runs <= 0:
            raise ValueError("max_cached_runs must be positive")
        if not 1 <= top_k <= 20:
            raise ValueError("top_k must be in [1, 20]")
        if lens is not None:
            validate_chatterbox_speech_lens(model, lens)
        self.model = model
        self.lens = lens
        self.max_cached_runs = max_cached_runs
        self.top_k = top_k
        self.lens_fingerprint = (
            None if lens is None else self._lens_fingerprint(lens)
        )
        self._runs: OrderedDict[str, ChatterboxCapturedRun] = OrderedDict()
        self._interventions: dict[str, dict[str, Any]] = {}
        self._thread_state = threading.local()
        self._lock = threading.Lock()

    @classmethod
    def load(
        cls,
        *,
        model_id: str = DEFAULT_CHATTERBOX_MODEL_ID,
        revision: str = DEFAULT_CHATTERBOX_MODEL_REVISION,
        s3_tokenizer_id: str = DEFAULT_S3_TOKENIZER_ID,
        s3_tokenizer_revision: str = DEFAULT_S3_TOKENIZER_REVISION,
        generation_config: ChatterboxGenerationConfig | None = None,
        max_cached_runs: int = 2,
        lens_path: str | None = None,
        top_k: int = 5,
    ) -> MLXChatterboxBackend:
        lens = (
            None
            if lens_path is None
            else ProjectedCrossJacobianLens.load(lens_path)
        )
        model = MLXChatterboxModel.from_pretrained(
            model_id,
            revision=revision,
            s3_tokenizer_id=s3_tokenizer_id,
            s3_tokenizer_revision=s3_tokenizer_revision,
            generation_config=generation_config,
        )
        return cls(
            model,
            lens,
            max_cached_runs=max_cached_runs,
            top_k=top_k,
        )

    @staticmethod
    def _lens_fingerprint(lens: ProjectedCrossJacobianLens) -> str:
        digest = hashlib.sha256()
        digest.update(lens.target_factors.numpy().tobytes())
        for layer in lens.source_layers:
            digest.update(layer.to_bytes(4, "big"))
            digest.update(lens.source_factors[layer].numpy().tobytes())
        digest.update(
            json.dumps(
                lens.metadata, sort_keys=True, separators=(",", ":")
            ).encode()
        )
        return digest.hexdigest()[:16]

    def _lens_summary(self) -> dict[str, Any] | None:
        if self.lens is None:
            return None
        return {
            "format": "projected-cross-jacobian-lens",
            "fingerprint": self.lens_fingerprint,
            "source_layers": self.lens.source_layers,
            "target_layer": self.lens.metadata.get("target_layer"),
            "n_examples": self.lens.n_examples,
            "target_positions": self.lens.metadata.get("target_positions"),
            "centered": self.lens.source_means is not None,
            "capture_convention": self.lens.metadata.get(
                "capture_convention"
            ),
            "source_reduction": self.lens.metadata.get("source_reduction"),
            "target_reduction": self.lens.metadata.get("target_reduction"),
            "corpus_reduction": self.lens.metadata.get("corpus_reduction"),
            "model_fingerprint": self.lens.metadata.get("model_fingerprint"),
            "examples_fingerprint": self.lens.metadata.get(
                "examples_fingerprint"
            ),
            "estimator": self.lens.metadata.get("estimator"),
            "projection": {
                "method": self.lens.projection_method,
                "rank": self.lens.projection_dim,
                "target_dim": self.lens.target_dim,
                "seed": self.lens.metadata.get("projection_seed"),
                "dense_at_full_rank": self.lens.metadata.get(
                    "dense_at_full_rank"
                ),
            },
        }

    def _stream(self) -> Any:
        import mlx.core as mx

        stream = getattr(self._thread_state, "stream", None)
        if stream is None:
            stream = mx.new_thread_local_stream(mx.gpu)
            self._thread_state.stream = stream
        return stream

    def status(self) -> dict[str, Any]:
        return {
            "ready": True,
            "backend": "mlx-chatterbox-turbo",
            "model": self.model.metadata(),
            "capabilities": {
                "text_input": True,
                "generated_audio": True,
                "nominal_frame_to_code": True,
                "code_to_text_gradient": True,
                "text_prefix_self_attention": True,
                "speech_head_candidates": callable(
                    getattr(self.model, "speech_head_candidate_payload", None)
                ),
                "forced_code_branching": callable(
                    getattr(self.model, "branch_synthesis", None)
                ),
                "residual_code_steering": callable(
                    getattr(self.model, "residual_branch_synthesis", None)
                ),
                "s3_frame_jacobian": False,
                "fitted_speech_code_jlens": self.lens is not None,
            },
            "speech_code_jlens": self._lens_summary(),
            "message": (
                "Chatterbox fitted speech-code J-lens and local text trace "
                "ready on Apple Metal"
                if self.lens is not None
                else "Chatterbox local text trace ready on Apple Metal; no "
                "compatible fitted speech-code lens was loaded"
            ),
        }

    def _fitted_lens_payload(
        self,
        run: ChatterboxCapturedRun,
        *,
        logits_by_layer: dict[int, torch.Tensor] | None = None,
    ) -> dict[str, Any] | None:
        if self.lens is None:
            return None
        if logits_by_layer is None:
            logits_by_layer = chatterbox_speech_lens_logits(
                self.model, self.lens, run
            )
        target_ids = torch.tensor(run.speech_code_ids, dtype=torch.long)
        target_probabilities: list[list[float]] = []
        target_log_probabilities: list[list[float]] = []
        target_ranks: list[list[int]] = []
        top_codes: list[list[list[dict[str, Any]]]] = []
        vocab_size = self.model.speech_vocab_size
        top_k = min(self.top_k, vocab_size)
        for layer in self.lens.source_layers:
            logits = logits_by_layer[layer].float()
            if logits.shape != (len(target_ids), vocab_size):
                raise RuntimeError(
                    f"fitted speech-head logits at L{layer} have shape "
                    f"{tuple(logits.shape)}, expected "
                    f"({len(target_ids)}, {vocab_size})"
                )
            log_probabilities = logits.log_softmax(dim=-1)
            probabilities = log_probabilities.exp()
            selected_logits = logits.gather(1, target_ids[:, None])[:, 0]
            selected_log_probabilities = log_probabilities.gather(
                1, target_ids[:, None]
            )[:, 0]
            ranks = (logits > selected_logits[:, None]).sum(dim=-1) + 1
            values, ids = probabilities.topk(top_k, dim=-1)
            target_probabilities.append(
                selected_log_probabilities.exp().tolist()
            )
            target_log_probabilities.append(
                selected_log_probabilities.tolist()
            )
            target_ranks.append([int(value) for value in ranks.tolist()])
            top_codes.append(
                [
                    [
                        {
                            "id": int(code_id),
                            "probability": float(probability),
                        }
                        for code_id, probability in zip(
                            position_ids.tolist(),
                            position_values.tolist(),
                            strict=True,
                        )
                    ]
                    for position_ids, position_values in zip(
                        ids, values, strict=True
                    )
                ]
            )
        summary = self._lens_summary()
        assert summary is not None
        return {
            "schema_version": 1,
            "layers": self.lens.source_layers,
            "target_ids": list(run.speech_code_ids),
            "target_probabilities": target_probabilities,
            "target_log_probabilities": target_log_probabilities,
            "target_ranks": target_ranks,
            "top_codes": top_codes,
            "source_coordinate": "post_block_speech_prediction_position",
            "target_head": "t3_speech_head_after_final_norm",
            "normalization": (
                "full_speech_head_softmax_before_generation_processors"
            ),
            "artifact": summary,
            "warnings": [
                "The fitted softmax value is a readout distribution, not calibrated emission confidence or percent causation.",
                "Speech-code IDs are learned acoustic symbols, not published phoneme labels.",
                "This projected artifact is a corpus average; compare held-out examples, projection ranks, and seeds before interpreting fine ordering.",
            ],
        }

    def _generation_payload(
        self,
        run: ChatterboxCapturedRun,
        *,
        fitted_logits_by_layer: dict[int, torch.Tensor] | None = None,
    ) -> dict[str, Any]:
        payload = self.model.generation_payload(run)
        candidate_builder = getattr(
            self.model, "speech_head_candidate_payload", None
        )
        if callable(candidate_builder):
            output = payload.get("output")
            if not isinstance(output, dict):
                raise RuntimeError(
                    "Chatterbox generation payload has no output object"
                )
            output["speech_head_candidates"] = candidate_builder(
                run, top_k=self.top_k
            )
        fitted_payload = self._fitted_lens_payload(
            run, logits_by_layer=fitted_logits_by_layer
        )
        if fitted_payload is not None:
            payload["fitted_speech_code_jlens"] = fitted_payload
        return payload

    @staticmethod
    def _target_values_from_torch_logits(
        logits: torch.Tensor,
        positions: list[int],
        target_code_id: int,
    ) -> tuple[list[float | None], list[int | None]]:
        probabilities: list[float | None] = []
        ranks: list[int | None] = []
        for position in positions:
            if position >= int(logits.shape[0]):
                probabilities.append(None)
                ranks.append(None)
                continue
            row = logits[position].float()
            target_logit = row[target_code_id]
            probabilities.append(float(row.softmax(dim=-1)[target_code_id]))
            ranks.append(int((row > target_logit).sum().item()) + 1)
        return probabilities, ranks

    @staticmethod
    def _target_values_from_raw_logits(
        run: ChatterboxCapturedRun,
        positions: list[int],
        target_code_id: int,
    ) -> tuple[list[float | None], list[int | None]]:
        probabilities: list[float | None] = []
        ranks: list[int | None] = []
        logits = run.raw_logits
        for position in positions:
            if position >= len(run.speech_code_ids):
                probabilities.append(None)
                ranks.append(None)
                continue
            row = torch.as_tensor(logits[position], dtype=torch.float32)
            target_logit = row[target_code_id]
            probabilities.append(float(row.softmax(dim=-1)[target_code_id]))
            ranks.append(int((row > target_logit).sum().item()) + 1)
        return probabilities, ranks

    def _residual_target_diagnostics(
        self,
        parent: ChatterboxCapturedRun,
        branch: ChatterboxCapturedRun,
        target_code_id: int,
        provenance: dict[str, Any],
        parent_fitted_logits: dict[int, torch.Tensor] | None,
        branch_fitted_logits: dict[int, torch.Tensor] | None,
    ) -> dict[str, Any]:
        positions = [int(value) for value in provenance["requested_positions"]]
        fitted_layers = [] if self.lens is None else list(self.lens.source_layers)
        before_probabilities: list[list[float | None]] = []
        after_probabilities: list[list[float | None]] = []
        before_ranks: list[list[int | None]] = []
        after_ranks: list[list[int | None]] = []
        if self.lens is not None:
            assert parent_fitted_logits is not None
            assert branch_fitted_logits is not None
            for layer in fitted_layers:
                probabilities, ranks = self._target_values_from_torch_logits(
                    parent_fitted_logits[layer], positions, target_code_id
                )
                before_probabilities.append(probabilities)
                before_ranks.append(ranks)
                probabilities, ranks = self._target_values_from_torch_logits(
                    branch_fitted_logits[layer], positions, target_code_id
                )
                after_probabilities.append(probabilities)
                after_ranks.append(ranks)

        head_before_probabilities, head_before_ranks = (
            self._target_values_from_raw_logits(
                parent, positions, target_code_id
            )
        )
        head_after_probabilities, head_after_ranks = (
            self._target_values_from_raw_logits(branch, positions, target_code_id)
        )
        edited_coordinates = [
            {
                "layer": int(coordinate["layer"]),
                "speech_code_index": int(coordinate["speech_code_index"]),
            }
            for coordinate in provenance["coordinates"]
            if bool(coordinate["applied"])
        ]
        return {
            "schema_version": 1,
            "normalization": (
                "full_speech_head_softmax_before_generation_processors"
            ),
            "positions": positions,
            "fitted_layers": fitted_layers,
            "before_probabilities": before_probabilities,
            "after_probabilities": after_probabilities,
            "before_ranks": before_ranks,
            "after_ranks": after_ranks,
            "head_before_probabilities": head_before_probabilities,
            "head_after_probabilities": head_after_probabilities,
            "head_before_ranks": head_before_ranks,
            "head_after_ranks": head_after_ranks,
            "parent_realized_ids": [
                int(parent.speech_code_ids[position]) for position in positions
            ],
            "branch_realized_ids": [
                (
                    int(branch.speech_code_ids[position])
                    if position < len(branch.speech_code_ids)
                    else None
                )
                for position in positions
            ],
            "edited_coordinates": edited_coordinates,
            "first_suffix_divergence_index": provenance[
                "first_suffix_divergence_index"
            ],
        }

    def _store_run(
        self,
        analysis_id: str,
        run: ChatterboxCapturedRun,
        *,
        intervention: dict[str, Any] | None = None,
        preserve: set[str] | None = None,
    ) -> None:
        """Store a run and evict LRU entries without splitting a fresh branch pair."""
        self._runs[analysis_id] = run
        self._runs.move_to_end(analysis_id)
        if intervention is None:
            self._interventions.pop(analysis_id, None)
        else:
            self._interventions[analysis_id] = intervention

        protected = set() if preserve is None else set(preserve)
        capacity = max(self.max_cached_runs, len(protected))
        while len(self._runs) > capacity:
            evicted_id = next(
                (
                    candidate_id
                    for candidate_id in self._runs
                    if candidate_id not in protected
                ),
                None,
            )
            if evicted_id is None:
                break
            del self._runs[evicted_id]
            self._interventions.pop(evicted_id, None)

    def synthesize(self, text: str) -> dict[str, Any]:
        if not self._lock.acquire(blocking=False):
            raise AnalysisBusyError(
                "another Chatterbox generation or trace is running; retry shortly"
            )
        try:
            import mlx.core as mx

            with mx.stream(self._stream()):
                run = self.model.synthesize(text)
                payload = self._generation_payload(run)
            analysis_id = uuid.uuid4().hex
            self._store_run(analysis_id, run)
            return {"analysis_id": analysis_id, **payload}
        finally:
            self._lock.release()

    def branch(
        self,
        analysis_id: str,
        speech_code_index: int,
        replacement_code_id: int,
    ) -> dict[str, Any]:
        parent = self._runs.get(analysis_id)
        if parent is None:
            raise UnknownChatterboxRunError(analysis_id)
        if not self._lock.acquire(blocking=False):
            raise AnalysisBusyError(
                "another Chatterbox generation, branch, or trace is running; "
                "retry shortly"
            )
        try:
            import mlx.core as mx

            branch_builder = getattr(self.model, "branch_synthesis", None)
            if not callable(branch_builder):
                raise RuntimeError(
                    "the loaded Chatterbox model does not support forced-code branches"
                )
            with mx.stream(self._stream()):
                run, provenance = branch_builder(
                    parent, speech_code_index, replacement_code_id
                )
                payload = self._generation_payload(run)

            branch_analysis_id = uuid.uuid4().hex
            intervention = {
                **provenance,
                "parent_analysis_id": analysis_id,
            }
            self._runs.move_to_end(analysis_id)
            self._store_run(
                branch_analysis_id,
                run,
                intervention=intervention,
                preserve={analysis_id, branch_analysis_id},
            )
            return {
                "analysis_id": branch_analysis_id,
                **payload,
                "intervention": intervention,
            }
        finally:
            self._lock.release()

    def residual_branch(
        self,
        analysis_id: str,
        speech_code_index: int,
        target_code_id: int,
        layers: list[int],
        forward_span: int,
        max_relative_residual_norm: float,
    ) -> dict[str, Any]:
        parent = self._runs.get(analysis_id)
        if parent is None:
            raise UnknownChatterboxRunError(analysis_id)
        if not self._lock.acquire(blocking=False):
            raise AnalysisBusyError(
                "another Chatterbox generation, branch, or trace is running; "
                "retry shortly"
            )
        try:
            import mlx.core as mx

            branch_builder = getattr(
                self.model, "residual_branch_synthesis", None
            )
            if not callable(branch_builder):
                raise RuntimeError(
                    "the loaded Chatterbox model does not support residual branches"
                )
            with mx.stream(self._stream()):
                run, provenance = branch_builder(
                    parent,
                    speech_code_index,
                    target_code_id,
                    layers,
                    forward_span,
                    max_relative_residual_norm,
                )
                parent_fitted_logits = (
                    None
                    if self.lens is None
                    else chatterbox_speech_lens_logits(
                        self.model, self.lens, parent
                    )
                )
                branch_fitted_logits = (
                    None
                    if self.lens is None
                    else chatterbox_speech_lens_logits(
                        self.model, self.lens, run
                    )
                )
                payload = self._generation_payload(
                    run, fitted_logits_by_layer=branch_fitted_logits
                )
                provenance["target_diagnostics"] = (
                    self._residual_target_diagnostics(
                        parent,
                        run,
                        target_code_id,
                        provenance,
                        parent_fitted_logits,
                        branch_fitted_logits,
                    )
                )

            branch_analysis_id = uuid.uuid4().hex
            intervention = {
                **provenance,
                "parent_analysis_id": analysis_id,
            }
            self._runs.move_to_end(analysis_id)
            self._store_run(
                branch_analysis_id,
                run,
                intervention=intervention,
                preserve={analysis_id, branch_analysis_id},
            )
            return {
                "analysis_id": branch_analysis_id,
                **payload,
                "intervention": intervention,
            }
        finally:
            self._lock.release()

    def trace(self, analysis_id: str, speech_code_index: int) -> dict[str, Any]:
        run = self._runs.get(analysis_id)
        if run is None:
            raise UnknownChatterboxRunError(analysis_id)
        if not self._lock.acquire(blocking=False):
            raise AnalysisBusyError(
                "another Chatterbox generation or trace is running; retry shortly"
            )
        try:
            import mlx.core as mx

            with mx.stream(self._stream()):
                payload = self.model.trace(run, speech_code_index)
            self._runs.move_to_end(analysis_id)
            result = {"analysis_id": analysis_id, **payload}
            intervention = self._interventions.get(analysis_id)
            if intervention is not None:
                result["intervention"] = intervention
            return result
        finally:
            self._lock.release()
