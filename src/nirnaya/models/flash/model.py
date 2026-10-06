"""Bidirectional marker-based Nirnaya-Flash decision model."""

from collections.abc import Mapping

import mlx.core as mx
import mlx.nn as nn

from nirnaya.contracts import DecisionLogits, FlashBatch
from nirnaya.models.base import BaseDecisionModel
from nirnaya.models.decision_head import DecisionHead
from nirnaya.models.flash.modernbert import ModernBert
from nirnaya.models.heads import ChoiceHead, NoulHead, ScoreHead
from nirnaya.types import Primitive

_PRIMITIVE_INDEX = {
    Primitive.CHOICE: 0,
    Primitive.NOUL: 1,
    Primitive.SCORE: 2,
}
_MASKED_LOGIT = -1.0e4
LEGACY_ARCHITECTURE = "primitive_heads_v1"
SHARED_MARKER_ARCHITECTURE = "shared_marker_v2"
_ARCHITECTURES = frozenset({LEGACY_ARCHITECTURE, SHARED_MARKER_ARCHITECTURE})


def architecture_version_from_config(model_config: Mapping[str, object]) -> str:
    value = model_config.get("architecture_version", LEGACY_ARCHITECTURE)
    if not isinstance(value, str):
        raise ValueError("model architecture_version must be a string")
    return value


class FlashDecisionModel(nn.Module, BaseDecisionModel):
    """ModernBERT plus a decision transformer and primitive-specific heads."""

    def __init__(
        self,
        encoder: ModernBert,
        decision_layers: int = 2,
        *,
        architecture_version: str = LEGACY_ARCHITECTURE,
    ):
        super().__init__()
        if architecture_version not in _ARCHITECTURES:
            raise ValueError(f"unsupported Flash architecture: {architecture_version}")
        self.architecture_version = architecture_version
        self.encoder = encoder
        hidden_size = encoder.config.hidden_size
        self.type_embedding = nn.Embedding(len(_PRIMITIVE_INDEX), hidden_size)
        self.decision_head = DecisionHead(
            hidden_size,
            decision_layers,
            activation="gelu" if architecture_version == SHARED_MARKER_ARCHITECTURE else "relu",
            dropout=0.1 if architecture_version == SHARED_MARKER_ARCHITECTURE else 0.0,
        )
        if architecture_version == SHARED_MARKER_ARCHITECTURE:
            self.marker_head = ChoiceHead(hidden_size)
        else:
            self.choice_head = ChoiceHead(hidden_size)
            self.noul_head = NoulHead(hidden_size)
            self.score_head = ScoreHead(hidden_size)

    def set_trainable_strategy(self, strategy: str) -> None:
        """Configure one independent Flash fine-tuning stage."""
        if strategy == "full":
            self.unfreeze()
            return
        layer_counts = {"head_only": 0, "top_4": 4, "top_8": 8}
        if strategy not in layer_counts:
            raise ValueError("strategy must be head_only, top_4, top_8, or full")
        count = layer_counts[strategy]
        if count > len(self.encoder.layers):
            raise ValueError("unfreeze strategy exceeds the encoder depth")

        self.freeze()
        self.type_embedding.unfreeze()
        self.decision_head.unfreeze()
        if self.architecture_version == SHARED_MARKER_ARCHITECTURE:
            self.marker_head.unfreeze()
        else:
            self.choice_head.unfreeze()
            self.noul_head.unfreeze()
            self.score_head.unfreeze()
        if count:
            self.encoder.final_norm.unfreeze()
            for layer in self.encoder.layers[-count:]:
                layer.unfreeze()

    def forward(self, batch: object) -> DecisionLogits:
        if not isinstance(batch, FlashBatch):
            raise TypeError("FlashDecisionModel requires a FlashBatch")
        if not isinstance(batch.input_ids, mx.array) or not isinstance(
            batch.attention_mask, mx.array
        ):
            raise TypeError("FlashDecisionModel requires MLX input arrays")
        if not isinstance(batch.marker_positions, mx.array) or not isinstance(
            batch.marker_mask, mx.array
        ):
            raise TypeError("FlashDecisionModel requires MLX marker arrays")

        primitive_indices = mx.array([_PRIMITIVE_INDEX[item] for item in batch.primitives])
        logits, valid_mask = self.forward_arrays(
            batch.input_ids,
            batch.attention_mask,
            batch.marker_positions,
            batch.marker_mask,
            primitive_indices,
        )
        option_counts = tuple(int(value) for value in mx.sum(valid_mask, axis=1).tolist())

        return DecisionLogits(
            logits=logits,
            valid_mask=valid_mask,
            primitives=batch.primitives,
            question_ids=batch.question_ids,
            option_counts=option_counts,
        )

    def forward_arrays(
        self,
        input_ids: mx.array,
        attention_mask: mx.array,
        marker_positions: mx.array,
        marker_mask: mx.array,
        primitive_indices: mx.array,
    ) -> tuple[mx.array, mx.array]:
        """Return raw arrays without Python synchronization for training."""
        hidden_states = self.encoder(input_ids, attention_mask)
        hidden_states = hidden_states + self.type_embedding(primitive_indices)[:, None, :]
        hidden_states = self.decision_head(hidden_states, attention_mask)

        marker_positions = mx.maximum(marker_positions, 0)
        marker_states = hidden_states[
            mx.arange(hidden_states.shape[0])[:, None],
            marker_positions,
        ]
        if marker_states.shape[1] < 2:
            raise ValueError("Flash batches require at least two marker slots")
        if self.architecture_version == SHARED_MARKER_ARCHITECTURE:
            logits = self.marker_head(marker_states)
        else:
            choice_logits = self.choice_head(marker_states)
            score_logits = self.score_head(marker_states)
            noul_logits = self.noul_head(hidden_states[:, 0])
            if choice_logits.shape[1] > 2:
                noul_logits = mx.pad(
                    noul_logits,
                    [(0, 0), (0, choice_logits.shape[1] - 2)],
                    constant_values=_MASKED_LOGIT,
                )

            logits = mx.where(
                primitive_indices[:, None] == _PRIMITIVE_INDEX[Primitive.CHOICE],
                choice_logits,
                mx.where(
                    primitive_indices[:, None] == _PRIMITIVE_INDEX[Primitive.NOUL],
                    noul_logits,
                    score_logits,
                ),
            )
        valid_mask = marker_mask.astype(mx.bool_)
        logits = mx.where(valid_mask, logits, _MASKED_LOGIT)
        return logits, valid_mask

    def __call__(self, batch: object) -> DecisionLogits:
        return self.forward(batch)
