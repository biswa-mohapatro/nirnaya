"""Model-neutral array contracts shared by Flash and Core."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from nirnaya.types import Primitive


class ArrayLike(Protocol):
    """Minimum shape interface required from NumPy and MLX arrays."""

    @property
    def shape(self) -> Sequence[int]: ...


def _shape(value: ArrayLike, name: str, dimensions: int) -> tuple[int, ...]:
    shape = tuple(value.shape)
    if len(shape) != dimensions:
        raise ValueError(f"{name} must have {dimensions} dimensions; got {shape}")
    return shape


@dataclass(frozen=True, slots=True)
class FlashBatch:
    """Padded Flash input with one row per logical question."""

    input_ids: ArrayLike
    attention_mask: ArrayLike
    marker_positions: ArrayLike
    marker_mask: ArrayLike
    primitives: tuple[Primitive, ...]
    question_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        input_shape = _shape(self.input_ids, "input_ids", 2)
        if _shape(self.attention_mask, "attention_mask", 2) != input_shape:
            raise ValueError("attention_mask must match input_ids")
        marker_shape = _shape(self.marker_positions, "marker_positions", 2)
        if _shape(self.marker_mask, "marker_mask", 2) != marker_shape:
            raise ValueError("marker_mask must match marker_positions")
        if marker_shape[0] != input_shape[0]:
            raise ValueError("marker rows must match input rows")
        _validate_logical_rows(input_shape[0], self.primitives, self.question_ids)


@dataclass(frozen=True, slots=True)
class CoreBatch:
    """Padded Core input with one causal row per candidate outcome."""

    input_ids: ArrayLike
    attention_mask: ArrayLike
    group_indices: tuple[int, ...]
    option_indices: tuple[int, ...]
    primitives: tuple[Primitive, ...]
    question_ids: tuple[str, ...]
    option_counts: tuple[int, ...]

    def __post_init__(self) -> None:
        input_shape = _shape(self.input_ids, "input_ids", 2)
        if _shape(self.attention_mask, "attention_mask", 2) != input_shape:
            raise ValueError("attention_mask must match input_ids")
        candidate_rows = input_shape[0]
        if len(self.group_indices) != candidate_rows or len(self.option_indices) != candidate_rows:
            raise ValueError("candidate indices must match input rows")
        logical_rows = len(self.option_counts)
        _validate_logical_rows(logical_rows, self.primitives, self.question_ids)
        if any(index < 0 or index >= logical_rows for index in self.group_indices):
            raise ValueError("group_indices contains an unknown logical question")
        for group_index, option_index in zip(self.group_indices, self.option_indices, strict=True):
            if option_index < 0 or option_index >= self.option_counts[group_index]:
                raise ValueError("option_indices contains an invalid candidate index")
        for group_index, option_count in enumerate(self.option_counts):
            actual = sorted(
                option_index
                for candidate_group, option_index in zip(
                    self.group_indices, self.option_indices, strict=True
                )
                if candidate_group == group_index
            )
            if actual != list(range(option_count)):
                raise ValueError("each logical question must contain every candidate exactly once")


@dataclass(frozen=True, slots=True)
class DecisionLogits:
    """Raw outcome logits returned by every Nirnaya model backend."""

    logits: ArrayLike
    valid_mask: ArrayLike
    primitives: tuple[Primitive, ...]
    question_ids: tuple[str, ...]
    option_counts: tuple[int, ...]
    metadata: tuple[Mapping[str, object], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        logits_shape = _shape(self.logits, "logits", 2)
        if _shape(self.valid_mask, "valid_mask", 2) != logits_shape:
            raise ValueError("valid_mask must match logits")
        rows, width = logits_shape
        _validate_logical_rows(rows, self.primitives, self.question_ids)
        if len(self.option_counts) != rows:
            raise ValueError("option_counts must match logits rows")
        if any(count < 2 or count > width for count in self.option_counts):
            raise ValueError("option_counts must be between 2 and the logits width")
        for primitive, count in zip(self.primitives, self.option_counts, strict=True):
            if primitive is Primitive.NOUL and count != 2:
                raise ValueError("noul logits must represent false and true outcomes")
            if primitive is Primitive.CHOICE and count > 10:
                raise ValueError("choice logits support at most 10 outcomes")
        if self.metadata and len(self.metadata) != rows:
            raise ValueError("metadata must be empty or match logits rows")


def _validate_logical_rows(
    rows: int, primitives: tuple[Primitive, ...], question_ids: tuple[str, ...]
) -> None:
    if len(primitives) != rows or len(question_ids) != rows:
        raise ValueError("primitive and question identifiers must match logical rows")
    if any(not question_id for question_id in question_ids):
        raise ValueError("question identifiers must not be empty")
