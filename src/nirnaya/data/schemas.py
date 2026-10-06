"""Versioned canonical sample schema shared by both model variants."""

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

from nirnaya.types import Primitive


def serialize_state(value: str | Mapping[str, object] | Sequence[object]) -> str:
    """Serialize structured state deterministically for every model backend."""
    if isinstance(value, str):
        return value
    if not isinstance(value, (Mapping, list, tuple)):
        raise TypeError("state must be a string, mapping, list, or tuple")
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as error:
        raise TypeError("state must contain JSON-serializable values") from error


def render_description(value: object) -> str:
    """Render one optional criterion description deterministically."""
    if value in (None, ""):
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def render_outcomes(sample: "CanonicalSample") -> tuple[str, ...]:
    """Render model-facing outcome text without changing semantic labels."""
    if sample.primitive is Primitive.NOUL:
        descriptions = sample.option_descriptions or ("", "")
        false_description = descriptions[0] or "no, the statement does not hold"
        true_description = descriptions[1] or "yes, the statement holds"
        return (f"false: {false_description}", f"true: {true_description}")
    labels = sample.options
    if not sample.option_descriptions:
        return labels
    if sample.primitive is Primitive.SCORE:
        return tuple(
            f"level {index}: {description or label}"
            for index, (label, description) in enumerate(
                zip(labels, sample.option_descriptions, strict=True)
            )
        )
    return tuple(
        label if not description else f"{label}: {description}"
        for label, description in zip(labels, sample.option_descriptions, strict=True)
    )


def _validate_options(options: tuple[str, ...]) -> None:
    if any(not isinstance(option, str) or not option.strip() for option in options):
        raise ValueError("options must be non-empty strings")
    normalized = [option.strip().casefold() for option in options]
    if len(normalized) != len(set(normalized)):
        raise ValueError("options must be unique after normalization")


def _validate_source_fields(
    identifier: str,
    lineage_id: str,
    state: str,
    question: str,
    metadata: Mapping[str, object],
) -> Mapping[str, object]:
    if not identifier or not lineage_id:
        raise ValueError("id and lineage_id must not be empty")
    if not isinstance(state, str):
        raise TypeError("state must be a string")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping")
    for name in ("source", "source_split"):
        value = metadata.get(name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"metadata.{name} must be a non-empty string")
    return MappingProxyType(dict(metadata))


@dataclass(frozen=True, slots=True)
class ChoiceSourceRecord:
    """Choice source row retaining its full label pool before augmentation."""

    id: str
    lineage_id: str
    state: str
    question: str
    correct_option: str
    candidate_pool: tuple[str, ...]
    metadata: Mapping[str, object]
    primitive: Primitive = field(default=Primitive.CHOICE, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "metadata",
            _validate_source_fields(
                self.id, self.lineage_id, self.state, self.question, self.metadata
            ),
        )
        _validate_options(self.candidate_pool)
        if self.correct_option.strip().casefold() not in {
            option.strip().casefold() for option in self.candidate_pool
        }:
            raise ValueError("correct_option must appear in candidate_pool")


@dataclass(frozen=True, slots=True)
class NoulSourceRecord:
    """Binary source row before canonical materialization."""

    id: str
    lineage_id: str
    state: str
    question: str
    target: bool
    metadata: Mapping[str, object]
    primitive: Primitive = field(default=Primitive.NOUL, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "metadata",
            _validate_source_fields(
                self.id, self.lineage_id, self.state, self.question, self.metadata
            ),
        )
        if not isinstance(self.target, bool):
            raise TypeError("noul target must be a boolean")


@dataclass(frozen=True, slots=True)
class ScoreSourceRecord:
    """Ordinal source row with an explicit rubric before canonical materialization."""

    id: str
    lineage_id: str
    state: str
    question: str
    options: tuple[str, ...]
    target: int
    metadata: Mapping[str, object]
    primitive: Primitive = field(default=Primitive.SCORE, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "metadata",
            _validate_source_fields(
                self.id, self.lineage_id, self.state, self.question, self.metadata
            ),
        )
        _validate_options(self.options)
        if len(self.options) < 2:
            raise ValueError("score source records require at least two rubric levels")
        if isinstance(self.target, bool) or not isinstance(self.target, int):
            raise TypeError("score target must be an integer index")
        if not 0 <= self.target < len(self.options):
            raise ValueError("score target is outside the rubric")


@dataclass(frozen=True, slots=True)
class TypedDecisionsCaseRecord:
    """One source case retaining all linked decisions until role assignment."""

    id: str
    lineage_id: str
    state: str
    question: str
    decisions: tuple["CanonicalSample", ...]
    metadata: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "metadata",
            _validate_source_fields(
                self.id, self.lineage_id, self.state, self.question, self.metadata
            ),
        )
        if len(self.decisions) != 5:
            raise ValueError("typed-decisions cases must contain exactly five decisions")
        if any(decision.lineage_id != self.lineage_id for decision in self.decisions):
            raise ValueError("typed-decisions descendants must share the case lineage")
        if any(decision.state != self.state for decision in self.decisions):
            raise ValueError("typed-decisions descendants must share the case state")


SourceRecord = ChoiceSourceRecord | NoulSourceRecord | ScoreSourceRecord | TypedDecisionsCaseRecord


@dataclass(frozen=True, slots=True)
class CanonicalSample:
    """Tokenizer-independent representation of one supervised decision."""

    id: str
    lineage_id: str
    primitive: Primitive
    state: str
    question: str
    target: int | bool
    options: tuple[str, ...] = ()
    option_descriptions: tuple[str, ...] = ()
    target_distribution: tuple[float, ...] = ()
    teacher_score: float | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version not in {1, 2}:
            raise ValueError(f"unsupported schema version: {self.schema_version}")
        if self.schema_version == 1 and (
            self.option_descriptions or self.target_distribution or self.teacher_score is not None
        ):
            raise ValueError("rich descriptions and teacher targets require schema version 2")
        if not self.id or not self.lineage_id:
            raise ValueError("id and lineage_id must not be empty")
        if not isinstance(self.state, str):
            raise TypeError("state must be a string")
        if not isinstance(self.question, str) or not self.question.strip():
            raise ValueError("question must be a non-empty string")
        if not isinstance(self.primitive, Primitive):
            raise TypeError("primitive must be a Primitive")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        source = self.metadata.get("source")
        if not isinstance(source, str) or not source.strip():
            raise ValueError("metadata.source must be a non-empty string")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

        if self.primitive is Primitive.NOUL:
            if self.options:
                raise ValueError("noul samples must not define options")
            if not isinstance(self.target, bool):
                raise TypeError("noul target must be a boolean")
            self._validate_enriched_fields(2, int(self.target))
            if self.teacher_score is not None:
                raise ValueError("teacher_score is only valid for score samples")
            return

        _validate_options(self.options)
        if isinstance(self.target, bool) or not isinstance(self.target, int):
            raise TypeError("choice and score targets must be integer indexes")
        if self.target < 0 or self.target >= len(self.options):
            raise ValueError("target index is outside the options")
        if self.primitive is Primitive.CHOICE and not 2 <= len(self.options) <= 10:
            raise ValueError("choice samples must contain 2-10 options")
        if self.primitive is Primitive.SCORE and len(self.options) < 2:
            raise ValueError("score samples must contain at least 2 ordered options")
        self._validate_enriched_fields(len(self.options), self.target)
        if self.primitive is not Primitive.SCORE and self.teacher_score is not None:
            raise ValueError("teacher_score is only valid for score samples")
        if self.teacher_score is not None and (
            isinstance(self.teacher_score, bool)
            or not isinstance(self.teacher_score, (int, float))
            or not math.isfinite(self.teacher_score)
            or not 0 <= self.teacher_score <= len(self.options) - 1
        ):
            raise ValueError("teacher_score must be finite and inside the score rubric")

    def _validate_enriched_fields(self, outcomes: int, target: int) -> None:
        if self.option_descriptions:
            if len(self.option_descriptions) != outcomes:
                raise ValueError("option descriptions must align with every outcome")
            if any(not isinstance(value, str) for value in self.option_descriptions):
                raise TypeError("option descriptions must be strings")
        if not self.target_distribution:
            return
        if len(self.target_distribution) != outcomes:
            raise ValueError("target distribution must align with every outcome")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
            for value in self.target_distribution
        ):
            raise ValueError("target distribution must contain finite non-negative values")
        if not math.isclose(sum(self.target_distribution), 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError("target distribution must sum to one")
        if not math.isclose(
            self.target_distribution[target],
            max(self.target_distribution),
            rel_tol=1e-6,
            abs_tol=1e-6,
        ):
            raise ValueError("hard target must agree with the teacher-distribution argmax")

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "CanonicalSample":
        """Parse and validate a JSON-compatible canonical sample."""
        try:
            primitive = Primitive(value["primitive"])
            sample_id = str(value["id"])
            raw_target = value["target"]
        except (KeyError, ValueError) as error:
            raise ValueError("sample is missing a valid id, primitive, or target") from error

        target: int | bool
        if primitive is Primitive.NOUL and isinstance(raw_target, int) and raw_target in (0, 1):
            target = bool(raw_target)
        else:
            target = raw_target  # type: ignore[assignment]

        raw_options = value.get("options", ())
        if isinstance(raw_options, str) or not isinstance(raw_options, (list, tuple)):
            raise TypeError("options must be a list or tuple")
        raw_metadata = value.get("metadata", {})
        if not isinstance(raw_metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        raw_descriptions = value.get("option_descriptions", ())
        if isinstance(raw_descriptions, str) or not isinstance(raw_descriptions, (list, tuple)):
            raise TypeError("option_descriptions must be a list or tuple")
        raw_distribution = value.get("target_distribution", ())
        if isinstance(raw_distribution, str) or not isinstance(raw_distribution, (list, tuple)):
            raise TypeError("target_distribution must be a list or tuple")

        return cls(
            id=sample_id,
            lineage_id=str(value.get("lineage_id", sample_id)),
            primitive=primitive,
            state=value.get("state", ""),  # type: ignore[arg-type]
            question=value.get("question", ""),  # type: ignore[arg-type]
            target=target,
            options=tuple(raw_options),
            option_descriptions=tuple(raw_descriptions),
            target_distribution=tuple(raw_distribution),
            teacher_score=value.get("teacher_score"),  # type: ignore[arg-type]
            metadata=raw_metadata,
            schema_version=int(value.get("schema_version", 1)),
        )

    def to_dict(self) -> dict[str, object]:
        """Return the stable JSON-compatible representation."""
        value: dict[str, object] = {
            "schema_version": self.schema_version,
            "id": self.id,
            "lineage_id": self.lineage_id,
            "primitive": self.primitive.value,
            "state": self.state,
            "question": self.question,
            "target": int(self.target) if isinstance(self.target, bool) else self.target,
            "metadata": dict(self.metadata),
        }
        if self.options:
            value["options"] = list(self.options)
        if self.option_descriptions:
            value["option_descriptions"] = list(self.option_descriptions)
        if self.target_distribution:
            value["target_distribution"] = list(self.target_distribution)
        if self.teacher_score is not None:
            value["teacher_score"] = self.teacher_score
        return value
