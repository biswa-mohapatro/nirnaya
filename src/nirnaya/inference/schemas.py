"""Typed public request and result schemas."""

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from nirnaya.data.schemas import render_description
from nirnaya.types import Primitive


def _validate_labels(labels: tuple[str, ...], *, minimum: int, maximum: int | None) -> None:
    if len(labels) < minimum or (maximum is not None and len(labels) > maximum):
        limit = f"{minimum}-{maximum}" if maximum is not None else f"at least {minimum}"
        raise ValueError(f"options must contain {limit} labels")
    if any(not label.strip() for label in labels):
        raise ValueError("option labels must not be empty")
    if len({label.strip().casefold() for label in labels}) != len(labels):
        raise ValueError("option labels must be unique after normalization")


def _validate_descriptions(descriptions: tuple[str, ...], outcomes: int) -> None:
    if descriptions and len(descriptions) != outcomes:
        raise ValueError("option descriptions must align with every outcome")
    if any(not isinstance(description, str) for description in descriptions):
        raise TypeError("option descriptions must be strings")


@dataclass(frozen=True, slots=True)
class ChoiceQuestion:
    question: str
    options: tuple[str, ...]
    option_descriptions: tuple[str, ...] = ()
    type: Primitive = field(default=Primitive.CHOICE, init=False)

    def __post_init__(self) -> None:
        if not self.question.strip():
            raise ValueError("question must not be empty")
        _validate_labels(self.options, minimum=2, maximum=10)
        _validate_descriptions(self.option_descriptions, len(self.options))


@dataclass(frozen=True, slots=True)
class NoulQuestion:
    question: str
    option_descriptions: tuple[str, ...] = ()
    type: Primitive = field(default=Primitive.NOUL, init=False)

    def __post_init__(self) -> None:
        if not self.question.strip():
            raise ValueError("question must not be empty")
        _validate_descriptions(self.option_descriptions, 2)


@dataclass(frozen=True, slots=True)
class ScoreQuestion:
    question: str
    options: tuple[str, ...]
    option_descriptions: tuple[str, ...] = ()
    type: Primitive = field(default=Primitive.SCORE, init=False)

    def __post_init__(self) -> None:
        if not self.question.strip():
            raise ValueError("question must not be empty")
        _validate_labels(self.options, minimum=2, maximum=None)
        _validate_descriptions(self.option_descriptions, len(self.options))


Question = ChoiceQuestion | NoulQuestion | ScoreQuestion


def parse_questions(value: Mapping[str, Mapping[str, object]]) -> dict[str, Question]:
    """Parse the public dictionary request format into validated questions."""
    questions: dict[str, Question] = {}
    for question_id, definition in value.items():
        if not question_id:
            raise ValueError("question identifiers must not be empty")
        try:
            primitive = Primitive(definition["type"])
        except (KeyError, ValueError) as error:
            raise ValueError(f"question {question_id!r} has an invalid type") from error
        prompt = definition.get("question", definition.get("instructions"))
        if not isinstance(prompt, str):
            raise TypeError(f"question {question_id!r} must contain question text")

        if primitive is Primitive.NOUL:
            if "options" in definition:
                raise ValueError(f"noul question {question_id!r} must not define options")
            raw_criteria = definition.get("criteria")
            if raw_criteria is None:
                descriptions = ()
            elif isinstance(raw_criteria, Mapping):
                if set(raw_criteria) != {"false", "true"}:
                    raise ValueError(
                        f"noul question {question_id!r} criteria must define false and true"
                    )
                descriptions = (
                    render_description(raw_criteria["false"]),
                    render_description(raw_criteria["true"]),
                )
            else:
                raise TypeError(f"noul question {question_id!r} criteria must be a mapping")
            questions[question_id] = NoulQuestion(prompt, descriptions)
            continue

        raw_criteria = definition.get("criteria")
        if raw_criteria is not None and "options" in definition:
            raise ValueError(f"question {question_id!r} must define options or criteria, not both")
        if primitive is Primitive.CHOICE and isinstance(raw_criteria, Mapping):
            options = tuple(raw_criteria)
            descriptions = tuple(render_description(value) for value in raw_criteria.values())
        elif primitive is Primitive.SCORE and isinstance(raw_criteria, (list, tuple)):
            options = tuple(str(index) for index in range(len(raw_criteria)))
            descriptions = tuple(render_description(value) for value in raw_criteria)
        else:
            raw_options = definition.get("options")
            if isinstance(raw_options, str) or not isinstance(raw_options, (list, tuple)):
                raise TypeError(f"question {question_id!r} options must be a list or tuple")
            options = tuple(raw_options)
            if not all(isinstance(option, str) for option in options):
                raise TypeError(f"question {question_id!r} options must be strings")
            descriptions = ()
        if primitive is Primitive.CHOICE:
            questions[question_id] = ChoiceQuestion(prompt, options, descriptions)
        else:
            questions[question_id] = ScoreQuestion(prompt, options, descriptions)
    return questions


@dataclass(frozen=True, slots=True)
class CalibrationMetadata:
    """Calibration provenance applied to one prediction."""

    method: str
    key: str
    temperature: float
    artifact_version: int = 1

    def __post_init__(self) -> None:
        if not self.method or not self.key:
            raise ValueError("calibration method and key must not be empty")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be finite and positive")


@dataclass(frozen=True, slots=True)
class DecisionResult:
    """Uniform result envelope used for every primitive."""

    type: Primitive
    decision: str | bool | float | None
    confidence: float
    abstained: bool
    distribution: Mapping[str, float]
    calibration: CalibrationMetadata
    details: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.distribution:
            raise ValueError("distribution must not be empty")
        probabilities = tuple(self.distribution.values())
        if any(not math.isfinite(value) or value < 0 or value > 1 for value in probabilities):
            raise ValueError("distribution probabilities must be finite and between 0 and 1")
        if not math.isclose(sum(probabilities), 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError("distribution probabilities must sum to 1")
        if not math.isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be finite and between 0 and 1")
        if not math.isclose(self.confidence, max(probabilities), rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError("confidence must equal the maximum calibrated probability")
        if self.abstained and self.decision is not None:
            raise ValueError("an abstained result must not contain a decision")
        if not self.abstained and self.decision is None:
            raise ValueError("a non-abstained result must contain a decision")
        object.__setattr__(self, "distribution", MappingProxyType(dict(self.distribution)))
        object.__setattr__(self, "details", MappingProxyType(dict(self.details)))

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible public result."""
        return {
            "type": self.type.value,
            "decision": self.decision,
            "confidence": self.confidence,
            "abstained": self.abstained,
            "distribution": dict(self.distribution),
            "calibration": {
                "method": self.calibration.method,
                "key": self.calibration.key,
                "temperature": self.calibration.temperature,
                "artifact_version": self.calibration.artifact_version,
            },
            **dict(self.details),
        }
