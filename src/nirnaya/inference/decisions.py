"""Framework-neutral request samples and calibrated result envelopes."""

from collections.abc import Mapping

import numpy as np

from nirnaya.data.schemas import CanonicalSample
from nirnaya.inference.calibration import TemperatureArtifact
from nirnaya.inference.schemas import (
    CalibrationMetadata,
    ChoiceQuestion,
    DecisionResult,
    NoulQuestion,
    Question,
)
from nirnaya.types import Primitive


def inference_sample(
    request_index: int,
    question_id: str,
    state: str,
    question: Question,
) -> CanonicalSample:
    options = () if isinstance(question, NoulQuestion) else question.options
    descriptions = question.option_descriptions
    return CanonicalSample(
        schema_version=2 if descriptions else 1,
        id=f"request-{request_index}:{question_id}",
        lineage_id=f"request-{request_index}:{question_id}",
        primitive=question.type,
        state=state,
        question=question.question,
        target=False if isinstance(question, NoulQuestion) else 0,
        options=options,
        option_descriptions=descriptions,
        metadata={"source": "runtime"},
    )


def question_labels(question: Question) -> tuple[str, ...]:
    if isinstance(question, NoulQuestion):
        return ("false", "true")
    return question.options


def build_result(
    question: Question,
    probabilities: np.ndarray,
    *,
    calibration: TemperatureArtifact,
    calibration_key: str,
    temperature: float,
    thresholds: Mapping[Primitive, float],
) -> DecisionResult:
    """Build the uniform result envelope from calibrated probabilities."""
    labels = question_labels(question)
    selected_index = int(probabilities.argmax())
    confidence = float(probabilities[selected_index])
    abstained = confidence < thresholds[question.type]
    distribution = {label: float(value) for label, value in zip(labels, probabilities, strict=True)}
    details: dict[str, object]
    if isinstance(question, ChoiceQuestion):
        decision: str | bool | float | None = None if abstained else labels[selected_index]
        details = {
            "selection": decision,
            "index": None if abstained else selected_index,
            "probabilities": distribution,
        }
    elif isinstance(question, NoulQuestion):
        decision = None if abstained else bool(selected_index)
        details = {
            "probability_false": float(probabilities[0]),
            "probability_true": float(probabilities[1]),
        }
    else:
        expected_score = float(np.dot(np.arange(len(probabilities)), probabilities))
        decision = None if abstained else expected_score
        details = {"expected_score": expected_score}
    return DecisionResult(
        type=question.type,
        decision=decision,
        confidence=confidence,
        abstained=abstained,
        distribution=distribution,
        calibration=CalibrationMetadata(
            method=calibration.method,
            key=calibration_key,
            temperature=temperature,
            artifact_version=calibration.artifact_version,
        ),
        details=details,
    )
