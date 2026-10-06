"""Framework-neutral temperature calibration for typed decision logits."""

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from nirnaya.types import Primitive


@dataclass(frozen=True, slots=True)
class CalibrationObservation:
    """One held-out raw-logit row used only for temperature fitting."""

    logits: tuple[float, ...]
    target: int
    primitive: Primitive

    def __post_init__(self) -> None:
        if len(self.logits) < 2 or not 0 <= self.target < len(self.logits):
            raise ValueError("calibration observation has an invalid target or cardinality")
        if any(not math.isfinite(value) for value in self.logits):
            raise ValueError("calibration logits must be finite")


@dataclass(frozen=True, slots=True)
class TemperatureArtifact:
    """Hierarchical temperatures fitted on a dedicated calibration split."""

    global_temperature: float = 1.0
    by_primitive: Mapping[str, float] = field(default_factory=dict)
    by_cardinality: Mapping[str, float] = field(default_factory=dict)
    method: str = "none"
    artifact_version: int = 1

    def __post_init__(self) -> None:
        temperatures = (
            self.global_temperature,
            *self.by_primitive.values(),
            *self.by_cardinality.values(),
        )
        if any(not math.isfinite(value) or value <= 0 for value in temperatures):
            raise ValueError("calibration temperatures must be finite and positive")

    def select(self, primitive: Primitive, cardinality: int) -> tuple[str, float]:
        cardinality_key = f"{primitive.value}:{cardinality}"
        if cardinality_key in self.by_cardinality:
            return cardinality_key, self.by_cardinality[cardinality_key]
        if primitive.value in self.by_primitive:
            return primitive.value, self.by_primitive[primitive.value]
        return "global", self.global_temperature

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_version": self.artifact_version,
            "method": self.method,
            "global_temperature": self.global_temperature,
            "by_primitive": dict(self.by_primitive),
            "by_cardinality": dict(self.by_cardinality),
        }

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "TemperatureArtifact":
        value = json.loads(Path(path).read_text())
        if not isinstance(value, dict):
            raise ValueError("calibration artifact must contain a JSON object")
        return cls(**value)


def fit_temperatures(
    observations: Sequence[CalibrationObservation],
    *,
    strategy: str = "auto",
    min_examples: int = 20,
    folds: int = 5,
) -> TemperatureArtifact:
    """Fit global and optional hierarchical temperatures by held-out NLL."""
    if not observations:
        raise ValueError("temperature fitting requires calibration observations")
    if strategy not in {"auto", "global", "primitive", "cardinality"}:
        raise ValueError("strategy must be auto, global, primitive, or cardinality")
    if min_examples <= 0:
        raise ValueError("min_examples must be positive")
    if folds < 2:
        raise ValueError("folds must be at least 2")
    if strategy == "auto":
        strategy = _select_temperature_strategy(
            observations,
            min_examples=min_examples,
            folds=folds,
        )

    global_temperature = _fit_temperature(observations)
    by_primitive: dict[str, float] = {}
    by_cardinality: dict[str, float] = {}
    if strategy in {"primitive", "cardinality"}:
        for primitive in Primitive:
            selected = [item for item in observations if item.primitive is primitive]
            if len(selected) >= min_examples:
                by_primitive[primitive.value] = _fit_temperature(selected)
    if strategy == "cardinality":
        keys = sorted({(item.primitive, len(item.logits)) for item in observations})
        for primitive, cardinality in keys:
            selected = [
                item
                for item in observations
                if item.primitive is primitive and len(item.logits) == cardinality
            ]
            if len(selected) >= min_examples:
                by_cardinality[f"{primitive.value}:{cardinality}"] = _fit_temperature(selected)
    return TemperatureArtifact(
        global_temperature,
        by_primitive,
        by_cardinality,
        method="temperature_scaling",
    )


def softmax(logits: np.ndarray) -> np.ndarray:
    exponentials = np.exp(logits - logits.max())
    return exponentials / exponentials.sum()


def _select_temperature_strategy(
    observations: Sequence[CalibrationObservation],
    *,
    min_examples: int,
    folds: int,
) -> str:
    if len(observations) < folds:
        return "global"
    strategies = ("global", "primitive", "cardinality")
    losses: dict[str, float] = {}
    for strategy in strategies:
        fold_losses = []
        for fold in range(folds):
            training = [item for index, item in enumerate(observations) if index % folds != fold]
            validation = [item for index, item in enumerate(observations) if index % folds == fold]
            artifact = fit_temperatures(
                training,
                strategy=strategy,
                min_examples=min_examples,
                folds=folds,
            )
            fold_losses.append(_calibrated_nll(validation, artifact))
        losses[strategy] = float(np.mean(fold_losses))
    return min(strategies, key=lambda name: (losses[name], strategies.index(name)))


def _calibrated_nll(
    observations: Sequence[CalibrationObservation],
    artifact: TemperatureArtifact,
) -> float:
    return float(
        np.mean(
            [
                -math.log(
                    max(
                        softmax(
                            np.asarray(item.logits)
                            / artifact.select(item.primitive, len(item.logits))[1]
                        )[item.target],
                        1e-12,
                    )
                )
                for item in observations
            ]
        )
    )


def _fit_temperature(observations: Sequence[CalibrationObservation]) -> float:
    candidates = np.logspace(-2, 2, 401)
    losses = [
        np.mean(
            [
                -math.log(max(softmax(np.asarray(item.logits) / temperature)[item.target], 1e-12))
                for item in observations
            ]
        )
        for temperature in candidates
    ]
    return float(candidates[int(np.argmin(losses))])
