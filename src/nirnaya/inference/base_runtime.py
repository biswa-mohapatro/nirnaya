"""Backend-neutral request handling shared by the MLX and PyTorch release runtimes."""

from collections.abc import Mapping, Sequence

import numpy as np

from nirnaya.data.schemas import CanonicalSample, serialize_state
from nirnaya.inference.calibration import TemperatureArtifact, softmax
from nirnaya.inference.decisions import build_result, inference_sample
from nirnaya.inference.schemas import Question, parse_questions
from nirnaya.models.flash.encoding import FlashInputEncoder, FormattingDiagnostics
from nirnaya.types import Primitive

PRIMITIVE_INDEX = {Primitive.CHOICE: 0, Primitive.NOUL: 1, Primitive.SCORE: 2}


class ReleaseRuntime:
    """Parse typed requests, score them with a backend, and return calibrated results."""

    def __init__(
        self,
        encoder: FlashInputEncoder,
        *,
        calibration: TemperatureArtifact | None = None,
        thresholds: Mapping[Primitive, float] | None = None,
        max_batch_rows: int = 16,
    ):
        self.encoder = encoder
        self.calibration = calibration or TemperatureArtifact()
        self.thresholds = {primitive: 0.0 for primitive in Primitive}
        self.thresholds.update(thresholds or {})
        if any(not 0 <= value <= 1 for value in self.thresholds.values()):
            raise ValueError("abstention thresholds must be between 0 and 1")
        if max_batch_rows <= 0:
            raise ValueError("max_batch_rows must be positive")
        self.max_batch_rows = max_batch_rows

    def predict(
        self,
        state: str | Mapping[str, object] | Sequence[object],
        questions: Mapping[str, Mapping[str, object]],
    ) -> dict[str, dict[str, object]]:
        """Answer every keyed question about one state."""
        return self.predict_batch([{"state": state, "questions": questions}])[0]

    def input_diagnostics(
        self,
        state: str | Mapping[str, object] | Sequence[object],
        questions: Mapping[str, Mapping[str, object]],
    ) -> dict[str, FormattingDiagnostics]:
        """Report per-question token retention without running the model."""
        state_text = serialize_state(state)
        return {
            question_id: self.encoder.format_sample_with_diagnostics(
                inference_sample(0, question_id, state_text, question)
            )[1]
            for question_id, question in parse_questions(questions).items()
        }

    def predict_batch(
        self,
        requests: Sequence[Mapping[str, object]],
    ) -> list[dict[str, dict[str, object]]]:
        """Answer several `{state, questions}` requests in shared model batches."""
        samples: list[CanonicalSample] = []
        entries: list[tuple[int, str, Question]] = []
        for request_index, request in enumerate(requests):
            raw_questions = request.get("questions")
            if not isinstance(raw_questions, Mapping):
                raise TypeError("each request requires state and mapping questions")
            state_text = serialize_state(request.get("state"))  # type: ignore[arg-type]
            questions = parse_questions(raw_questions)
            if not questions:
                raise ValueError("each request must contain at least one question")
            for question_id, question in questions.items():
                samples.append(inference_sample(request_index, question_id, state_text, question))
                entries.append((request_index, question_id, question))

        results: list[dict[str, dict[str, object]]] = [{} for _ in requests]
        for (request_index, question_id, question), logits in zip(
            entries, self.logits(samples), strict=True
        ):
            key, temperature = self.calibration.select(question.type, len(logits))
            results[request_index][question_id] = build_result(
                question,
                softmax(logits / temperature),
                calibration=self.calibration,
                calibration_key=key,
                temperature=temperature,
                thresholds=self.thresholds,
            ).to_dict()
        return results

    def logits(self, samples: Sequence[CanonicalSample]) -> list[np.ndarray]:
        """Return raw logits over each sample's valid options, in input order."""
        outputs: list[np.ndarray] = []
        pad = self.encoder.pad_id
        for start in range(0, len(samples), self.max_batch_rows):
            chunk = samples[start : start + self.max_batch_rows]
            encoded = [self.encoder.format_sample(sample) for sample in chunk]
            width = max(len(ids) for ids, _, _ in encoded)
            slots = max(2, max(len(markers) for _, markers, _ in encoded))
            logits = self._forward(
                input_ids=np.array(
                    [ids + [pad] * (width - len(ids)) for ids, _, _ in encoded], dtype=np.int64
                ),
                marker_positions=np.array(
                    [markers + [0] * (slots - len(markers)) for _, markers, _ in encoded],
                    dtype=np.int64,
                ),
                marker_mask=np.array(
                    [
                        [True] * len(markers) + [False] * (slots - len(markers))
                        for _, markers, _ in encoded
                    ]
                ),
                primitive_indices=np.array(
                    [PRIMITIVE_INDEX[sample.primitive] for sample in chunk], dtype=np.int64
                ),
            )
            outputs.extend(
                logits[row, : len(markers)].astype(np.float64)
                for row, (_, markers, _) in enumerate(encoded)
            )
        return outputs

    def _forward(
        self,
        *,
        input_ids: np.ndarray,
        marker_positions: np.ndarray,
        marker_mask: np.ndarray,
        primitive_indices: np.ndarray,
    ) -> np.ndarray:
        raise NotImplementedError
