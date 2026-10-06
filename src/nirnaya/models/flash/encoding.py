"""Framework-neutral token layout for marker-based Nirnaya-Flash inputs."""

from dataclasses import dataclass

from tokenizers import Tokenizer

from nirnaya.data.schemas import CanonicalSample, render_outcomes


@dataclass(frozen=True, slots=True)
class FormattingDiagnostics:
    """Token retention by semantic component for one formatted sample."""

    question_tokens: int
    question_tokens_used: int
    option_tokens: int
    option_tokens_used: int
    markers: int
    markers_preserved: int
    state_tokens: int
    state_tokens_used: int
    sequence_tokens: int


class FlashInputEncoder:
    """Build `[CLS] question [SEP] [MASK] options [SEP] state [SEP]` token rows."""

    def __init__(
        self,
        tokenizer: Tokenizer,
        *,
        max_length: int = 512,
        head_max_length: int = 192,
        option_max_length: int = 48,
    ):
        if not 8 < head_max_length < max_length:
            raise ValueError("expected 8 < head_max_length < max_length")
        if option_max_length <= 0:
            raise ValueError("option_max_length must be positive")
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.head_max_length = head_max_length
        self.option_max_length = option_max_length
        self.cls_id = self._required_token("[CLS]")
        self.sep_id = self._required_token("[SEP]")
        self.mask_id = self._required_token("[MASK]")
        self.pad_id = self._required_token("[PAD]")

    def format_sample(self, sample: CanonicalSample) -> tuple[list[int], list[int], int]:
        """Format one sample while preserving every outcome marker and target."""
        formatted, _ = self.format_sample_with_diagnostics(sample)
        return formatted

    def format_sample_with_diagnostics(
        self,
        sample: CanonicalSample,
    ) -> tuple[tuple[list[int], list[int], int], FormattingDiagnostics]:
        """Format one sample and report component-level token retention."""
        options = self._options(sample)
        raw_option_tokens = [self._encode(f" {option}") for option in options]
        option_rows = [
            [self.mask_id, *tokens[: self.option_max_length]] for tokens in raw_option_tokens
        ]
        question_budget = self.head_max_length - sum(len(row) for row in option_rows)
        if question_budget < 16:
            per_option = max(4, (self.head_max_length - 16) // len(option_rows))
            option_rows = [row[:per_option] for row in option_rows]
            question_budget = self.head_max_length - sum(len(row) for row in option_rows)
        raw_question_tokens = self._encode(f"{sample.primitive.value} question: {sample.question}")
        question_tokens = raw_question_tokens[: max(8, question_budget)]

        input_ids = [self.cls_id, *question_tokens, self.sep_id]
        marker_positions: list[int] = []
        for row in option_rows:
            marker_positions.append(len(input_ids))
            input_ids.extend(row)
        input_ids.append(self.sep_id)

        state_room = max(0, self.max_length - len(input_ids) - 1)
        raw_state_tokens = self._encode(sample.state)
        state_tokens = raw_state_tokens[:state_room]
        input_ids.extend(state_tokens)
        input_ids.append(self.sep_id)
        input_ids = input_ids[: self.max_length]
        marker_positions = [position for position in marker_positions if position < self.max_length]
        if len(marker_positions) != len(options):
            raise ValueError("max_length cannot preserve all option markers")
        target = int(sample.target)
        if not 0 <= target < len(marker_positions):
            raise ValueError("target does not identify a preserved option marker")
        diagnostics = FormattingDiagnostics(
            question_tokens=len(raw_question_tokens),
            question_tokens_used=len(question_tokens),
            option_tokens=sum(len(tokens) for tokens in raw_option_tokens),
            option_tokens_used=sum(len(row) - 1 for row in option_rows),
            markers=len(options),
            markers_preserved=len(marker_positions),
            state_tokens=len(raw_state_tokens),
            state_tokens_used=len(state_tokens),
            sequence_tokens=len(input_ids),
        )
        return (input_ids, marker_positions, target), diagnostics

    def _options(self, sample: CanonicalSample) -> tuple[str, ...]:
        return render_outcomes(sample)

    def _encode(self, text: str) -> list[int]:
        cleaned = text.replace("[MASK]", " ")
        return self.tokenizer.encode(cleaned, add_special_tokens=False).ids

    def _required_token(self, token: str) -> int:
        token_id = self.tokenizer.token_to_id(token)
        if token_id is None:
            raise ValueError(f"tokenizer is missing required token {token}")
        return token_id
