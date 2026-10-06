"""Primitive-specific raw-logit heads."""

import mlx.core as mx
import mlx.nn as nn


class MarkerHead(nn.Module):
    """Score each dynamic option-marker representation independently."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

    def __call__(self, marker_states: mx.array) -> mx.array:
        return self.layers(marker_states).squeeze(-1).astype(mx.float32)


class ChoiceHead(MarkerHead):
    """Dynamic candidate scorer for Choice decisions."""


class ScoreHead(MarkerHead):
    """Ordered outcome scorer for Score decisions."""


class NoulHead(nn.Module):
    """Binary log-odds scorer exposed as false/true logits."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

    def __call__(self, decision_states: mx.array) -> mx.array:
        log_odds = self.layers(decision_states).astype(mx.float32)
        return mx.concatenate([mx.zeros_like(log_odds), log_odds], axis=-1)
