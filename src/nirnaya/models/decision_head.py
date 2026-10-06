"""Sequence-level decision transformer shared by Nirnaya encoders."""

import mlx.core as mx
import mlx.nn as nn


class DecisionAttention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.num_heads = max(1, hidden_size // 64)
        if hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by the decision-head count")
        self.head_dim = hidden_size // self.num_heads
        self.in_proj = nn.Linear(hidden_size, 3 * hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def __call__(self, hidden_states: mx.array, attention_mask: mx.array) -> mx.array:
        batch_size, sequence_length, hidden_size = hidden_states.shape
        qkv = self.in_proj(hidden_states).reshape(
            batch_size,
            sequence_length,
            3,
            self.num_heads,
            self.head_dim,
        )
        query, key, value = [qkv[:, :, index].transpose(0, 2, 1, 3) for index in range(3)]
        attended = mx.fast.scaled_dot_product_attention(
            query,
            key,
            value,
            scale=self.head_dim**-0.5,
            mask=attention_mask,
        )
        return self.out_proj(
            attended.transpose(0, 2, 1, 3).reshape(batch_size, sequence_length, hidden_size)
        )


class DecisionLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        *,
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        super().__init__()
        if activation not in {"relu", "gelu"}:
            raise ValueError("activation must be relu or gelu")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be between zero and one")
        self.self_attn = DecisionAttention(hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.linear1 = nn.Linear(hidden_size, 4 * hidden_size)
        self.linear2 = nn.Linear(4 * hidden_size, hidden_size)
        self.activation = activation
        self.attention_dropout = nn.Dropout(dropout)
        self.activation_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)

    def __call__(self, hidden_states: mx.array, attention_mask: mx.array) -> mx.array:
        attended = self.self_attn(self.norm1(hidden_states), attention_mask)
        hidden_states = hidden_states + self.attention_dropout(attended)
        activated = self.linear1(self.norm2(hidden_states))
        activated = nn.gelu(activated) if self.activation == "gelu" else nn.relu(activated)
        feed_forward = self.linear2(self.activation_dropout(activated))
        return hidden_states + self.output_dropout(feed_forward)


class DecisionHead(nn.Module):
    """Small full-attention transformer applied after the backbone."""

    def __init__(
        self,
        hidden_size: int,
        num_layers: int = 2,
        *,
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        super().__init__()
        if num_layers < 0:
            raise ValueError("num_layers must be non-negative")
        self.layers = [
            DecisionLayer(hidden_size, activation=activation, dropout=dropout)
            for _ in range(num_layers)
        ]

    def __call__(self, hidden_states: mx.array, attention_mask: mx.array) -> mx.array:
        if attention_mask.ndim != 2 or attention_mask.shape != hidden_states.shape[:2]:
            raise ValueError("attention_mask must match the hidden-state batch and sequence")
        key_mask = attention_mask[:, None, None, :].astype(mx.bool_)
        for layer in self.layers:
            hidden_states = layer(hidden_states, key_mask)
        return hidden_states
