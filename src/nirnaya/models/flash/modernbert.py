"""Trainable ModernBERT encoder implemented natively in MLX."""

from dataclasses import dataclass, fields
from typing import Any

import mlx.core as mx
import mlx.nn as nn

FULL_ATTENTION = "full_attention"
SLIDING_ATTENTION = "sliding_attention"
_ATTENTION_TYPES = frozenset({FULL_ATTENTION, SLIDING_ATTENTION})


@dataclass(frozen=True, slots=True)
class ModernBertConfig:
    """ModernBERT architecture fields required by the native MLX encoder."""

    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    model_type: str = "modernbert"
    norm_eps: float = 1e-5
    norm_bias: bool = False
    attention_bias: bool = False
    mlp_bias: bool = False
    hidden_activation: str = "gelu"
    local_attention: int = 128
    global_attn_every_n_layers: int = 3
    global_rope_theta: float = 160_000.0
    local_rope_theta: float = 10_000.0
    max_position_embeddings: int = 8192
    layer_types: tuple[str, ...] | None = None
    rope_parameters: dict[str, dict[str, Any]] | None = None

    def __post_init__(self) -> None:
        if self.model_type != "modernbert":
            raise ValueError(f"unsupported model_type: {self.model_type!r}")
        if self.hidden_activation != "gelu":
            raise ValueError(f"unsupported hidden activation: {self.hidden_activation!r}")
        if self.hidden_size <= 0 or self.intermediate_size <= 0 or self.vocab_size <= 0:
            raise ValueError("model dimensions must be positive")
        if self.num_hidden_layers <= 0 or self.num_attention_heads <= 0:
            raise ValueError("layer and attention-head counts must be positive")
        if self.hidden_size % self.num_attention_heads != 0 or self.head_dim % 2 != 0:
            raise ValueError("attention head dimensions must be even and integral")
        if self.local_attention <= 0 or self.local_attention % 2 != 0:
            raise ValueError("local_attention must be a positive even number")
        if self.global_attn_every_n_layers <= 0:
            raise ValueError("global_attn_every_n_layers must be positive")
        if self.max_position_embeddings <= 0:
            raise ValueError("max_position_embeddings must be positive")

        layer_types = self.layer_types
        if layer_types is None:
            layer_types = tuple(
                FULL_ATTENTION
                if index % self.global_attn_every_n_layers == 0
                else SLIDING_ATTENTION
                for index in range(self.num_hidden_layers)
            )
            object.__setattr__(self, "layer_types", layer_types)
        if len(layer_types) != self.num_hidden_layers:
            raise ValueError("layer_types must match num_hidden_layers")
        if set(layer_types) - _ATTENTION_TYPES:
            raise ValueError("layer_types contains an unsupported attention type")

        for attention_type in set(layer_types):
            rope = (self.rope_parameters or {}).get(attention_type, {})
            if rope.get("rope_type", "default") != "default":
                raise ValueError("only unscaled default RoPE is supported")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    def rope_base(self, attention_type: str) -> float:
        fallback = (
            self.global_rope_theta if attention_type == FULL_ATTENTION else self.local_rope_theta
        )
        return float(
            (self.rope_parameters or {}).get(attention_type, {}).get("rope_theta", fallback)
        )

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ModernBertConfig":
        """Parse a Hugging Face ModernBERT config without accepting unknown behavior."""
        names = {item.name for item in fields(cls)}
        parsed = {name: item for name, item in value.items() if name in names}
        if "norm_eps" not in parsed and "layer_norm_eps" in value:
            parsed["norm_eps"] = value["layer_norm_eps"]
        if "layer_types" in parsed and parsed["layer_types"] is not None:
            parsed["layer_types"] = tuple(parsed["layer_types"])
        return cls(**parsed)


class ModernBertEmbeddings(nn.Module):
    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.norm = nn.LayerNorm(
            config.hidden_size,
            eps=config.norm_eps,
            bias=config.norm_bias,
        )

    def __call__(self, input_ids: mx.array) -> mx.array:
        return self.norm(self.tok_embeddings(input_ids))


class ModernBertAttention(nn.Module):
    def __init__(self, config: ModernBertConfig, attention_type: str):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.rope_base = config.rope_base(attention_type)
        self.Wqkv = nn.Linear(
            config.hidden_size,
            3 * config.hidden_size,
            bias=config.attention_bias,
        )
        self.Wo = nn.Linear(
            config.hidden_size,
            config.hidden_size,
            bias=config.attention_bias,
        )

    def __call__(self, hidden_states: mx.array, attention_mask: mx.array) -> mx.array:
        batch_size, sequence_length, _ = hidden_states.shape
        qkv = self.Wqkv(hidden_states).reshape(
            batch_size,
            sequence_length,
            3,
            self.num_heads,
            self.head_dim,
        )
        query, key, value = [qkv[:, :, index].transpose(0, 2, 1, 3) for index in range(3)]
        query = mx.fast.rope(
            query,
            self.head_dim,
            traditional=False,
            base=self.rope_base,
            scale=1.0,
            offset=0,
        )
        key = mx.fast.rope(
            key,
            self.head_dim,
            traditional=False,
            base=self.rope_base,
            scale=1.0,
            offset=0,
        )
        attended = mx.fast.scaled_dot_product_attention(
            query,
            key,
            value,
            scale=self.head_dim**-0.5,
            mask=attention_mask,
        )
        return self.Wo(attended.transpose(0, 2, 1, 3).reshape(batch_size, sequence_length, -1))


class ModernBertMlp(nn.Module):
    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.Wi = nn.Linear(
            config.hidden_size,
            2 * config.intermediate_size,
            bias=config.mlp_bias,
        )
        self.Wo = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            bias=config.mlp_bias,
        )

    def __call__(self, hidden_states: mx.array) -> mx.array:
        value, gate = mx.split(self.Wi(hidden_states), 2, axis=-1)
        return self.Wo(nn.gelu(value) * gate)


class ModernBertLayer(nn.Module):
    def __init__(self, config: ModernBertConfig, index: int):
        super().__init__()
        self.attention_type = config.layer_types[index]
        self.attn_norm = (
            nn.Identity()
            if index == 0
            else nn.LayerNorm(
                config.hidden_size,
                eps=config.norm_eps,
                bias=config.norm_bias,
            )
        )
        self.attn = ModernBertAttention(config, self.attention_type)
        self.mlp_norm = nn.LayerNorm(
            config.hidden_size,
            eps=config.norm_eps,
            bias=config.norm_bias,
        )
        self.mlp = ModernBertMlp(config)

    def __call__(
        self,
        hidden_states: mx.array,
        attention_masks: dict[str, mx.array],
    ) -> mx.array:
        hidden_states = hidden_states + self.attn(
            self.attn_norm(hidden_states),
            attention_masks[self.attention_type],
        )
        return hidden_states + self.mlp(self.mlp_norm(hidden_states))


def build_attention_masks(attention_mask: mx.array, local_window: int) -> dict[str, mx.array]:
    """Build full and inclusive sliding-window masks over valid key tokens."""
    if attention_mask.ndim != 2:
        raise ValueError("attention_mask must have shape [batch, sequence]")
    valid = attention_mask.astype(mx.bool_)
    full = valid[:, None, None, :]
    positions = mx.arange(valid.shape[1])
    local = mx.abs(positions[:, None] - positions[None, :]) <= local_window // 2
    local = (local[None, None] | ~valid[:, None, :, None]) & full
    return {FULL_ATTENTION: full, SLIDING_ATTENTION: local}


class ModernBert(nn.Module):
    """Bidirectional ModernBERT encoder returning contextual hidden states."""

    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.config = config
        self.embeddings = ModernBertEmbeddings(config)
        self.layers = [ModernBertLayer(config, index) for index in range(config.num_hidden_layers)]
        self.final_norm = nn.LayerNorm(
            config.hidden_size,
            eps=config.norm_eps,
            bias=config.norm_bias,
        )

    def __call__(self, input_ids: mx.array, attention_mask: mx.array) -> mx.array:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.shape != attention_mask.shape:
            raise ValueError("attention_mask must match input_ids")
        if input_ids.shape[1] > self.config.max_position_embeddings:
            raise ValueError("input sequence exceeds max_position_embeddings")

        hidden_states = self.embeddings(input_ids)
        masks = build_attention_masks(attention_mask, self.config.local_attention)
        for layer in self.layers:
            hidden_states = layer(hidden_states, masks)
        return self.final_norm(hidden_states)
