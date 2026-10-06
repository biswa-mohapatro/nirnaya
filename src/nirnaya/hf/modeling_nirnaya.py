"""PyTorch Nirnaya-Flash: ModernBERT, a decision transformer, and a shared option-marker scorer."""

import torch
import torch.nn.functional as functional
from torch import nn
from transformers import ModernBertModel, PreTrainedModel

from .configuration_nirnaya import SUPPORTED_ARCHITECTURE, NirnayaFlashConfig

MASKED_LOGIT = -1.0e4
PRIMITIVE_INDEX = {"choice": 0, "noul": 1, "score": 2}


class DecisionAttention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.num_heads = max(1, hidden_size // 64)
        self.head_dim = hidden_size // self.num_heads
        self.in_proj = nn.Linear(hidden_size, 3 * hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states: torch.Tensor, key_mask: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_size = hidden_states.shape
        qkv = self.in_proj(hidden_states).view(
            batch_size, sequence_length, 3, self.num_heads, self.head_dim
        )
        query, key, value = (qkv[:, :, index].transpose(1, 2) for index in range(3))
        attended = functional.scaled_dot_product_attention(
            query, key, value, attn_mask=key_mask, scale=self.head_dim**-0.5
        )
        return self.out_proj(
            attended.transpose(1, 2).reshape(batch_size, sequence_length, hidden_size)
        )


class DecisionLayer(nn.Module):
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = DecisionAttention(hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.linear1 = nn.Linear(hidden_size, 4 * hidden_size)
        self.linear2 = nn.Linear(4 * hidden_size, hidden_size)
        self.attention_dropout = nn.Dropout(dropout)
        self.activation_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: torch.Tensor, key_mask: torch.Tensor) -> torch.Tensor:
        attended = self.self_attn(self.norm1(hidden_states), key_mask)
        hidden_states = hidden_states + self.attention_dropout(attended)
        activated = functional.gelu(self.linear1(self.norm2(hidden_states)))
        return hidden_states + self.output_dropout(self.linear2(self.activation_dropout(activated)))


class DecisionHead(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList(DecisionLayer(hidden_size) for _ in range(num_layers))

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        key_mask = attention_mask[:, None, None, :].to(torch.bool)
        for layer in self.layers:
            hidden_states = layer(hidden_states, key_mask)
        return hidden_states


class _MarkerScorer(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, marker_states: torch.Tensor) -> torch.Tensor:
        return self.layers(marker_states)


class MarkerHead(nn.Module):
    # Nested so parameter names match the MLX checkpoint layout exactly.
    def __init__(self, hidden_size: int):
        super().__init__()
        self.layers = _MarkerScorer(hidden_size)

    def forward(self, marker_states: torch.Tensor) -> torch.Tensor:
        return self.layers(marker_states).squeeze(-1).float()


class NirnayaFlashModel(PreTrainedModel):
    """Score every option marker in one bidirectional forward pass; never generates text."""

    config_class = NirnayaFlashConfig
    base_model_prefix = "nirnaya_flash"
    _no_split_modules = ["ModernBertEncoderLayer", "DecisionLayer"]

    def __init__(self, config: NirnayaFlashConfig):
        super().__init__(config)
        if config.architecture_version != SUPPORTED_ARCHITECTURE:
            raise ValueError(
                f"unsupported Nirnaya-Flash architecture: {config.architecture_version}"
            )
        self.encoder = ModernBertModel(config.encoder_config())
        hidden_size = self.encoder.config.hidden_size
        self.type_embedding = nn.Embedding(len(PRIMITIVE_INDEX), hidden_size)
        self.decision_head = DecisionHead(hidden_size, config.decision_layers)
        self.marker_head = MarkerHead(hidden_size)
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        marker_positions: torch.Tensor,
        marker_mask: torch.Tensor,
        primitive_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Return float32 option logits; invalid marker slots hold a large negative value."""
        hidden_states = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        hidden_states = hidden_states + self.type_embedding(primitive_indices)[:, None, :]
        hidden_states = self.decision_head(hidden_states, attention_mask)
        rows = torch.arange(hidden_states.shape[0], device=hidden_states.device)[:, None]
        marker_states = hidden_states[rows, marker_positions.clamp(min=0)]
        logits = self.marker_head(marker_states)
        return logits.masked_fill(~marker_mask.to(torch.bool), MASKED_LOGIT)
