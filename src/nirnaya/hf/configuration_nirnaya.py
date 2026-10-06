"""Transformers configuration for Nirnaya-Flash typed decision models."""

from transformers import ModernBertConfig, PretrainedConfig

SUPPORTED_ARCHITECTURE = "shared_marker_v2"


class NirnayaFlashConfig(PretrainedConfig):
    model_type = "nirnaya_flash"

    def __init__(
        self,
        encoder: dict | None = None,
        decision_layers: int = 2,
        architecture_version: str = SUPPORTED_ARCHITECTURE,
        max_length: int = 512,
        head_max_length: int = 192,
        option_max_length: int = 48,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.encoder = dict(encoder or {})
        self.decision_layers = decision_layers
        self.architecture_version = architecture_version
        self.max_length = max_length
        self.head_max_length = head_max_length
        self.option_max_length = option_max_length

    def encoder_config(self) -> ModernBertConfig:
        return ModernBertConfig(**self.encoder)
