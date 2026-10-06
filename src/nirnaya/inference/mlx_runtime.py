"""Apple Silicon (MLX) runtime for exported release directories."""

import json
from collections.abc import Mapping
from pathlib import Path

import mlx.core as mx
import numpy as np
from tokenizers import Tokenizer

from nirnaya.inference.base_runtime import ReleaseRuntime
from nirnaya.inference.calibration import TemperatureArtifact
from nirnaya.models.flash.encoding import FlashInputEncoder
from nirnaya.models.flash.model import FlashDecisionModel
from nirnaya.models.flash.modernbert import ModernBert, ModernBertConfig
from nirnaya.types import Primitive

_DTYPES = {"float16": mx.float16, "bfloat16": mx.bfloat16, "float32": mx.float32}


class MLXFlashRuntime(ReleaseRuntime):
    """Same request format and result envelope as the PyTorch runtime."""

    def __init__(self, model: FlashDecisionModel, encoder: FlashInputEncoder, **kwargs):
        super().__init__(encoder, **kwargs)
        self.model = model

    @classmethod
    def from_pretrained(
        cls,
        path: str | Path,
        *,
        thresholds: Mapping[Primitive, float] | None = None,
        max_batch_rows: int = 16,
    ) -> "MLXFlashRuntime":
        """Load a release directory or Hugging Face repository id."""
        directory = Path(path).expanduser()
        if not directory.is_dir():
            from huggingface_hub import snapshot_download

            directory = Path(snapshot_download(str(path)))
        config = json.loads((directory / "config.json").read_text())
        model = FlashDecisionModel(
            ModernBert(ModernBertConfig.from_dict(config["encoder"])),
            decision_layers=int(config["decision_layers"]),
            architecture_version=str(config["architecture_version"]),
        )
        model.load_weights(str(directory / "model.safetensors"), strict=True)
        model.set_dtype(_DTYPES[str(config.get("dtype", "float16"))])
        mx.eval(model.parameters())
        model.eval()
        encoder = FlashInputEncoder(
            Tokenizer.from_file(str(directory / "tokenizer.json")),
            max_length=int(config["max_length"]),
            head_max_length=int(config["head_max_length"]),
            option_max_length=int(config.get("option_max_length", 48)),
        )
        calibration_path = directory / "calibration.json"
        return cls(
            model,
            encoder,
            calibration=TemperatureArtifact.load(calibration_path)
            if calibration_path.is_file()
            else None,
            thresholds=thresholds,
            max_batch_rows=max_batch_rows,
        )

    def _forward(
        self,
        *,
        input_ids: np.ndarray,
        marker_positions: np.ndarray,
        marker_mask: np.ndarray,
        primitive_indices: np.ndarray,
    ) -> np.ndarray:
        logits, _ = self.model.forward_arrays(
            mx.array(input_ids, dtype=mx.int32),
            mx.array(input_ids != self.encoder.pad_id),
            mx.array(marker_positions, dtype=mx.int32),
            mx.array(marker_mask),
            mx.array(primitive_indices, dtype=mx.int32),
        )
        mx.eval(logits)
        return np.asarray(logits.astype(mx.float32))
