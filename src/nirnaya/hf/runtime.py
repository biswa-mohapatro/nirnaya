"""Calibrated typed decisions on CPU, CUDA, or Apple MPS through PyTorch."""

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

from nirnaya.hf.modeling_nirnaya import NirnayaFlashModel
from nirnaya.inference.base_runtime import ReleaseRuntime
from nirnaya.inference.calibration import TemperatureArtifact
from nirnaya.models.flash.encoding import FlashInputEncoder
from nirnaya.types import Primitive


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class TorchFlashRuntime(ReleaseRuntime):
    """Same request format and result envelope as the MLX runtime."""

    def __init__(self, model: NirnayaFlashModel, encoder: FlashInputEncoder, **kwargs):
        super().__init__(encoder, **kwargs)
        self.model = model.eval()

    @classmethod
    def from_pretrained(
        cls,
        path: str | Path,
        *,
        device: str | None = None,
        dtype: torch.dtype | None = None,
        thresholds: Mapping[Primitive, float] | None = None,
        max_batch_rows: int = 16,
    ) -> "TorchFlashRuntime":
        """Load a release directory or Hugging Face repository id."""
        directory = Path(path).expanduser()
        if not directory.is_dir():
            from huggingface_hub import snapshot_download

            directory = Path(snapshot_download(str(path)))
        target = torch.device(device or default_device())
        model = NirnayaFlashModel.from_pretrained(
            directory, dtype=dtype or (torch.float32 if target.type == "cpu" else torch.float16)
        ).to(target)
        config = model.config
        encoder = FlashInputEncoder(
            Tokenizer.from_file(str(directory / "tokenizer.json")),
            max_length=config.max_length,
            head_max_length=config.head_max_length,
            option_max_length=config.option_max_length,
        )
        calibration_path = directory / "calibration.json"
        calibration = (
            TemperatureArtifact.load(calibration_path) if calibration_path.is_file() else None
        )
        return cls(
            model,
            encoder,
            calibration=calibration,
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
        device = next(self.model.parameters()).device
        ids = torch.from_numpy(input_ids).to(device)
        with torch.inference_mode():
            return (
                self.model(
                    input_ids=ids,
                    attention_mask=ids != self.encoder.pad_id,
                    marker_positions=torch.from_numpy(marker_positions).to(device),
                    marker_mask=torch.from_numpy(marker_mask).to(device),
                    primitive_indices=torch.from_numpy(primitive_indices).to(device),
                )
                .cpu()
                .numpy()
            )
