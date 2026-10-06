import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
mx = pytest.importorskip("mlx.core")

from mlx.utils import tree_flatten  # noqa: E402

from nirnaya.hf.configuration_nirnaya import NirnayaFlashConfig  # noqa: E402
from nirnaya.hf.modeling_nirnaya import NirnayaFlashModel  # noqa: E402
from nirnaya.models.flash.model import FlashDecisionModel  # noqa: E402
from nirnaya.models.flash.modernbert import ModernBert, ModernBertConfig  # noqa: E402

ENCODER = {
    "model_type": "modernbert",
    "vocab_size": 64,
    "hidden_size": 64,
    "intermediate_size": 96,
    "num_hidden_layers": 3,
    "num_attention_heads": 4,
    "local_attention": 8,
    "global_attn_every_n_layers": 3,
    "global_rope_theta": 160000.0,
    "local_rope_theta": 10000.0,
    "max_position_embeddings": 128,
    "pad_token_id": 0,
    "hidden_activation": "gelu",
    "norm_bias": False,
    "attention_bias": False,
    "mlp_bias": False,
}


def test_torch_model_reproduces_mlx_logits_from_the_same_tensors() -> None:
    mx.random.seed(7)
    mlx_model = FlashDecisionModel(
        ModernBert(ModernBertConfig.from_dict(ENCODER)),
        decision_layers=2,
        architecture_version="shared_marker_v2",
    )
    mlx_model.eval()
    weights = {name: np.array(value) for name, value in tree_flatten(mlx_model.parameters())}
    torch_model = NirnayaFlashModel(NirnayaFlashConfig(encoder=ENCODER, decision_layers=2))
    torch_model.load_state_dict({name: torch.from_numpy(value) for name, value in weights.items()})
    torch_model.eval()

    rng = np.random.default_rng(3)
    lengths = (20, 14, 9)
    ids = np.zeros((3, 20), dtype=np.int64)
    for row, length in enumerate(lengths):
        ids[row, :length] = rng.integers(1, 64, size=length)
    markers = np.array([[2, 5, 9], [3, 6, 0], [2, 4, 0]], dtype=np.int64)
    marker_mask = np.array([[True, True, True], [True, True, False], [True, True, False]])
    primitives = np.array([0, 1, 2], dtype=np.int64)

    expected = _mlx_cpu_logits(mlx_model, ids, markers, marker_mask, primitives)
    with torch.inference_mode():
        actual = torch_model(
            input_ids=torch.from_numpy(ids),
            attention_mask=torch.from_numpy(ids != 0),
            marker_positions=torch.from_numpy(markers),
            marker_mask=torch.from_numpy(marker_mask),
            primitive_indices=torch.from_numpy(primitives),
        ).numpy()

    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)


def _mlx_cpu_logits(model, ids, markers, marker_mask, primitives) -> np.ndarray:
    # Apple-GPU float32 matmuls run at reduced precision, so exact parity is checked on CPU.
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        logits, _ = model.forward_arrays(
            mx.array(ids),
            mx.array(ids != 0),
            mx.array(markers),
            mx.array(marker_mask),
            mx.array(primitives),
        )
        return np.array(logits)
    finally:
        mx.set_default_device(previous)
