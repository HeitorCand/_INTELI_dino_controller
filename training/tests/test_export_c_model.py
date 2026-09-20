import re

import numpy as np

from training.export_c_model import format_c_array, generate_header
from training.train import CommandClassifier, export_onnx


def test_format_c_array_produces_valid_c_float_literals():
    values = np.array([1.5, -2.25, 0.0], dtype=np.float32)
    text = format_c_array("MY_ARRAY", values)

    assert text.startswith("static const float MY_ARRAY[3] = {")
    literals = re.findall(r"-?\d+\.\d+f", text)
    assert len(literals) == 3
    assert float(literals[0][:-1]) == 1.5
    assert float(literals[1][:-1]) == -2.25


def test_generate_header_has_correct_dimensions_and_arrays(tmp_path):
    model = CommandClassifier(input_dim=14, hidden_dim=16, num_classes=3)
    export_onnx(model, tmp_path / "model.onnx")
    np.save(tmp_path / "feature_mean.npy", np.zeros(14, dtype=np.float32))
    np.save(tmp_path / "feature_std.npy", np.ones(14, dtype=np.float32))

    header = generate_header(tmp_path)

    assert "#define MODEL_INPUT_DIM 14" in header
    assert "#define MODEL_HIDDEN_DIM 16" in header
    assert "#define MODEL_OUTPUT_DIM 3" in header
    assert "MODEL_W1[224]" in header  # 16 * 14
    assert "MODEL_B1[16]" in header
    assert "MODEL_W2[48]" in header  # 3 * 16
    assert "MODEL_B2[3]" in header
    assert "MODEL_FEATURE_MEAN[14]" in header
    assert "MODEL_FEATURE_STD[14]" in header


def test_generate_header_weights_match_onnx_initializer_values(tmp_path):
    model = CommandClassifier(input_dim=14, hidden_dim=16, num_classes=3)
    export_onnx(model, tmp_path / "model.onnx")
    np.save(tmp_path / "feature_mean.npy", np.zeros(14, dtype=np.float32))
    np.save(tmp_path / "feature_std.npy", np.ones(14, dtype=np.float32))

    header = generate_header(tmp_path)

    w1 = model.net[0].weight.detach().numpy().astype(np.float32).flatten()

    match = re.search(r"MODEL_W1\[224\] = \{([^}]+)\}", header)
    assert match is not None
    header_values = [float(v.strip().rstrip("f")) for v in match.group(1).split(",")]

    np.testing.assert_allclose(header_values, w1, atol=1e-4)
