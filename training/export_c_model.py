from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


def load_initializer(model: onnx.ModelProto, name: str) -> np.ndarray:
    for init in model.graph.initializer:
        if init.name == name:
            return numpy_helper.to_array(init)
    raise KeyError(f"initializer not found: {name}")


def format_c_array(name: str, values: np.ndarray) -> str:
    flat = values.astype(np.float32).flatten()
    body = ", ".join(f"{v:.8f}f" for v in flat)
    return f"static const float {name}[{flat.size}] = {{{body}}};"


def generate_header(models_dir: Path) -> str:
    model = onnx.load(str(models_dir / "model.onnx"))
    w1 = load_initializer(model, "net.0.weight")  # [16, 14]
    b1 = load_initializer(model, "net.0.bias")  # [16]
    w2 = load_initializer(model, "net.2.weight")  # [3, 16]
    b2 = load_initializer(model, "net.2.bias")  # [3]
    mean = np.load(models_dir / "feature_mean.npy")  # [14]
    std = np.load(models_dir / "feature_std.npy")  # [14]

    input_dim, hidden_dim, output_dim = w1.shape[1], w1.shape[0], w2.shape[0]

    lines = [
        "// Gerado automaticamente por training/export_c_model.py — NAO EDITAR A MAO.",
        "// Pesos do classificador (14 -> 16 -> 3) exportados de models/model.onnx.",
        "#pragma once",
        "",
        f"#define MODEL_INPUT_DIM {input_dim}",
        f"#define MODEL_HIDDEN_DIM {hidden_dim}",
        f"#define MODEL_OUTPUT_DIM {output_dim}",
        "",
        format_c_array("MODEL_FEATURE_MEAN", mean),
        format_c_array("MODEL_FEATURE_STD", std),
        format_c_array("MODEL_W1", w1),  # row-major [hidden_dim][input_dim]
        format_c_array("MODEL_B1", b1),
        format_c_array("MODEL_W2", w2),  # row-major [output_dim][hidden_dim]
        format_c_array("MODEL_B2", b2),
        "",
    ]
    return "\n".join(lines)


def main():
    repo_root = Path(__file__).parent.parent
    models_dir = repo_root / "models"
    header = generate_header(models_dir)

    out_path = repo_root / "firmware" / "dino_voice_controller" / "model_weights.h"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(header)
    print(f"Escrito: {out_path}")


if __name__ == "__main__":
    main()
