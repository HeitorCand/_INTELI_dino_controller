from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn

from training.dataset import CLASSES, build_dataset, split_dataset
from training.features import FEATURE_DIM

HIDDEN_DIM = 16
NUM_CLASSES = len(CLASSES)


class CommandClassifier(nn.Module):
    def __init__(
        self, input_dim: int = FEATURE_DIM, hidden_dim: int = HIDDEN_DIM, num_classes: int = NUM_CLASSES
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def normalize_features(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((X - mean) / std).astype(np.float32)


def train_model(
    X_train: np.ndarray, y_train: np.ndarray, epochs: int = 200, lr: float = 0.01
) -> CommandClassifier:
    model = CommandClassifier()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    X_tensor = torch.from_numpy(X_train.astype(np.float32))
    y_tensor = torch.from_numpy(y_train.astype(np.int64))

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(model(X_tensor), y_tensor)
        loss.backward()
        optimizer.step()

    return model


def evaluate_model(model: CommandClassifier, X_test: np.ndarray, y_test: np.ndarray) -> str:
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X_test.astype(np.float32)))
        predictions = torch.argmax(logits, dim=1).numpy()

    report = classification_report(y_test, predictions, target_names=CLASSES, zero_division=0)
    matrix = confusion_matrix(y_test, predictions)
    return f"{report}\n\nConfusion matrix (rows=true, cols=pred), order {CLASSES}:\n{matrix}"


def export_onnx(model: CommandClassifier, output_path: Path) -> None:
    model.eval()
    dummy_input = torch.zeros(1, FEATURE_DIM, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=["features"],
        output_names=["logits"],
        opset_version=13,
        dynamo=False,
    )


def main():
    raw_dir = Path(__file__).parent / "data" / "raw"
    X, y = build_dataset(raw_dir, augment=True)
    X_train, X_test, y_train, y_test = split_dataset(X, y)

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    X_train_norm = normalize_features(X_train, mean, std)
    X_test_norm = normalize_features(X_test, mean, std)

    model = train_model(X_train_norm, y_train)
    report = evaluate_model(model, X_test_norm, y_test)
    print(report)

    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    export_onnx(model, models_dir / "model.onnx")
    (models_dir / "evaluation.txt").write_text(report)
    np.save(models_dir / "feature_mean.npy", mean)
    np.save(models_dir / "feature_std.npy", std)


if __name__ == "__main__":
    main()
