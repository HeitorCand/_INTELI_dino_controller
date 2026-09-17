import numpy as np
import onnxruntime as ort
import torch
from sklearn.metrics import classification_report

from training.dataset import CLASSES
from training.features import FEATURE_DIM
from training.train import (
    evaluate_model,
    export_onnx,
    normalize_features,
    train_model,
)


def _make_synthetic_dataset(samples_per_class=40, seed=0):
    rng = np.random.default_rng(seed)
    centers = {0: np.zeros(FEATURE_DIM), 1: np.full(FEATURE_DIM, 5.0), 2: np.full(FEATURE_DIM, -5.0)}
    X, y = [], []
    for label, center in centers.items():
        X.append(rng.normal(center, 0.5, size=(samples_per_class, FEATURE_DIM)))
        y.append(np.full(samples_per_class, label))
    return np.vstack(X).astype(np.float32), np.concatenate(y).astype(np.int64)


def test_train_model_learns_separable_classes():
    X, y = _make_synthetic_dataset()
    mean, std = X.mean(axis=0), X.std(axis=0) + 1e-8
    X_norm = normalize_features(X, mean, std)

    model = train_model(X_norm, y, epochs=200)

    model.eval()
    with torch.no_grad():
        predictions = torch.argmax(model(torch.from_numpy(X_norm)), dim=1).numpy()
    accuracy = (predictions == y).mean()
    assert accuracy > 0.9


def test_export_onnx_matches_torch_output(tmp_path):
    X, y = _make_synthetic_dataset()
    mean, std = X.mean(axis=0), X.std(axis=0) + 1e-8
    X_norm = normalize_features(X, mean, std)
    model = train_model(X_norm, y, epochs=200)

    onnx_path = tmp_path / "model.onnx"
    export_onnx(model, onnx_path)

    sample = X_norm[:1]
    model.eval()
    with torch.no_grad():
        torch_output = model(torch.from_numpy(sample)).numpy()

    session = ort.InferenceSession(str(onnx_path))
    onnx_output = session.run(None, {"features": sample.astype(np.float32)})[0]

    assert onnx_output.shape == (1, 3)
    np.testing.assert_allclose(torch_output, onnx_output, rtol=1e-4, atol=1e-4)


def test_evaluate_model_report_mentions_all_classes():
    X, y = _make_synthetic_dataset()
    mean, std = X.mean(axis=0), X.std(axis=0) + 1e-8
    X_norm = normalize_features(X, mean, std)
    model = train_model(X_norm, y, epochs=200)

    report = evaluate_model(model, X_norm, y)

    assert "pular" in report
    assert "abaixa" in report
    assert "ruido" in report


def test_evaluate_model_report_uses_class_names_not_numeric_labels():
    X, y = _make_synthetic_dataset()
    mean, std = X.mean(axis=0), X.std(axis=0) + 1e-8
    X_norm = normalize_features(X, mean, std)
    model = train_model(X_norm, y, epochs=200)

    model.eval()
    with torch.no_grad():
        predictions = torch.argmax(model(torch.from_numpy(X_norm)), dim=1).numpy()

    expected_report = classification_report(y, predictions, target_names=CLASSES, zero_division=0)
    report = evaluate_model(model, X_norm, y)

    assert expected_report in report
