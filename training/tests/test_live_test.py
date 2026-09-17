from pathlib import Path

import numpy as np
import pytest

from training.live_test import (
    CONFIDENCE_THRESHOLD,
    classify_clip,
    decide_class,
    load_classifier,
    softmax,
)

MODELS_DIR = Path(__file__).parent.parent.parent / "models"


def test_softmax_sums_to_one_and_picks_largest_logit():
    probabilities = softmax(np.array([1.0, 5.0, 2.0], dtype=np.float32))
    assert abs(probabilities.sum() - 1.0) < 1e-6
    assert int(np.argmax(probabilities)) == 1


@pytest.mark.skipif(not (MODELS_DIR / "model.onnx").exists(), reason="trained model not present")
def test_classify_clip_on_silence_returns_a_valid_class_and_distribution():
    session, mean, std = load_classifier(MODELS_DIR)
    silence = np.zeros(16000, dtype=np.float32)

    predicted_class, probabilities = classify_clip(silence, session, mean, std)

    assert predicted_class in {"ruido", "pular", "abaixa"}
    assert probabilities.shape == (3,)
    assert abs(probabilities.sum() - 1.0) < 1e-4


def test_decide_class_accepts_confident_prediction():
    probabilities = np.array([0.05, 0.9, 0.05], dtype=np.float32)  # confidently "pular"
    assert decide_class(probabilities) == "pular"


def test_decide_class_rejects_low_confidence_to_ruido():
    probabilities = np.array([0.34, 0.33, 0.33], dtype=np.float32)  # ambiguous
    assert probabilities.max() < CONFIDENCE_THRESHOLD
    assert decide_class(probabilities) == "ruido"
