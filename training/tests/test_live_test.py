from pathlib import Path

import numpy as np
import pytest

from training.features import SR
from training.live_test import classify_clip, compute_rms, load_classifier, softmax

MODELS_DIR = Path(__file__).parent.parent.parent / "models"


def test_compute_rms_of_silence_is_zero():
    assert compute_rms(np.zeros(SR, dtype=np.float32)) == 0.0


def test_softmax_sums_to_one_and_picks_largest_logit():
    probabilities = softmax(np.array([1.0, 5.0, 2.0], dtype=np.float32))
    assert abs(probabilities.sum() - 1.0) < 1e-6
    assert int(np.argmax(probabilities)) == 1


@pytest.mark.skipif(not (MODELS_DIR / "model.onnx").exists(), reason="trained model not present")
def test_classify_clip_on_silence_returns_a_valid_class_and_distribution():
    session, mean, std = load_classifier(MODELS_DIR)
    silence = np.zeros(SR, dtype=np.float32)

    predicted_class, probabilities = classify_clip(silence, session, mean, std)

    assert predicted_class in {"ruido", "pular", "abaixa"}
    assert probabilities.shape == (3,)
    assert abs(probabilities.sum() - 1.0) < 1e-4
