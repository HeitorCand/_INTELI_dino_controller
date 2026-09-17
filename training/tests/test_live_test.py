from pathlib import Path

import numpy as np
import pytest

from training.features import SR
from training.live_test import (
    CONFIDENCE_THRESHOLD,
    MAX_COMMAND_DURATION,
    classify_clip,
    compute_rms,
    decide_class,
    load_classifier,
    softmax,
)

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


def test_decide_class_accepts_confident_short_prediction():
    probabilities = np.array([0.05, 0.9, 0.05], dtype=np.float32)  # confidently "pular"
    decided, reason = decide_class(probabilities, capture_duration=0.8, hit_max_duration=False)
    assert decided == "pular"
    assert reason == ""


def test_decide_class_rejects_low_confidence_to_ruido():
    probabilities = np.array([0.34, 0.33, 0.33], dtype=np.float32)  # ambiguous
    assert probabilities.max() < CONFIDENCE_THRESHOLD
    decided, reason = decide_class(probabilities, capture_duration=0.8, hit_max_duration=False)
    assert decided == "ruido"
    assert reason != ""


def test_decide_class_rejects_utterance_that_hit_max_duration():
    probabilities = np.array([0.05, 0.9, 0.05], dtype=np.float32)  # would-be confident "pular"
    decided, reason = decide_class(probabilities, capture_duration=1.5, hit_max_duration=True)
    assert decided == "ruido"
    assert "longa" in reason


def test_decide_class_rejects_long_utterance_even_without_hitting_the_cap():
    probabilities = np.array([0.05, 0.9, 0.05], dtype=np.float32)
    decided, reason = decide_class(
        probabilities, capture_duration=MAX_COMMAND_DURATION + 0.01, hit_max_duration=False
    )
    assert decided == "ruido"
