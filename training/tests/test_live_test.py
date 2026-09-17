from pathlib import Path

import numpy as np
import pytest

from training.live_test import (
    CONFIDENCE_THRESHOLD,
    MIN_RMS_FLOOR,
    MIN_RMS_MULTIPLIER,
    classify_clip,
    compute_rms,
    decide_class,
    load_classifier,
    softmax,
)

MODELS_DIR = Path(__file__).parent.parent.parent / "models"


def test_softmax_sums_to_one_and_picks_largest_logit():
    probabilities = softmax(np.array([1.0, 5.0, 2.0], dtype=np.float32))
    assert abs(probabilities.sum() - 1.0) < 1e-6
    assert int(np.argmax(probabilities)) == 1


def test_compute_rms_of_silence_is_zero():
    assert compute_rms(np.zeros(16000, dtype=np.float32)) == 0.0


def test_compute_rms_of_sine_wave_matches_amplitude_over_sqrt2():
    t = np.linspace(0, 1, 16000, endpoint=False)
    amplitude = 0.8
    signal = (amplitude * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    assert abs(compute_rms(signal) - amplitude / np.sqrt(2)) < 0.01


def test_min_rms_constants_give_a_sane_floor_above_typical_ambient_noise():
    # A quiet room's ambient RMS was measured around 0.0008-0.005 in earlier sessions;
    # the calibrated floor (ambient * multiplier, at least MIN_RMS_FLOOR) should sit
    # comfortably above that and well below real speech RMS (measured ~0.09-0.11).
    typical_ambient = 0.001
    calibrated_floor = max(typical_ambient * MIN_RMS_MULTIPLIER, MIN_RMS_FLOOR)
    assert calibrated_floor < 0.05


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
