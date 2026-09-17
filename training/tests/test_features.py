import numpy as np

from training.features import compute_rms, extract_features, FEATURE_DIM, SR


def test_rms_of_silence_is_zero():
    signal = np.zeros(SR, dtype=np.float32)
    assert compute_rms(signal) == 0.0


def test_rms_of_sine_wave_matches_amplitude_over_sqrt2():
    t = np.linspace(0, 1, SR, endpoint=False)
    amplitude = 0.8
    signal = (amplitude * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    rms = compute_rms(signal)
    expected = amplitude / np.sqrt(2)
    assert abs(rms - expected) < 0.01


def test_extract_features_has_correct_shape_and_dtype():
    signal = np.random.default_rng(0).uniform(-0.1, 0.1, SR).astype(np.float32)
    features = extract_features(signal)
    assert features.shape == (FEATURE_DIM,)
    assert features.dtype == np.float32


def test_extract_features_silence_has_lower_rms_than_tone():
    silence = np.zeros(SR, dtype=np.float32)
    t = np.linspace(0, 1, SR, endpoint=False)
    tone = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    silence_features = extract_features(silence)
    tone_features = extract_features(tone)
    assert silence_features[0] < tone_features[0]
