import numpy as np

from training.features import (
    FEATURE_DIM,
    SR,
    compute_rms,
    compute_zero_crossing_rate,
    extract_features,
    normalize_amplitude,
)


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


def test_normalize_amplitude_scales_peak_to_target():
    t = np.linspace(0, 1, SR, endpoint=False)
    signal = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    normalized = normalize_amplitude(signal, target_peak=0.5)
    assert abs(np.max(np.abs(normalized)) - 0.5) < 1e-4


def test_normalize_amplitude_leaves_silence_unchanged():
    silence = np.zeros(SR, dtype=np.float32)
    assert np.array_equal(normalize_amplitude(silence), silence)


def test_extract_features_spectral_shape_is_gain_invariant():
    t = np.linspace(0, 1, SR, endpoint=False)
    quiet_tone = (0.02 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    loud_tone = (0.4 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    quiet_features = extract_features(quiet_tone)
    loud_features = extract_features(loud_tone)

    # RMS (index 0) still reflects the real loudness difference...
    assert quiet_features[0] < loud_features[0]
    # ...but ZCR, spectral centroid and MFCCs (indices 1+) are gain-invariant.
    np.testing.assert_allclose(quiet_features[1:], loud_features[1:], atol=1e-3)


def test_compute_zero_crossing_rate_is_higher_for_higher_frequency():
    t = np.linspace(0, 1, SR, endpoint=False)
    low_freq = (0.5 * np.sin(2 * np.pi * 100 * t)).astype(np.float32)
    high_freq = (0.5 * np.sin(2 * np.pi * 4000 * t)).astype(np.float32)
    assert compute_zero_crossing_rate(low_freq) < compute_zero_crossing_rate(high_freq)
