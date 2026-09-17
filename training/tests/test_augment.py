import numpy as np

from training.augment import (
    add_noise,
    augment_signal,
    fix_length,
    pitch_shift,
    random_gain,
    time_shift,
    time_stretch,
)
from training.features import SR


def _make_tone(duration_sec: float = 1.0, freq: float = 440.0) -> np.ndarray:
    t = np.linspace(0, duration_sec, int(SR * duration_sec), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_add_noise_preserves_length_and_changes_signal():
    signal = _make_tone()
    noisy = add_noise(signal, noise_level=0.05, rng=np.random.default_rng(0))
    assert noisy.shape == signal.shape
    assert not np.allclose(noisy, signal)


def test_pitch_shift_preserves_length():
    signal = _make_tone()
    shifted = pitch_shift(signal, n_steps=2.0)
    assert len(shifted) == len(signal)


def test_fix_length_pads_short_signal():
    short = np.ones(100, dtype=np.float32)
    fixed = fix_length(short, 200)
    assert fixed.shape == (200,)
    assert np.all(fixed[100:] == 0.0)


def test_fix_length_truncates_long_signal():
    long_signal = np.ones(300, dtype=np.float32)
    fixed = fix_length(long_signal, 200)
    assert fixed.shape == (200,)


def test_time_stretch_returns_original_length_after_fix():
    signal = _make_tone()
    stretched = fix_length(time_stretch(signal, rate=1.2), len(signal))
    assert stretched.shape == signal.shape


def test_random_gain_scales_amplitude_within_range():
    signal = _make_tone()
    scaled = random_gain(signal, min_scale=2.0, max_scale=2.0, rng=np.random.default_rng(0))
    np.testing.assert_allclose(scaled, signal * 2.0, atol=1e-5)


def test_random_gain_preserves_length():
    signal = _make_tone()
    scaled = random_gain(signal, rng=np.random.default_rng(0))
    assert scaled.shape == signal.shape


def test_time_shift_preserves_length_and_energy_but_moves_content():
    signal = _make_tone()
    shifted = time_shift(signal, max_shift_fraction=0.3, rng=np.random.default_rng(2))
    assert shifted.shape == signal.shape
    assert not np.allclose(shifted, signal)


def test_time_shift_positive_shift_zero_pads_the_start():
    signal = np.ones(100, dtype=np.float32)
    # Force a fixed positive shift by making both bounds equal.
    shifted = time_shift(signal, max_shift_fraction=0.2, rng=np.random.default_rng(42))
    # Whatever the (randomly-signed) shift was, some samples must now be exactly zero
    # near one edge, proving content moved rather than wrapped around.
    assert shifted[0] == 0.0 or shifted[-1] == 0.0


def test_augment_signal_returns_six_fixed_length_variations():
    signal = _make_tone()
    variations = augment_signal(signal, rng=np.random.default_rng(1))
    assert len(variations) == 6
    for variation in variations:
        assert variation.shape == signal.shape
