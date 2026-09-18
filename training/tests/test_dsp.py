import numpy as np

from training.dsp import (
    HOP_LENGTH,
    MEL_FILTERBANK,
    N_FFT,
    N_MELS,
    build_dct_matrix,
    build_mel_filterbank,
    compute_centroid_from_magnitudes,
    compute_frame_magnitudes,
    compute_mfcc_from_magnitudes,
    frame_signal,
    hann_window,
    hz_to_mel,
    mel_to_hz,
)

SR = 16000


def _make_tone(freq: float, duration_sec: float = 1.0, sr: int = SR) -> np.ndarray:
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    return (0.8 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_hz_to_mel_and_back_are_inverses():
    freqs = np.array([0.0, 100.0, 1000.0, 4000.0, 8000.0])
    mel = hz_to_mel(freqs)
    recovered = mel_to_hz(mel)
    np.testing.assert_allclose(recovered, freqs, atol=1e-3)


def test_hann_window_starts_and_ends_near_zero_and_peaks_at_one():
    window = hann_window(512)
    assert window.shape == (512,)
    assert window[0] < 0.01
    assert abs(window[256] - 1.0) < 0.01


def test_build_mel_filterbank_shape_and_nonzero_filters():
    filters = build_mel_filterbank(SR, N_FFT, N_MELS)
    assert filters.shape == (N_MELS, N_FFT // 2 + 1)
    # Every filter should touch at least one FFT bin.
    assert np.all(filters.sum(axis=1) > 0)


def test_build_dct_matrix_shape():
    matrix = build_dct_matrix(N_MELS, 11)
    assert matrix.shape == (11, N_MELS)


def test_frame_signal_returns_expected_frame_count():
    signal = np.zeros(16000, dtype=np.float32)
    frames = frame_signal(signal, n_fft=N_FFT, hop_length=HOP_LENGTH)
    expected_frames = 1 + (16000 - N_FFT) // HOP_LENGTH
    assert frames.shape == (expected_frames, N_FFT)


def test_compute_frame_magnitudes_peaks_near_tone_frequency_bin():
    tone_freq = 2000.0
    signal = _make_tone(tone_freq)
    magnitudes = compute_frame_magnitudes(signal)

    mean_spectrum = magnitudes.mean(axis=0)
    peak_bin = int(np.argmax(mean_spectrum))
    freqs = np.fft.rfftfreq(N_FFT, d=1.0 / SR)
    peak_freq = freqs[peak_bin]

    assert abs(peak_freq - tone_freq) < (SR / N_FFT) * 2  # within ~2 FFT bins


def test_compute_centroid_from_magnitudes_matches_pure_tone_frequency():
    tone_freq = 3000.0
    signal = _make_tone(tone_freq)
    magnitudes = compute_frame_magnitudes(signal)
    centroid = compute_centroid_from_magnitudes(magnitudes, sr=SR)

    assert abs(centroid - tone_freq) < 200.0  # generous tolerance for windowing spread


def test_compute_centroid_is_higher_for_higher_frequency_tone():
    low = compute_centroid_from_magnitudes(compute_frame_magnitudes(_make_tone(300.0)), sr=SR)
    high = compute_centroid_from_magnitudes(compute_frame_magnitudes(_make_tone(5000.0)), sr=SR)
    assert low < high


def test_compute_mfcc_from_magnitudes_has_correct_shape():
    signal = _make_tone(1000.0)
    magnitudes = compute_frame_magnitudes(signal)
    mfcc = compute_mfcc_from_magnitudes(magnitudes, n_mfcc=11)
    assert mfcc.shape == (11,)
    assert np.all(np.isfinite(mfcc))


def test_mel_filterbank_module_constant_matches_builder():
    rebuilt = build_mel_filterbank(SR, N_FFT, N_MELS)
    np.testing.assert_array_equal(MEL_FILTERBANK, rebuilt)
