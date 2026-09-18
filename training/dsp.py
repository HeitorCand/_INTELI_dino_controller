"""Custom, from-scratch FFT/mel-filterbank/DCT pipeline for spectral centroid and
MFCC, sized for an embedded target (ESP32) rather than librosa's desktop-scale
defaults (2048-point FFT, 128 mel bands). The exact same formulas are ported to C
in firmware/ and verified numerically against this module - what matters is that
training and on-device inference agree with EACH OTHER, not with librosa."""

import numpy as np

N_FFT = 512
HOP_LENGTH = 256
N_MELS = 26


def hann_window(n: int) -> np.ndarray:
    return (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / (n - 1))).astype(np.float32)


def hz_to_mel(frequency_hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + frequency_hz / 700.0)


def mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def build_mel_filterbank(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
    """Returns a (n_mels, n_fft//2+1) matrix of triangular mel filters spanning
    0 Hz to sr/2, using the classic HTK-style mel scale."""
    n_bins = n_fft // 2 + 1
    fmax = sr / 2.0
    mel_points = np.linspace(hz_to_mel(np.array(0.0)), hz_to_mel(np.array(fmax)), n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    filters = np.zeros((n_mels, n_bins), dtype=np.float32)
    for m in range(1, n_mels + 1):
        left, center, right = bin_points[m - 1], bin_points[m], bin_points[m + 1]
        if center == left:
            center += 1
        if right == center:
            right += 1

        rising = np.arange(left, center)
        rising = rising[(rising >= 0) & (rising < n_bins)]
        filters[m - 1, rising] = (rising - left) / (center - left)

        falling = np.arange(center, right)
        falling = falling[(falling >= 0) & (falling < n_bins)]
        filters[m - 1, falling] = (right - falling) / (right - center)

    return filters


def build_dct_matrix(n_in: int, n_out: int) -> np.ndarray:
    """DCT-II basis, (n_out, n_in): mfcc = log_mel_energies @ DCT_MATRIX.T"""
    k = np.arange(n_out).reshape(-1, 1)
    i = np.arange(n_in).reshape(1, -1)
    return (2.0 * np.cos(np.pi * k * (2 * i + 1) / (2 * n_in))).astype(np.float32)


MEL_FILTERBANK = build_mel_filterbank(16000, N_FFT, N_MELS)  # (N_MELS, N_FFT//2+1)


def frame_signal(signal: np.ndarray, n_fft: int = N_FFT, hop_length: int = HOP_LENGTH) -> np.ndarray:
    n_frames = 1 + (len(signal) - n_fft) // hop_length
    frames = np.zeros((n_frames, n_fft), dtype=np.float32)
    for i in range(n_frames):
        start = i * hop_length
        frames[i] = signal[start : start + n_fft]
    return frames


def compute_frame_magnitudes(
    signal: np.ndarray, n_fft: int = N_FFT, hop_length: int = HOP_LENGTH
) -> np.ndarray:
    """Returns (n_frames, n_fft//2+1) magnitude spectra of Hann-windowed frames."""
    frames = frame_signal(signal, n_fft, hop_length)
    windowed = frames * hann_window(n_fft)
    spectrum = np.fft.rfft(windowed, n=n_fft, axis=1)
    return np.abs(spectrum).astype(np.float32)


def compute_centroid_from_magnitudes(magnitudes: np.ndarray, sr: int, n_fft: int = N_FFT) -> float:
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr).astype(np.float32)
    numerator = (magnitudes * freqs).sum(axis=1)
    denominator = magnitudes.sum(axis=1) + 1e-10
    return float(np.mean(numerator / denominator))


def compute_mfcc_from_magnitudes(magnitudes: np.ndarray, n_mfcc: int) -> np.ndarray:
    mel_energies = magnitudes @ MEL_FILTERBANK.T  # (frames, N_MELS)
    log_mel = np.log(mel_energies + 1e-6)
    dct_matrix = build_dct_matrix(N_MELS, n_mfcc)
    mfcc_frames = log_mel @ dct_matrix.T  # (frames, n_mfcc)
    return np.mean(mfcc_frames, axis=0).astype(np.float32)
