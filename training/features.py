import numpy as np

from training.dsp import compute_centroid_from_magnitudes, compute_frame_magnitudes, compute_mfcc_from_magnitudes

SR = 16000
N_MFCC = 11
FEATURE_DIM = 1 + 1 + 1 + N_MFCC  # RMS + ZCR + spectral centroid + MFCC means


def compute_rms(signal: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(signal))))


def compute_zero_crossing_rate(signal: np.ndarray) -> float:
    signs = np.sign(signal)
    signs[signs == 0] = 1
    crossings = np.sum(signs[1:] != signs[:-1])
    return float(crossings) / len(signal)


def normalize_amplitude(signal: np.ndarray, target_peak: float = 0.5) -> np.ndarray:
    """Peak-normalizes a signal so spectral-shape features (centroid, MFCC) reflect
    the sound's shape rather than the recording device's absolute gain. RMS is
    deliberately computed on the raw (non-normalized) signal elsewhere, since it is
    the one feature meant to carry loudness information (speech vs. quiet/ruido)."""
    peak = float(np.max(np.abs(signal)))
    if peak < 1e-6:
        return signal
    return (signal / peak * target_peak).astype(np.float32)


def compute_spectral_centroid(signal: np.ndarray, sr: int = SR) -> float:
    magnitudes = compute_frame_magnitudes(signal)
    return compute_centroid_from_magnitudes(magnitudes, sr=sr)


def compute_mfcc_means(signal: np.ndarray, sr: int = SR, n_mfcc: int = N_MFCC) -> np.ndarray:
    magnitudes = compute_frame_magnitudes(signal)
    return compute_mfcc_from_magnitudes(magnitudes, n_mfcc=n_mfcc)


def extract_features(signal: np.ndarray, sr: int = SR) -> np.ndarray:
    rms = compute_rms(signal)
    zcr = compute_zero_crossing_rate(signal)
    shape_signal = normalize_amplitude(signal)

    magnitudes = compute_frame_magnitudes(shape_signal)
    centroid = compute_centroid_from_magnitudes(magnitudes, sr=sr)
    mfcc_means = compute_mfcc_from_magnitudes(magnitudes, n_mfcc=N_MFCC)

    return np.concatenate([[rms], [zcr], [centroid], mfcc_means]).astype(np.float32)
