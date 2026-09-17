import numpy as np
import librosa

SR = 16000
N_MFCC = 11
FEATURE_DIM = 1 + 1 + N_MFCC  # RMS + spectral centroid + MFCC means


def compute_rms(signal: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(signal))))


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
    centroid = librosa.feature.spectral_centroid(y=signal, sr=sr)
    return float(np.mean(centroid))


def compute_mfcc_means(signal: np.ndarray, sr: int = SR, n_mfcc: int = N_MFCC) -> np.ndarray:
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc, axis=1)


def extract_features(signal: np.ndarray, sr: int = SR) -> np.ndarray:
    rms = compute_rms(signal)
    shape_signal = normalize_amplitude(signal)
    centroid = compute_spectral_centroid(shape_signal, sr)
    mfcc_means = compute_mfcc_means(shape_signal, sr)
    return np.concatenate([[rms], [centroid], mfcc_means]).astype(np.float32)
