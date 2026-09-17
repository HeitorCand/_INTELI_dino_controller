from typing import List, Optional

import numpy as np
import librosa

from training.features import SR


def add_noise(
    signal: np.ndarray, noise_level: float = 0.01, rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    rng = rng or np.random.default_rng()
    noise = rng.normal(0, noise_level, size=signal.shape).astype(np.float32)
    return (signal + noise).astype(np.float32)


def pitch_shift(signal: np.ndarray, sr: int = SR, n_steps: float = 2.0) -> np.ndarray:
    shifted = librosa.effects.pitch_shift(y=signal, sr=sr, n_steps=n_steps)
    return shifted.astype(np.float32)


def time_stretch(signal: np.ndarray, rate: float = 1.1) -> np.ndarray:
    stretched = librosa.effects.time_stretch(y=signal, rate=rate)
    return stretched.astype(np.float32)


def fix_length(signal: np.ndarray, target_length: int) -> np.ndarray:
    if len(signal) >= target_length:
        return signal[:target_length].astype(np.float32)
    padded = np.zeros(target_length, dtype=np.float32)
    padded[: len(signal)] = signal
    return padded


def augment_signal(
    signal: np.ndarray, sr: int = SR, rng: Optional[np.random.Generator] = None
) -> List[np.ndarray]:
    rng = rng or np.random.default_rng()
    target_length = len(signal)
    variations = [signal]
    variations.append(add_noise(signal, noise_level=0.01, rng=rng))
    variations.append(
        fix_length(pitch_shift(signal, sr=sr, n_steps=float(rng.uniform(-2, 2))), target_length)
    )
    variations.append(
        fix_length(time_stretch(signal, rate=float(rng.uniform(0.85, 1.15))), target_length)
    )
    return variations
