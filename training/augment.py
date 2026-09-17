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


def random_gain(
    signal: np.ndarray,
    min_scale: float = 0.3,
    max_scale: float = 3.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Randomly rescales amplitude to simulate different microphones/distances/gain
    staging, without needing new recordings from those devices."""
    rng = rng or np.random.default_rng()
    scale = float(rng.uniform(min_scale, max_scale))
    return (signal * scale).astype(np.float32)


def time_shift(
    signal: np.ndarray, max_shift_fraction: float = 0.3, rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """Randomly shifts where the sound sits within the fixed-length window (zero-padding
    the vacated side, no wraparound), so the model doesn't depend on the word always
    starting at the same offset."""
    rng = rng or np.random.default_rng()
    target_length = len(signal)
    max_shift = int(target_length * max_shift_fraction)
    if max_shift == 0:
        return signal.astype(np.float32)
    shift = int(rng.integers(-max_shift, max_shift + 1))

    shifted = np.zeros(target_length, dtype=np.float32)
    if shift >= 0:
        shifted[shift:] = signal[: target_length - shift]
    else:
        shifted[: target_length + shift] = signal[-shift:]
    return shifted


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
    variations.append(random_gain(signal, rng=rng))
    variations.append(time_shift(signal, rng=rng))
    return variations
