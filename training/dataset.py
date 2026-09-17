from pathlib import Path
from typing import List, Tuple

import librosa
import numpy as np
from sklearn.model_selection import train_test_split

from training.augment import augment_signal, fix_length
from training.features import extract_features, SR

CLASSES = ["ruido", "pular", "abaixa"]
CLIP_DURATION_SEC = 1.0
CLIP_LENGTH = int(SR * CLIP_DURATION_SEC)


def load_clip(path: Path) -> np.ndarray:
    signal, _ = librosa.load(str(path), sr=SR, mono=True)
    return fix_length(signal.astype(np.float32), CLIP_LENGTH)


def load_raw_clips(raw_dir: Path) -> Tuple[List[np.ndarray], np.ndarray]:
    """Loads one unaugmented signal per raw wav file. Returns (signals, labels)."""
    signals: List[np.ndarray] = []
    labels: List[int] = []
    for label_index, class_name in enumerate(CLASSES):
        class_dir = raw_dir / class_name
        if not class_dir.is_dir():
            continue
        for wav_path in sorted(class_dir.glob("*.wav")):
            signals.append(load_clip(wav_path))
            labels.append(label_index)
    return signals, np.array(labels, dtype=np.int64)


def split_raw_clips(
    signals: List[np.ndarray], labels: np.ndarray, test_size: float = 0.2, seed: int = 42
) -> Tuple[List[np.ndarray], np.ndarray, List[np.ndarray], np.ndarray]:
    """Splits at the raw-clip level (before augmentation) so augmented siblings of a
    clip can never end up on both sides of the split."""
    indices = np.arange(len(signals))
    train_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=seed, stratify=labels
    )
    train_signals = [signals[i] for i in train_idx]
    test_signals = [signals[i] for i in test_idx]
    return train_signals, labels[train_idx], test_signals, labels[test_idx]


def build_features(
    signals: List[np.ndarray], labels: np.ndarray, augment: bool, rng_seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Extracts features from the given raw signals. When augment=True, each signal is
    expanded into its augmented variants first (use this for the training partition
    only — never augment the test partition, so test accuracy reflects real recordings)."""
    rng = np.random.default_rng(rng_seed)
    features: List[np.ndarray] = []
    out_labels: List[int] = []
    for signal, label in zip(signals, labels):
        variations = augment_signal(signal, sr=SR, rng=rng) if augment else [signal]
        for variation in variations:
            features.append(extract_features(variation, sr=SR))
            out_labels.append(int(label))
    return np.stack(features), np.array(out_labels, dtype=np.int64)
