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


def build_dataset(
    raw_dir: Path, augment: bool = True, rng_seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(rng_seed)
    features: List[np.ndarray] = []
    labels: List[int] = []

    for label_index, class_name in enumerate(CLASSES):
        class_dir = raw_dir / class_name
        if not class_dir.is_dir():
            continue
        for wav_path in sorted(class_dir.glob("*.wav")):
            clip = load_clip(wav_path)
            variations = augment_signal(clip, sr=SR, rng=rng) if augment else [clip]
            for variation in variations:
                features.append(extract_features(variation, sr=SR))
                labels.append(label_index)

    return np.stack(features), np.array(labels, dtype=np.int64)


def split_dataset(
    X: np.ndarray, y: np.ndarray, test_size: float = 0.2, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
