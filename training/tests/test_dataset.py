import numpy as np
import soundfile as sf

from training.dataset import build_dataset, split_dataset
from training.features import SR, FEATURE_DIM


def _write_tone(path, freq, duration=1.0):
    t = np.linspace(0, duration, int(SR * duration), endpoint=False)
    signal = (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    sf.write(str(path), signal, SR)


def _make_raw_dir(tmp_path):
    raw_dir = tmp_path / "raw"
    freqs = {"ruido": 200.0, "pular": 440.0, "abaixa": 880.0}
    for class_name, freq in freqs.items():
        class_dir = raw_dir / class_name
        class_dir.mkdir(parents=True)
        for i in range(3):
            _write_tone(class_dir / f"clip_{i}.wav", freq)
    return raw_dir


def test_build_dataset_without_augment_has_one_row_per_clip(tmp_path):
    raw_dir = _make_raw_dir(tmp_path)
    X, y = build_dataset(raw_dir, augment=False)
    assert X.shape == (9, FEATURE_DIM)
    assert set(y.tolist()) == {0, 1, 2}


def test_build_dataset_with_augment_multiplies_rows_by_four(tmp_path):
    raw_dir = _make_raw_dir(tmp_path)
    X, _ = build_dataset(raw_dir, augment=True)
    assert X.shape[0] == 9 * 4


def test_split_dataset_preserves_total_count_and_label_set(tmp_path):
    raw_dir = _make_raw_dir(tmp_path)
    X, y = build_dataset(raw_dir, augment=True)
    X_train, X_test, y_train, y_test = split_dataset(X, y, test_size=0.25)
    assert len(X_train) + len(X_test) == len(X)
    assert len(y_test) > 0
    assert set(y_test.tolist()).issubset({0, 1, 2})
