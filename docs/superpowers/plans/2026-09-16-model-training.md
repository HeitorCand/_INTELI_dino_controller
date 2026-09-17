# Treino do Modelo de Detecção de Comandos de Voz — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a trained, tested, ONNX-exported classifier that distinguishes three
acoustic classes — `ruido`, `pular`, `abaixa` — from short audio clips, ready to be
ported into the ESP32 firmware in a later plan.

**Architecture:** Python training pipeline: record labeled WAV clips → extract features
(RMS, Spectral Centroid, MFCC means) → augment (noise/pitch/time-stretch) to compensate
for a small per-class sample count and improve cross-speaker generalization → train a
small MLP (PyTorch) → evaluate (accuracy, confusion matrix) → export to ONNX plus the
feature normalization stats needed later for the on-device C port.

**Tech Stack:** Python 3, numpy, scipy, librosa, soundfile, sounddevice, scikit-learn
(train/test split), PyTorch (model + training), onnx / onnxruntime (export + validation),
pytest.

**Spec:** `docs/superpowers/specs/2026-09-16-dino-voice-controller-design.md`

## Global Constraints

- Sample rate: 16000 Hz, mono — matches the INMP441's expected I2S configuration used
  later on-device; all loaded/recorded audio is resampled/captured at this rate.
- Clip duration: 1.0 second, fixed length (16000 samples) — short spoken commands.
- Classes (fixed order, index = label): `["ruido", "pular", "abaixa"]`.
- Feature vector: 13 values — `[RMS, spectral_centroid, mfcc_mean_0..10]`
  (`N_MFCC = 11`).
- Small per-class dataset (~10-15 raw recordings per class) expected to generalize to a
  second speaker (e.g. a professor testing it) — data augmentation is required, not
  optional, to reduce overfitting to a single voice.
- Model must be exported as `.onnx` — required deliverable for the ponderada.
- Feature normalization (mean/std) computed on the training split must be persisted
  (`.npy` files) alongside the model, since the firmware will need to replicate the same
  normalization before running the forward pass on-device.

---

## Task 1: Project scaffolding

**Files:**
- Create: `training/__init__.py` (empty)
- Create: `training/requirements.txt`
- Create: `pytest.ini`
- Create: `.gitignore`

**Interfaces:**
- Produces: an importable `training` package usable as `from training.<module> import ...`
  in every later task; a working `pytest` command run from the repo root.

- [ ] **Step 1: Create the package and dependency list**

`training/__init__.py`:
```python
```
(empty file — marks `training/` as a package)

`training/requirements.txt`:
```
numpy
scipy
librosa
soundfile
sounddevice
scikit-learn
torch
onnx
onnxruntime
pytest
```

- [ ] **Step 2: Configure pytest to resolve `training.*` imports from repo root**

`pytest.ini`:
```ini
[pytest]
pythonpath = .
```

- [ ] **Step 3: Ignore recordings and local environment files**

`.gitignore`:
```
training/data/raw/
.venv/
__pycache__/
*.pyc
.pytest_cache/
```

- [ ] **Step 4: Create and activate a virtual environment, install dependencies**

Run:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r training/requirements.txt
```
Expected: install completes with no errors (torch may take a few minutes to download).

- [ ] **Step 5: Verify pytest runs (no tests yet, should report 0 collected)**

Run: `pytest`
Expected: `no tests ran` (exit code may be 5, that's fine — confirms pytest + pythonpath config work)

- [ ] **Step 6: Commit**

```bash
git add training/__init__.py training/requirements.txt pytest.ini .gitignore
git commit -m "Scaffold Python training project structure"
```

---

## Task 2: Feature extraction

**Files:**
- Create: `training/features.py`
- Test: `training/tests/test_features.py`
- Create: `training/tests/__init__.py` (empty)

**Interfaces:**
- Produces: `SR = 16000`, `N_MFCC = 11`, `FEATURE_DIM = 13`,
  `compute_rms(signal: np.ndarray) -> float`,
  `compute_spectral_centroid(signal: np.ndarray, sr: int = SR) -> float`,
  `compute_mfcc_means(signal: np.ndarray, sr: int = SR, n_mfcc: int = N_MFCC) -> np.ndarray`,
  `extract_features(signal: np.ndarray, sr: int = SR) -> np.ndarray` (shape `(FEATURE_DIM,)`,
  dtype `float32`) — used by `dataset.py` (Task 4) and `train.py` (Task 6).

- [ ] **Step 1: Write the failing tests**

`training/tests/__init__.py`:
```python
```

`training/tests/test_features.py`:
```python
import numpy as np

from training.features import compute_rms, extract_features, FEATURE_DIM, SR


def test_rms_of_silence_is_zero():
    signal = np.zeros(SR, dtype=np.float32)
    assert compute_rms(signal) == 0.0


def test_rms_of_sine_wave_matches_amplitude_over_sqrt2():
    t = np.linspace(0, 1, SR, endpoint=False)
    amplitude = 0.8
    signal = (amplitude * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    rms = compute_rms(signal)
    expected = amplitude / np.sqrt(2)
    assert abs(rms - expected) < 0.01


def test_extract_features_has_correct_shape_and_dtype():
    signal = np.random.default_rng(0).uniform(-0.1, 0.1, SR).astype(np.float32)
    features = extract_features(signal)
    assert features.shape == (FEATURE_DIM,)
    assert features.dtype == np.float32


def test_extract_features_silence_has_lower_rms_than_tone():
    silence = np.zeros(SR, dtype=np.float32)
    t = np.linspace(0, 1, SR, endpoint=False)
    tone = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    silence_features = extract_features(silence)
    tone_features = extract_features(tone)
    assert silence_features[0] < tone_features[0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest training/tests/test_features.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'training.features'`

- [ ] **Step 3: Implement `training/features.py`**

```python
import numpy as np
import librosa

SR = 16000
N_MFCC = 11
FEATURE_DIM = 1 + 1 + N_MFCC  # RMS + spectral centroid + MFCC means


def compute_rms(signal: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(signal))))


def compute_spectral_centroid(signal: np.ndarray, sr: int = SR) -> float:
    centroid = librosa.feature.spectral_centroid(y=signal, sr=sr)
    return float(np.mean(centroid))


def compute_mfcc_means(signal: np.ndarray, sr: int = SR, n_mfcc: int = N_MFCC) -> np.ndarray:
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc, axis=1)


def extract_features(signal: np.ndarray, sr: int = SR) -> np.ndarray:
    rms = compute_rms(signal)
    centroid = compute_spectral_centroid(signal, sr)
    mfcc_means = compute_mfcc_means(signal, sr)
    return np.concatenate([[rms], [centroid], mfcc_means]).astype(np.float32)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest training/tests/test_features.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add training/features.py training/tests/__init__.py training/tests/test_features.py
git commit -m "Add acoustic feature extraction (RMS, spectral centroid, MFCC)"
```

---

## Task 3: Data augmentation

**Files:**
- Create: `training/augment.py`
- Test: `training/tests/test_augment.py`

**Interfaces:**
- Consumes: `SR` from `training.features` (Task 2).
- Produces: `add_noise(signal, noise_level=0.01, rng=None) -> np.ndarray`,
  `pitch_shift(signal, sr=SR, n_steps=2.0) -> np.ndarray`,
  `time_stretch(signal, rate=1.1) -> np.ndarray`,
  `fix_length(signal, target_length: int) -> np.ndarray`,
  `augment_signal(signal, sr=SR, rng=None) -> list[np.ndarray]` (all fixed-length,
  same length as input) — used by `dataset.py` (Task 4).

- [ ] **Step 1: Write the failing tests**

`training/tests/test_augment.py`:
```python
import numpy as np

from training.augment import add_noise, pitch_shift, time_stretch, fix_length, augment_signal
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


def test_augment_signal_returns_four_fixed_length_variations():
    signal = _make_tone()
    variations = augment_signal(signal, rng=np.random.default_rng(1))
    assert len(variations) == 4
    for variation in variations:
        assert variation.shape == signal.shape
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest training/tests/test_augment.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'training.augment'`

- [ ] **Step 3: Implement `training/augment.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest training/tests/test_augment.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add training/augment.py training/tests/test_augment.py
git commit -m "Add audio data augmentation (noise, pitch shift, time stretch)"
```

---

## Task 4: Dataset builder

**Files:**
- Create: `training/dataset.py`
- Test: `training/tests/test_dataset.py`

**Interfaces:**
- Consumes: `extract_features`, `SR` from `training.features` (Task 2);
  `augment_signal`, `fix_length` from `training.augment` (Task 3).
- Produces: `CLASSES = ["ruido", "pular", "abaixa"]`, `CLIP_LENGTH: int`,
  `load_clip(path: Path) -> np.ndarray`,
  `build_dataset(raw_dir: Path, augment: bool = True, rng_seed: int = 42) -> tuple[np.ndarray, np.ndarray]`
  (`X` shape `(N, FEATURE_DIM)`, `y` shape `(N,)` int64 labels),
  `split_dataset(X, y, test_size=0.2, seed=42) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]`
  — used by `train.py` (Task 6) and `record_audio.py`'s expected directory layout (Task 5).

- [ ] **Step 1: Write the failing tests**

`training/tests/test_dataset.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest training/tests/test_dataset.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'training.dataset'`

- [ ] **Step 3: Implement `training/dataset.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest training/tests/test_dataset.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add training/dataset.py training/tests/test_dataset.py
git commit -m "Add dataset builder (load clips, augment, extract features, split)"
```

---

## Task 5: Recording script

**Files:**
- Create: `training/record_audio.py`
- Test: `training/tests/test_record_audio.py`

**Interfaces:**
- Consumes: `SR` from `training.features` (Task 2); expects to write into
  `training/data/raw/<class_name>/clip_NNN.wav`, the same layout `build_dataset`
  (Task 4) reads from.
- Produces: `record_clip(duration_sec=1.0, sr=SR) -> np.ndarray` (hardware-dependent,
  not unit tested), `next_clip_index(class_dir: Path) -> int`,
  `save_clip(class_dir: Path, signal: np.ndarray, sr: int = SR) -> Path`.

- [ ] **Step 1: Write the failing tests**

`training/tests/test_record_audio.py`:
```python
import numpy as np

from training.record_audio import next_clip_index, save_clip


def test_next_clip_index_starts_at_zero(tmp_path):
    class_dir = tmp_path / "pular"
    class_dir.mkdir()
    assert next_clip_index(class_dir) == 0


def test_next_clip_index_counts_existing_clips(tmp_path):
    class_dir = tmp_path / "pular"
    class_dir.mkdir()
    (class_dir / "clip_000.wav").touch()
    (class_dir / "clip_001.wav").touch()
    assert next_clip_index(class_dir) == 2


def test_save_clip_writes_wav_file_and_increments_index(tmp_path):
    class_dir = tmp_path / "abaixa"
    signal = np.zeros(16000, dtype=np.float32)

    first_path = save_clip(class_dir, signal, sr=16000)
    second_path = save_clip(class_dir, signal, sr=16000)

    assert first_path.name == "clip_000.wav"
    assert second_path.name == "clip_001.wav"
    assert first_path.exists()
    assert second_path.exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest training/tests/test_record_audio.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'training.record_audio'`

- [ ] **Step 3: Implement `training/record_audio.py`**

```python
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

from training.features import SR

CLASSES = ["ruido", "pular", "abaixa"]
CLIP_DURATION_SEC = 1.0


def record_clip(duration_sec: float = CLIP_DURATION_SEC, sr: int = SR) -> np.ndarray:
    recording = sd.rec(int(duration_sec * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    return recording.flatten()


def next_clip_index(class_dir: Path) -> int:
    return len(list(class_dir.glob("clip_*.wav")))


def save_clip(class_dir: Path, signal: np.ndarray, sr: int = SR) -> Path:
    class_dir.mkdir(parents=True, exist_ok=True)
    index = next_clip_index(class_dir)
    path = class_dir / f"clip_{index:03d}.wav"
    sf.write(str(path), signal, sr)
    return path


def main():
    raw_dir = Path(__file__).parent / "data" / "raw"
    for class_name in CLASSES:
        class_dir = raw_dir / class_name
        print(f"\n=== Classe '{class_name}' ===")
        while True:
            response = input(
                f"  Enter para gravar {CLIP_DURATION_SEC}s de '{class_name}' "
                "(ou 'n' + Enter para próxima classe): "
            )
            if response.strip().lower() == "n":
                break
            recording = record_clip()
            path = save_clip(class_dir, recording)
            print(f"  Salvo: {path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest training/tests/test_record_audio.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add training/record_audio.py training/tests/test_record_audio.py
git commit -m "Add interactive recording script for labeled voice clips"
```

---

## Task 6: Model, training loop, evaluation, ONNX export

**Files:**
- Create: `training/train.py`
- Test: `training/tests/test_train.py`

**Interfaces:**
- Consumes: `build_dataset`, `split_dataset` from `training.dataset` (Task 4);
  `FEATURE_DIM` from `training.features` (Task 2); `CLASSES` from `training.dataset`.
- Produces: `CommandClassifier` (PyTorch `nn.Module`, `FEATURE_DIM -> 16 -> 3`),
  `normalize_features(X, mean, std) -> np.ndarray`,
  `train_model(X_train, y_train, epochs=200, lr=0.01) -> CommandClassifier`,
  `evaluate_model(model, X_test, y_test) -> str` (classification report + confusion
  matrix as text), `export_onnx(model, output_path: Path) -> None`, and a `main()`
  CLI entry point that runs the full pipeline and writes
  `models/model.onnx`, `models/evaluation.txt`, `models/feature_mean.npy`,
  `models/feature_std.npy`.

- [ ] **Step 1: Write the failing tests**

`training/tests/test_train.py`:
```python
import numpy as np
import onnxruntime as ort
import torch

from training.features import FEATURE_DIM
from training.train import (
    evaluate_model,
    export_onnx,
    normalize_features,
    train_model,
)


def _make_synthetic_dataset(samples_per_class=40, seed=0):
    rng = np.random.default_rng(seed)
    centers = {0: np.zeros(FEATURE_DIM), 1: np.full(FEATURE_DIM, 5.0), 2: np.full(FEATURE_DIM, -5.0)}
    X, y = [], []
    for label, center in centers.items():
        X.append(rng.normal(center, 0.5, size=(samples_per_class, FEATURE_DIM)))
        y.append(np.full(samples_per_class, label))
    return np.vstack(X).astype(np.float32), np.concatenate(y).astype(np.int64)


def test_train_model_learns_separable_classes():
    X, y = _make_synthetic_dataset()
    mean, std = X.mean(axis=0), X.std(axis=0) + 1e-8
    X_norm = normalize_features(X, mean, std)

    model = train_model(X_norm, y, epochs=200)

    model.eval()
    with torch.no_grad():
        predictions = torch.argmax(model(torch.from_numpy(X_norm)), dim=1).numpy()
    accuracy = (predictions == y).mean()
    assert accuracy > 0.9


def test_export_onnx_matches_torch_output(tmp_path):
    X, y = _make_synthetic_dataset()
    mean, std = X.mean(axis=0), X.std(axis=0) + 1e-8
    X_norm = normalize_features(X, mean, std)
    model = train_model(X_norm, y, epochs=200)

    onnx_path = tmp_path / "model.onnx"
    export_onnx(model, onnx_path)

    sample = X_norm[:1]
    model.eval()
    with torch.no_grad():
        torch_output = model(torch.from_numpy(sample)).numpy()

    session = ort.InferenceSession(str(onnx_path))
    onnx_output = session.run(None, {"features": sample.astype(np.float32)})[0]

    assert onnx_output.shape == (1, 3)
    np.testing.assert_allclose(torch_output, onnx_output, rtol=1e-4, atol=1e-4)


def test_evaluate_model_report_mentions_all_classes():
    X, y = _make_synthetic_dataset()
    mean, std = X.mean(axis=0), X.std(axis=0) + 1e-8
    X_norm = normalize_features(X, mean, std)
    model = train_model(X_norm, y, epochs=200)

    report = evaluate_model(model, X_norm, y)

    assert "pular" in report
    assert "abaixa" in report
    assert "ruido" in report
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest training/tests/test_train.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'training.train'`

- [ ] **Step 3: Implement `training/train.py`**

```python
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn

from training.dataset import CLASSES, build_dataset, split_dataset
from training.features import FEATURE_DIM

HIDDEN_DIM = 16
NUM_CLASSES = len(CLASSES)


class CommandClassifier(nn.Module):
    def __init__(
        self, input_dim: int = FEATURE_DIM, hidden_dim: int = HIDDEN_DIM, num_classes: int = NUM_CLASSES
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def normalize_features(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((X - mean) / std).astype(np.float32)


def train_model(
    X_train: np.ndarray, y_train: np.ndarray, epochs: int = 200, lr: float = 0.01
) -> CommandClassifier:
    model = CommandClassifier()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    X_tensor = torch.from_numpy(X_train.astype(np.float32))
    y_tensor = torch.from_numpy(y_train.astype(np.int64))

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(model(X_tensor), y_tensor)
        loss.backward()
        optimizer.step()

    return model


def evaluate_model(model: CommandClassifier, X_test: np.ndarray, y_test: np.ndarray) -> str:
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X_test.astype(np.float32)))
        predictions = torch.argmax(logits, dim=1).numpy()

    report = classification_report(y_test, predictions, target_names=CLASSES, zero_division=0)
    matrix = confusion_matrix(y_test, predictions)
    return f"{report}\n\nConfusion matrix (rows=true, cols=pred), order {CLASSES}:\n{matrix}"


def export_onnx(model: CommandClassifier, output_path: Path) -> None:
    model.eval()
    dummy_input = torch.zeros(1, FEATURE_DIM, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=["features"],
        output_names=["logits"],
        opset_version=13,
    )


def main():
    raw_dir = Path(__file__).parent / "data" / "raw"
    X, y = build_dataset(raw_dir, augment=True)
    X_train, X_test, y_train, y_test = split_dataset(X, y)

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    X_train_norm = normalize_features(X_train, mean, std)
    X_test_norm = normalize_features(X_test, mean, std)

    model = train_model(X_train_norm, y_train)
    report = evaluate_model(model, X_test_norm, y_test)
    print(report)

    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    export_onnx(model, models_dir / "model.onnx")
    (models_dir / "evaluation.txt").write_text(report)
    np.save(models_dir / "feature_mean.npy", mean)
    np.save(models_dir / "feature_std.npy", std)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest training/tests/test_train.py -v`
Expected: 3 passed

- [ ] **Step 5: Run the full test suite**

Run: `pytest -v`
Expected: all tests across Tasks 2-6 pass (19 tests total)

- [ ] **Step 6: Commit**

```bash
git add training/train.py training/tests/test_train.py
git commit -m "Add MLP training, evaluation, and ONNX export"
```

---

## Task 7: Record real data and produce the trained model (manual, requires the user)

This task cannot be delegated to an automated worker — it requires the project owner's
own voice (and ideally a second speaker) recorded through their own microphone.

**Files:**
- Populates: `training/data/raw/ruido/*.wav`, `training/data/raw/pular/*.wav`,
  `training/data/raw/abaixa/*.wav` (gitignored, not committed).
- Produces: `models/model.onnx`, `models/evaluation.txt`, `models/feature_mean.npy`,
  `models/feature_std.npy` (committed — required deliverable).

- [ ] **Step 1: Record clips for all three classes**

Run: `python -m training.record_audio`

Record at least 10-15 clips per class as agreed (`ruido`/`pular`/`abaixa`). For `ruido`,
include: silence, background noise, and a few seconds of other spoken words/phrases
(not the two commands) so the model learns to reject unrelated speech. If a second
speaker is available, have them record an additional 5-10 clips per class into the same
folders (same naming scheme — `save_clip` auto-increments the index, so re-running the
script for the second speaker is enough).

- [ ] **Step 2: Run the training pipeline**

Run: `python -m training.train`

Expected: prints a classification report + confusion matrix, then writes
`models/model.onnx`, `models/evaluation.txt`, `models/feature_mean.npy`,
`models/feature_std.npy`.

- [ ] **Step 3: Check accuracy is usable**

Read `models/evaluation.txt`. If per-class accuracy/F1 is poor (e.g. below ~0.7) for any
class, most likely causes: too few/too similar clips, or `ruido` class not covering
enough variety. Fix by recording more/varied clips for the weak class and re-running
Step 2 — no code changes needed.

- [ ] **Step 4: Commit the trained model artifacts**

```bash
git add models/model.onnx models/evaluation.txt models/feature_mean.npy models/feature_std.npy
git commit -m "Add trained voice command classifier (ONNX) and evaluation report"
```

---

## Self-Review Notes

- **Spec coverage:** feature extraction (RMS/centroid/MFCC) → Task 2; data augmentation
  for small/multi-speaker dataset → Task 3; dataset/label pipeline → Task 4; recording
  workflow → Task 5; MLP + ONNX export + evaluation → Task 6; actual model produced from
  real recordings → Task 7. Firmware (C port of the model, FreeRTOS tasks, servo
  actuation) and the RTOS diagram/report are out of scope for this plan — they are
  later phases in the spec, to be planned separately once the model is validated.
- **Type consistency:** `FEATURE_DIM` (13) defined once in `training.features` and
  reused everywhere; `CLASSES` defined once in `training.dataset` and reused in
  `train.py`; `SR` (16000) defined once in `training.features` and imported by
  `augment.py`, `dataset.py`, `record_audio.py`.
- **No placeholders:** every step has runnable code; Task 7 is the only non-automatable
  step and is explicit about why (requires the user's own voice).
