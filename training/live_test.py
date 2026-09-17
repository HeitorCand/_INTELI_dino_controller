from collections import deque
from pathlib import Path
from queue import Queue
from typing import Tuple

import numpy as np
import onnxruntime as ort
import sounddevice as sd

from training.augment import fix_length
from training.dataset import CLASSES
from training.features import SR, extract_features

CHUNK_DURATION = 0.1
CHUNK_SAMPLES = int(SR * CHUNK_DURATION)
CLIP_DURATION = 1.0
CLIP_LENGTH = int(SR * CLIP_DURATION)
WINDOW_CHUNKS = int(CLIP_DURATION / CHUNK_DURATION)
CONFIDENCE_THRESHOLD = 0.6
CALIBRATION_DURATION = 1.0
MIN_RMS_MULTIPLIER = 3.0
MIN_RMS_FLOOR = 0.005


def compute_rms(signal: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(signal))))


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / exp.sum()


def load_classifier(models_dir: Path) -> Tuple[ort.InferenceSession, np.ndarray, np.ndarray]:
    session = ort.InferenceSession(str(models_dir / "model.onnx"))
    mean = np.load(models_dir / "feature_mean.npy")
    std = np.load(models_dir / "feature_std.npy")
    return session, mean, std


def classify_clip(
    signal: np.ndarray, session: ort.InferenceSession, mean: np.ndarray, std: np.ndarray
) -> Tuple[str, np.ndarray]:
    clip = fix_length(signal.astype(np.float32), CLIP_LENGTH)
    features = extract_features(clip, sr=SR)
    normalized = ((features - mean) / std).astype(np.float32).reshape(1, -1)
    logits = session.run(None, {"features": normalized})[0][0]
    probabilities = softmax(logits)
    predicted_class = CLASSES[int(np.argmax(probabilities))]
    return predicted_class, probabilities


def decide_class(probabilities: np.ndarray) -> str:
    """Rejects to 'ruido' when the model isn't confident, instead of always taking
    the argmax even when it's a close, unconvincing call."""
    if float(np.max(probabilities)) < CONFIDENCE_THRESHOLD:
        return "ruido"
    return CLASSES[int(np.argmax(probabilities))]


def format_probabilities(probabilities: np.ndarray) -> str:
    return ", ".join(f"{name}: {p * 100:.0f}%" for name, p in zip(CLASSES, probabilities))


def calibrate_min_rms(duration: float = CALIBRATION_DURATION) -> float:
    """Measures the room's ambient noise floor so near-silent windows can be skipped
    without even running the model. True silence/quiet ambient noise is out of the
    training distribution (every recording has *some* real sound in it) and the model
    can extrapolate confidently wrong on it - this floor is a cheap safety net,
    independent of the model, that catches that case directly."""
    print(f"Calibrando ruído de fundo — fique em silêncio por {duration:.1f}s...")
    samples = int(SR * duration)
    recording = sd.rec(samples, samplerate=SR, channels=1, dtype="float32")
    sd.wait()
    ambient_rms = compute_rms(recording.flatten())
    min_rms = max(ambient_rms * MIN_RMS_MULTIPLIER, MIN_RMS_FLOOR)
    print(f"Ruído ambiente: {ambient_rms:.4f} | Piso mínimo para classificar: {min_rms:.4f}\n")
    return min_rms


def main():
    models_dir = Path(__file__).parent.parent / "models"
    session, mean, std = load_classifier(models_dir)
    min_rms = calibrate_min_rms()

    audio_queue: "Queue[np.ndarray]" = Queue()

    def callback(indata, frames, time_info, status):
        audio_queue.put(indata[:, 0].copy())

    print("Ouvindo continuamente... fale 'pular' ou 'abaixa' (Ctrl+C para sair)\n")

    window: "deque[np.ndarray]" = deque(maxlen=WINDOW_CHUNKS)

    with sd.InputStream(
        samplerate=SR, channels=1, blocksize=CHUNK_SAMPLES, callback=callback, dtype="float32"
    ):
        try:
            while True:
                chunk = audio_queue.get()
                window.append(chunk)
                if len(window) < WINDOW_CHUNKS:
                    continue

                signal = np.concatenate(window)
                if compute_rms(signal) < min_rms:
                    continue

                _, probabilities = classify_clip(signal, session, mean, std)
                decided_class = decide_class(probabilities)

                if decided_class != "ruido":
                    print(f">> {decided_class.upper():8s} ({format_probabilities(probabilities)})")
                    # Debounce: drop the window so the same utterance doesn't fire
                    # again on the next slide; classification pauses until it
                    # refills with a full second of fresh audio.
                    window.clear()
        except KeyboardInterrupt:
            print("\nEncerrando.")


if __name__ == "__main__":
    main()
