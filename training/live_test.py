from collections import deque
from pathlib import Path
from queue import Queue
from typing import List, Tuple

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
MAX_CAPTURE_DURATION = 1.5
HANGOVER_DURATION = 0.3
PRE_ROLL_DURATION = 0.2
PRE_ROLL_CHUNKS = int(PRE_ROLL_DURATION / CHUNK_DURATION)
CALIBRATION_DURATION = 1.5
THRESHOLD_MULTIPLIER = 2.5
MIN_THRESHOLD = 0.005


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


def calibrate_threshold(duration: float = CALIBRATION_DURATION) -> float:
    print(f"Calibrando ruído de fundo — fique em silêncio por {duration:.1f}s...")
    samples = int(SR * duration)
    recording = sd.rec(samples, samplerate=SR, channels=1, dtype="float32")
    sd.wait()
    ambient_rms = compute_rms(recording.flatten())
    threshold = max(ambient_rms * THRESHOLD_MULTIPLIER, MIN_THRESHOLD)
    print(f"Ruído ambiente: {ambient_rms:.4f} | Limiar de fala: {threshold:.4f}")
    return threshold


def format_probabilities(probabilities: np.ndarray) -> str:
    return ", ".join(f"{name}: {p * 100:.0f}%" for name, p in zip(CLASSES, probabilities))


def main():
    models_dir = Path(__file__).parent.parent / "models"
    session, mean, std = load_classifier(models_dir)
    threshold = calibrate_threshold()

    audio_queue: "Queue[np.ndarray]" = Queue()

    def callback(indata, frames, time_info, status):
        audio_queue.put(indata[:, 0].copy())

    print("\nOuvindo... fale 'pular' ou 'abaixa' (Ctrl+C para sair)\n")

    capturing = False
    capture_chunks: List[np.ndarray] = []
    capture_duration = 0.0
    silence_duration = 0.0
    pre_roll: "deque[np.ndarray]" = deque(maxlen=PRE_ROLL_CHUNKS)

    with sd.InputStream(
        samplerate=SR, channels=1, blocksize=CHUNK_SAMPLES, callback=callback, dtype="float32"
    ):
        try:
            while True:
                chunk = audio_queue.get()
                chunk_rms = compute_rms(chunk)

                if not capturing:
                    if chunk_rms > threshold:
                        capturing = True
                        capture_chunks = list(pre_roll) + [chunk]
                        capture_duration = len(capture_chunks) * CHUNK_DURATION
                        silence_duration = 0.0
                    else:
                        pre_roll.append(chunk)
                    continue

                capture_chunks.append(chunk)
                capture_duration += CHUNK_DURATION
                if chunk_rms > threshold:
                    silence_duration = 0.0
                else:
                    silence_duration += CHUNK_DURATION

                if silence_duration >= HANGOVER_DURATION or capture_duration >= MAX_CAPTURE_DURATION:
                    signal = np.concatenate(capture_chunks)
                    predicted_class, probabilities = classify_clip(signal, session, mean, std)
                    print(f">> {predicted_class.upper():8s} ({format_probabilities(probabilities)})")

                    capturing = False
                    capture_chunks = []
                    pre_roll.clear()
        except KeyboardInterrupt:
            print("\nEncerrando.")


if __name__ == "__main__":
    main()
