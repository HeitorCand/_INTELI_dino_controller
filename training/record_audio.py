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
