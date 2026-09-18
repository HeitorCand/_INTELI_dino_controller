import sys
import time
from pathlib import Path

import numpy as np

from training.dataset import CLASSES
from training.features import SR
from training.record_audio import save_clip

CLIP_LENGTH = 16000
MAGIC = b"REC1"
TRIGGER_BYTE = b"r"


def parse_clip_response(header: bytes, raw_bytes: bytes) -> np.ndarray:
    """Converts the ESP32's raw reply (magic header + int16 PCM bytes) into a
    float32 signal in the same [-1, 1]-ish range used throughout training."""
    if header != MAGIC:
        raise ValueError(f"cabeçalho inesperado do ESP32: {header!r}")
    expected_bytes = CLIP_LENGTH * 2
    if len(raw_bytes) != expected_bytes:
        raise ValueError(f"tamanho inesperado: {len(raw_bytes)} bytes (esperado {expected_bytes})")

    int16_samples = np.frombuffer(raw_bytes, dtype="<i2")
    return (int16_samples.astype(np.float32) / 32768.0).astype(np.float32)


def capture_one_clip(ser) -> np.ndarray:
    ser.reset_input_buffer()
    ser.write(TRIGGER_BYTE)
    header = ser.read(4)
    raw_bytes = ser.read(CLIP_LENGTH * 2)
    return parse_clip_response(header, raw_bytes)


def main():
    if len(sys.argv) < 2:
        print("uso: python -m training.record_from_esp32 <porta_serial>")
        print("  ex: python -m training.record_from_esp32 /dev/cu.usbserial-0001")
        sys.exit(1)

    import serial  # import tardio: só é necessário pra rodar de fato, não pros testes

    port = sys.argv[1]
    ser = serial.Serial(port, 115200, timeout=5)
    time.sleep(2)  # espera o ESP32 reiniciar após abrir a porta serial
    ser.reset_input_buffer()

    raw_dir = Path(__file__).parent / "data" / "raw"
    for class_name in CLASSES:
        class_dir = raw_dir / class_name
        print(f"\n=== Classe '{class_name}' (gravando pelo ESP32) ===")
        while True:
            response = input(
                f"  Enter para gravar 1s de '{class_name}' "
                "(ou 'n' + Enter para próxima classe): "
            )
            if response.strip().lower() == "n":
                break
            signal = capture_one_clip(ser)
            path = save_clip(class_dir, signal, sr=SR)
            rms = float(np.sqrt(np.mean(np.square(signal))))
            print(f"  Salvo: {path} (rms={rms:.4f})")

    ser.close()


if __name__ == "__main__":
    main()
