import numpy as np
import pytest

from training.record_from_esp32 import CLIP_LENGTH, MAGIC, parse_clip_response


def _make_raw_bytes(int16_values):
    return np.array(int16_values, dtype="<i2").tobytes()


def test_parse_clip_response_converts_int16_to_float_range():
    raw = _make_raw_bytes([32767, -32768, 0] + [0] * (CLIP_LENGTH - 3))
    signal = parse_clip_response(MAGIC, raw)

    assert signal.shape == (CLIP_LENGTH,)
    assert signal.dtype == np.float32
    assert abs(signal[0] - 1.0) < 1e-3
    assert abs(signal[1] - (-1.0)) < 1e-3
    assert signal[2] == 0.0


def test_parse_clip_response_rejects_wrong_header():
    raw = _make_raw_bytes([0] * CLIP_LENGTH)
    with pytest.raises(ValueError, match="cabeçalho"):
        parse_clip_response(b"XXXX", raw)


def test_parse_clip_response_rejects_wrong_length():
    raw = _make_raw_bytes([0] * (CLIP_LENGTH - 1))  # um a menos
    with pytest.raises(ValueError, match="tamanho"):
        parse_clip_response(MAGIC, raw)
