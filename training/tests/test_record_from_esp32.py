import numpy as np
import pytest

from training.record_from_esp32 import CLIP_LENGTH, MAGIC, parse_clip_response, read_until_magic


class FakeSerial:
    """Simula o .read(1) do pyserial sobre um buffer de bytes fixo."""

    def __init__(self, data: bytes):
        self._data = data
        self._pos = 0

    def read(self, n: int = 1) -> bytes:
        chunk = self._data[self._pos : self._pos + n]
        self._pos += len(chunk)
        return chunk


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


def test_read_until_magic_skips_leading_boot_garbage():
    fake = FakeSerial(b"\x00\xff\x00\xff\x01\x02" + MAGIC + b"resto")
    read_until_magic(fake)  # não deve levantar exceção
    assert fake.read(5) == b"resto"


def test_read_until_magic_finds_marker_at_the_very_start():
    fake = FakeSerial(MAGIC + b"resto")
    read_until_magic(fake)
    assert fake.read(5) == b"resto"


def test_read_until_magic_raises_timeout_when_stream_ends():
    fake = FakeSerial(b"\x00\x01\x02")  # nunca aparece o marcador, e acaba os dados
    with pytest.raises(TimeoutError):
        read_until_magic(fake)


def test_read_until_magic_raises_value_error_when_scan_limit_exceeded():
    fake = FakeSerial(b"\x00" * 100)
    with pytest.raises(ValueError, match="não encontrado"):
        read_until_magic(fake, max_scan_bytes=10)
