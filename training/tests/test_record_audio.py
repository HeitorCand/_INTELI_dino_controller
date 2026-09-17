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


def test_next_clip_index_does_not_reuse_index_after_gap(tmp_path):
    # Reproduces a data-destroying bug: if a bad take (clip_001.wav) is deleted
    # while clip_000.wav and clip_002.wav remain, the next recording must not be
    # saved as clip_002.wav (which would silently overwrite the existing file).
    class_dir = tmp_path / "pular"
    class_dir.mkdir()
    (class_dir / "clip_000.wav").touch()
    (class_dir / "clip_001.wav").touch()
    (class_dir / "clip_002.wav").touch()
    (class_dir / "clip_001.wav").unlink()

    assert next_clip_index(class_dir) == 3


def test_save_clip_writes_wav_file_and_increments_index(tmp_path):
    class_dir = tmp_path / "abaixa"
    signal = np.zeros(16000, dtype=np.float32)

    first_path = save_clip(class_dir, signal, sr=16000)
    second_path = save_clip(class_dir, signal, sr=16000)

    assert first_path.name == "clip_000.wav"
    assert second_path.name == "clip_001.wav"
    assert first_path.exists()
    assert second_path.exists()
