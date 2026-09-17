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
