import numpy as np
import soundfile as sf

from training.features import SR, FEATURE_DIM
from training.test_detector import (
    MIN_RMS_FLOOR,
    DetectionResult,
    format_report,
    run_test,
    simulate_detection,
)
from training.train import export_onnx, run_pipeline


def _write_tone(path, freq, duration=1.0):
    t = np.linspace(0, duration, int(SR * duration), endpoint=False)
    signal = (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    sf.write(str(path), signal, SR)


def _make_raw_dir(tmp_path, clips_per_class=6):
    raw_dir = tmp_path / "raw"
    freqs = {"ruido": 200.0, "pular": 440.0, "abaixa": 880.0}
    for class_name, freq in freqs.items():
        class_dir = raw_dir / class_name
        class_dir.mkdir(parents=True)
        for i in range(clips_per_class):
            _write_tone(class_dir / f"clip_{i}.wav", freq)
    return raw_dir


def test_simulate_detection_rejects_silence_below_rms_floor_without_calling_model():
    silence = np.zeros(SR, dtype=np.float32)
    mean = np.zeros(FEATURE_DIM, dtype=np.float32)
    std = np.ones(FEATURE_DIM, dtype=np.float32)

    result = simulate_detection(silence, "ruido", session=None, mean=mean, std=std)

    assert result.rejected_by_rms_floor is True
    assert result.predicted_class == "ruido"
    assert result.inference_ms == 0.0
    assert result.feature_extraction_ms >= 0.0


def test_format_report_computes_detection_rate_and_false_positive_rate():
    results = [
        DetectionResult("pular", "pular", 0.9, False, 1.0, 0.1),
        DetectionResult("pular", "pular", 0.9, False, 1.0, 0.1),
        DetectionResult("pular", "ruido", 0.4, False, 1.0, 0.1),
        DetectionResult("abaixa", "abaixa", 0.9, False, 1.0, 0.1),
        DetectionResult("ruido", "pular", 0.9, False, 1.0, 0.1),
        DetectionResult("ruido", "ruido", 0.9, False, 1.0, 0.1),
        DetectionResult("ruido", "ruido", 0.0, True, 1.0, 0.0),
    ]

    report = format_report(results)

    assert "3/4 (75.0%)" in report  # taxa de deteccao: 3 anomalias corretas em 4
    assert "1/3 (33.3%)" in report  # taxa de falso positivo: 1 ruido virou comando em 3
    assert f"MIN_RMS_FLOOR={MIN_RMS_FLOOR}" in report
    assert "1/7" in report  # um clipe rejeitado pelo piso de RMS


def test_run_test_end_to_end_on_synthetic_dataset(tmp_path):
    raw_dir = _make_raw_dir(tmp_path)
    model, _, mean, std = run_pipeline(raw_dir)

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    export_onnx(model, models_dir / "model.onnx")
    np.save(models_dir / "feature_mean.npy", mean)
    np.save(models_dir / "feature_std.npy", std)

    results, report = run_test(raw_dir, models_dir)

    assert len(results) > 0
    assert all(r.true_class in {"ruido", "pular", "abaixa"} for r in results)
    assert all(r.feature_extraction_ms >= 0.0 for r in results)
    assert "Matriz de confusao" in report
    assert "taxa de deteccao" in report
