"""Script de teste do detector: simula a chegada de anomalias acusticas (os
clipes de teste, nunca vistos no treino) pelo pipeline de deteccao e mede
performance — latencia de cada etapa e acuracia da decisao final.

Roda a mesma logica de decisao usada no firmware (extracao de features,
inferencia do modelo, piso minimo de RMS e limiar de confianca), fora do
fluxo de treino, para servir como o script de teste dedicado pedido pela
ponderada.

Uso: python -m training.test_detector
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import onnxruntime as ort
from sklearn.metrics import classification_report, confusion_matrix

from training.dataset import CLASSES, load_raw_clips, split_raw_clips
from training.features import SR, extract_features
from training.live_test import load_classifier, softmax

# Mesmos valores usados em firmware/dino_voice_controller/dino_voice_controller.ino
# (CONFIDENCE_THRESHOLD e MIN_RMS_FLOOR) — manter em sincronia se o firmware mudar.
CONFIDENCE_THRESHOLD = 0.75
MIN_RMS_FLOOR = 0.07

ANOMALY_CLASSES = {"pular", "abaixa"}


@dataclass
class DetectionResult:
    true_class: str
    predicted_class: str
    confidence: float
    rejected_by_rms_floor: bool
    feature_extraction_ms: float
    inference_ms: float


def simulate_detection(
    signal: np.ndarray,
    true_class: str,
    session: Optional[ort.InferenceSession],
    mean: np.ndarray,
    std: np.ndarray,
) -> DetectionResult:
    """Simulates one anomaly event arriving at the detector, timing feature
    extraction and inference separately and applying the same RMS-floor +
    confidence-threshold decision the firmware applies on-device."""
    t0 = time.perf_counter()
    features = extract_features(signal, sr=SR)
    t1 = time.perf_counter()
    feature_extraction_ms = (t1 - t0) * 1000

    rms = float(features[0])
    if rms < MIN_RMS_FLOOR:
        return DetectionResult(
            true_class=true_class,
            predicted_class="ruido",
            confidence=0.0,
            rejected_by_rms_floor=True,
            feature_extraction_ms=feature_extraction_ms,
            inference_ms=0.0,
        )

    normalized = ((features - mean) / std).astype(np.float32).reshape(1, -1)
    t2 = time.perf_counter()
    logits = session.run(None, {"features": normalized})[0][0]
    t3 = time.perf_counter()
    inference_ms = (t3 - t2) * 1000

    probabilities = softmax(logits)
    confidence = float(np.max(probabilities))
    if confidence < CONFIDENCE_THRESHOLD:
        predicted_class = "ruido"
    else:
        predicted_class = CLASSES[int(np.argmax(probabilities))]

    return DetectionResult(
        true_class=true_class,
        predicted_class=predicted_class,
        confidence=confidence,
        rejected_by_rms_floor=False,
        feature_extraction_ms=feature_extraction_ms,
        inference_ms=inference_ms,
    )


def _latency_stats_line(label: str, values: np.ndarray) -> str:
    return (
        f"  {label}: media={values.mean():.2f}  mediana={np.median(values):.2f}  "
        f"p95={np.percentile(values, 95):.2f}  max={values.max():.2f}"
    )


def format_report(results: List[DetectionResult]) -> str:
    lines = [
        f"Simulacao de deteccao de anomalias — {len(results)} clipes de teste "
        "(dados reais do INMP441, held-out)",
        "",
        "Latencia por etapa (ms):",
        _latency_stats_line(
            "extracao de features", np.array([r.feature_extraction_ms for r in results])
        ),
    ]

    inference_ms = np.array([r.inference_ms for r in results if not r.rejected_by_rms_floor])
    if inference_ms.size:
        lines.append(_latency_stats_line("inferencia do modelo", inference_ms))
    lines.append("")

    y_true = [r.true_class for r in results]
    y_pred = [r.predicted_class for r in results]
    lines.append("Relatorio de classificacao (decisao final, apos piso de RMS e limiar de confianca):")
    lines.append(classification_report(y_true, y_pred, labels=CLASSES, zero_division=0))
    matrix = confusion_matrix(y_true, y_pred, labels=CLASSES)
    lines.append(f"Matriz de confusao (linhas=real, colunas=previsto, ordem {CLASSES}):")
    lines.append(str(matrix))
    lines.append("")

    anomaly_total = sum(1 for r in results if r.true_class in ANOMALY_CLASSES)
    anomaly_detected = sum(
        1 for r in results if r.true_class in ANOMALY_CLASSES and r.predicted_class == r.true_class
    )
    ruido_total = sum(1 for r in results if r.true_class == "ruido")
    false_positives = sum(
        1 for r in results if r.true_class == "ruido" and r.predicted_class in ANOMALY_CLASSES
    )
    rms_rejected = sum(1 for r in results if r.rejected_by_rms_floor)

    lines.append("Metricas de deteccao de anomalia:")
    if anomaly_total:
        lines.append(
            f"  taxa de deteccao (pular+abaixa corretos / total de anomalias): "
            f"{anomaly_detected}/{anomaly_total} ({100 * anomaly_detected / anomaly_total:.1f}%)"
        )
    if ruido_total:
        lines.append(
            f"  taxa de falso positivo (ruido classificado como comando): "
            f"{false_positives}/{ruido_total} ({100 * false_positives / ruido_total:.1f}%)"
        )
    lines.append(
        f"  clipes rejeitados pelo piso de RMS (MIN_RMS_FLOOR={MIN_RMS_FLOOR}): "
        f"{rms_rejected}/{len(results)}"
    )

    return "\n".join(lines)


def run_test(raw_dir: Path, models_dir: Path) -> Tuple[List[DetectionResult], str]:
    signals, labels = load_raw_clips(raw_dir)
    _, _, test_signals, test_labels = split_raw_clips(signals, labels)

    session, mean, std = load_classifier(models_dir)

    results = [
        simulate_detection(signal, CLASSES[label], session, mean, std)
        for signal, label in zip(test_signals, test_labels)
    ]
    return results, format_report(results)


def main():
    repo_root = Path(__file__).parent.parent
    raw_dir = repo_root / "training" / "data" / "raw"
    models_dir = repo_root / "models"

    _, report = run_test(raw_dir, models_dir)
    print(report)
    (models_dir / "test_report.txt").write_text(report)
    print(f"\nEscrito: {models_dir / 'test_report.txt'}")


if __name__ == "__main__":
    main()
