import numpy as np
import soundfile as sf

from training.dataset import build_features, load_raw_clips, split_raw_clips
from training.features import SR, FEATURE_DIM
from training.train import evaluate_model, normalize_features, run_pipeline


def _write_tone(path, freq, duration=1.0):
    t = np.linspace(0, duration, int(SR * duration), endpoint=False)
    signal = (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    sf.write(str(path), signal, SR)


def _make_raw_dir(tmp_path, clips_per_class=3):
    raw_dir = tmp_path / "raw"
    freqs = {"ruido": 200.0, "pular": 440.0, "abaixa": 880.0}
    for class_name, freq in freqs.items():
        class_dir = raw_dir / class_name
        class_dir.mkdir(parents=True)
        for i in range(clips_per_class):
            _write_tone(class_dir / f"clip_{i}.wav", freq)
    return raw_dir


def test_load_raw_clips_returns_one_signal_per_clip(tmp_path):
    raw_dir = _make_raw_dir(tmp_path)
    signals, labels = load_raw_clips(raw_dir)
    assert len(signals) == 9
    assert set(labels.tolist()) == {0, 1, 2}


def test_split_raw_clips_has_no_overlap_between_train_and_test(tmp_path):
    raw_dir = _make_raw_dir(tmp_path)
    signals, labels = load_raw_clips(raw_dir)
    train_signals, train_labels, test_signals, test_labels = split_raw_clips(
        signals, labels, test_size=0.25
    )

    assert len(train_signals) + len(test_signals) == 9
    assert len(test_labels) > 0

    # split_raw_clips must partition the original signals, not copy/regenerate them,
    # so identity is a reliable (and here, more reliable than value equality — the
    # fixture's clips within a class are tonally identical) way to check no overlap.
    train_ids = {id(signal) for signal in train_signals}
    test_ids = {id(signal) for signal in test_signals}
    assert train_ids.isdisjoint(test_ids)


def test_build_features_without_augment_has_one_row_per_signal(tmp_path):
    raw_dir = _make_raw_dir(tmp_path)
    signals, labels = load_raw_clips(raw_dir)
    X, y = build_features(signals, labels, augment=False)
    assert X.shape == (len(signals), FEATURE_DIM)
    assert y.shape == (len(signals),)


def test_build_features_with_augment_multiplies_rows_by_four(tmp_path):
    raw_dir = _make_raw_dir(tmp_path)
    signals, labels = load_raw_clips(raw_dir)
    X, _ = build_features(signals, labels, augment=True)
    assert X.shape[0] == len(signals) * 4


def test_augmentation_does_not_leak_into_test_partition(tmp_path):
    raw_dir = _make_raw_dir(tmp_path)
    signals, labels = load_raw_clips(raw_dir)
    train_signals, train_labels, test_signals, test_labels = split_raw_clips(
        signals, labels, test_size=0.25
    )

    X_train, y_train = build_features(train_signals, train_labels, augment=True)
    X_test, y_test = build_features(test_signals, test_labels, augment=False)

    assert X_train.shape[0] + X_test.shape[0] == len(train_signals) * 4 + len(test_signals)
    assert X_test.shape[0] == len(test_signals)


def test_run_pipeline_evaluates_on_unaugmented_test_partition(tmp_path):
    # Regression test for the augment-before-split bug: if run_pipeline ever went
    # back to augmenting before splitting (or augmenting the test partition), the
    # evaluation report it produces would be computed over a different (larger,
    # augmented) test set than the one reconstructed here from split_raw_clips +
    # build_features(..., augment=False), and the reports below would diverge.
    raw_dir = _make_raw_dir(tmp_path, clips_per_class=5)

    model, report, mean, std = run_pipeline(raw_dir)

    signals, labels = load_raw_clips(raw_dir)
    _, _, test_signals, test_labels = split_raw_clips(signals, labels)
    X_test, y_test = build_features(test_signals, test_labels, augment=False)
    X_test_norm = normalize_features(X_test, mean, std)
    expected_report = evaluate_model(model, X_test_norm, y_test)

    assert report == expected_report
