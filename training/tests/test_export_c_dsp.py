import re

from training.export_c_dsp import generate_header
from training.dsp import HOP_LENGTH, N_FFT, N_MELS
from training.features import N_MFCC


def test_generate_header_has_expected_constants():
    header = generate_header()
    assert f"#define DSP_N_FFT {N_FFT}" in header
    assert f"#define DSP_N_MELS {N_MELS}" in header
    assert f"#define DSP_N_MFCC {N_MFCC}" in header
    assert f"#define DSP_HOP_LENGTH {HOP_LENGTH}" in header


def test_generate_header_array_sizes_match_expected_dimensions():
    header = generate_header()
    n_bins = N_FFT // 2 + 1

    assert f"DSP_HANN_WINDOW[{N_FFT}]" in header
    assert f"DSP_BIN_FREQS[{n_bins}]" in header
    assert f"DSP_MEL_FILTERBANK[{N_MELS * n_bins}]" in header
    assert f"DSP_DCT_MATRIX[{N_MFCC * N_MELS}]" in header


def test_generate_header_produces_valid_c_float_literals():
    header = generate_header()
    literals = re.findall(r"static const float \w+\[\d+\] = \{([^}]*)\};", header)
    assert len(literals) == 4
    for body in literals:
        for value in body.split(","):
            assert value.strip().endswith("f")
