from pathlib import Path

import numpy as np

from training.dsp import HOP_LENGTH, MEL_FILTERBANK, N_FFT, N_MELS, build_dct_matrix, hann_window
from training.features import N_MFCC, SR


def format_c_array(name: str, values: np.ndarray) -> str:
    flat = values.astype(np.float32).flatten()
    body = ", ".join(f"{v:.8f}f" for v in flat)
    return f"static const float {name}[{flat.size}] = {{{body}}};"


def generate_header() -> str:
    window = hann_window(N_FFT)
    dct_matrix = build_dct_matrix(N_MELS, N_MFCC)
    n_bins = N_FFT // 2 + 1
    bin_freqs = np.fft.rfftfreq(N_FFT, d=1.0 / SR).astype(np.float32)
    n_frames = 1 + (16000 - N_FFT) // HOP_LENGTH

    lines = [
        "// Gerado automaticamente por training/export_c_dsp.py — NAO EDITAR A MAO.",
        "// Tabelas do pipeline de DSP (janela Hann, filtros mel, matriz DCT) —",
        "// devem ficar em sincronia com training/dsp.py.",
        "#pragma once",
        "",
        f"#define DSP_N_FFT {N_FFT}",
        f"#define DSP_N_BINS {n_bins}",
        f"#define DSP_HOP_LENGTH {HOP_LENGTH}",
        f"#define DSP_N_MELS {N_MELS}",
        f"#define DSP_N_MFCC {N_MFCC}",
        f"#define DSP_N_FRAMES {n_frames}",
        f"#define DSP_CLIP_LENGTH 16000",
        "",
        format_c_array("DSP_HANN_WINDOW", window),
        format_c_array("DSP_BIN_FREQS", bin_freqs),
        format_c_array("DSP_MEL_FILTERBANK", MEL_FILTERBANK),  # [N_MELS][N_BINS] linha-major
        format_c_array("DSP_DCT_MATRIX", dct_matrix),  # [N_MFCC][N_MELS] linha-major
        "",
    ]
    return "\n".join(lines)


def main():
    repo_root = Path(__file__).parent.parent
    out_path = repo_root / "firmware" / "dsp_tables.h"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(generate_header())
    print(f"Escrito: {out_path}")


if __name__ == "__main__":
    main()
