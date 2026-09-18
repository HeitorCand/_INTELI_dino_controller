// Extração de features (RMS, Zero-Crossing Rate, Spectral Centroid, MFCC),
// espelhando exatamente training/dsp.py e training/features.py em Python —
// mesma FFT de 512 pontos, 26 filtros mel, 11 coeficientes MFCC (ver
// firmware/dsp_tables.h, gerado por training/export_c_dsp.py).
#pragma once

#include "dsp_tables.h"

#define FEATURE_DIM (1 + 1 + 1 + DSP_N_MFCC)  // RMS + ZCR + centroide + MFCC

// signal deve ter exatamente DSP_CLIP_LENGTH (16000) amostras normalizadas
// em [-1, 1]. outFeatures recebe FEATURE_DIM valores, na ordem
// [RMS, ZCR, centroide, mfcc_0..mfcc_10] — mesma ordem de training/features.py.
void extractFeatures(const float *signal, float outFeatures[FEATURE_DIM]);
