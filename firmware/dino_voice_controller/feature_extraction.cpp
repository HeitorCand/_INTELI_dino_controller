#include "feature_extraction.h"

#include <math.h>

#include "fft.h"

// Convenção PCM de 16 bits — mesma usada pelos arquivos .wav do dataset de
// treino (training/dataset.py carrega via librosa, que aplica essa mesma
// divisão por 32768 ao ler um WAV int16).
#define PCM_SCALE (1.0f / 32768.0f)

static float computeRms(const int16_t *signal, int length) {
  double sumSquares = 0.0;
  for (int i = 0; i < length; i++) {
    float sample = signal[i] * PCM_SCALE;
    sumSquares += (double)sample * (double)sample;
  }
  return sqrtf((float)(sumSquares / length));
}

static float computeZcr(const int16_t *signal, int length) {
  int crossings = 0;
  int prevSign = signal[0] >= 0 ? 1 : -1;
  for (int i = 1; i < length; i++) {
    int sign = signal[i] >= 0 ? 1 : -1;
    if (sign != prevSign) crossings++;
    prevSign = sign;
  }
  return (float)crossings / (float)length;
}

// Amplitude de pico (já em escala float), pra normalizar o ganho antes de
// extrair centroide/MFCC — mesma lógica de training/features.py
// normalize_amplitude, mas aplicada aqui sem materializar uma cópia
// float do clipe inteiro (economiza 64KB de RAM: só guarda o fator de
// escala, aplicado depois quadro a quadro).
static float findPeakAbs(const int16_t *signal, int length) {
  int16_t peak = 0;
  for (int i = 0; i < length; i++) {
    int16_t absVal = signal[i] < 0 ? (int16_t)(-signal[i]) : signal[i];
    if (absVal > peak) peak = absVal;
  }
  return (float)peak * PCM_SCALE;
}

void extractFeatures(const int16_t *signal, float outFeatures[FEATURE_DIM]) {
  float rms = computeRms(signal, DSP_CLIP_LENGTH);
  float zcr = computeZcr(signal, DSP_CLIP_LENGTH);

  float peak = findPeakAbs(signal, DSP_CLIP_LENGTH);
  const float targetPeak = 0.5f;
  float gainScale = (peak < 1e-6f) ? 1.0f : (targetPeak / peak);

  float centroidSum = 0.0f;
  float melLogSum[DSP_N_MELS];
  for (int m = 0; m < DSP_N_MELS; m++) melLogSum[m] = 0.0f;

  float windowed[DSP_N_FFT];
  float magnitudes[DSP_N_BINS];

  for (int frame = 0; frame < DSP_N_FRAMES; frame++) {
    int start = frame * DSP_HOP_LENGTH;
    for (int i = 0; i < DSP_N_FFT; i++) {
      float sample = (float)signal[start + i] * PCM_SCALE * gainScale;
      windowed[i] = sample * DSP_HANN_WINDOW[i];
    }
    fftComputeMagnitudes(windowed, DSP_N_FFT, magnitudes);

    float num = 0.0f, den = 0.0f;
    for (int k = 0; k < DSP_N_BINS; k++) {
      num += magnitudes[k] * DSP_BIN_FREQS[k];
      den += magnitudes[k];
    }
    centroidSum += num / (den + 1e-10f);

    for (int m = 0; m < DSP_N_MELS; m++) {
      float energy = 0.0f;
      const float *filterRow = &DSP_MEL_FILTERBANK[m * DSP_N_BINS];
      for (int k = 0; k < DSP_N_BINS; k++) {
        energy += filterRow[k] * magnitudes[k];
      }
      melLogSum[m] += logf(energy + 1e-6f);
    }
  }

  float centroidMean = centroidSum / (float)DSP_N_FRAMES;
  float melLogMean[DSP_N_MELS];
  for (int m = 0; m < DSP_N_MELS; m++) {
    melLogMean[m] = melLogSum[m] / (float)DSP_N_FRAMES;
  }

  // DCT-II aplicada na média dos log-mel (equivalente a aplicar por frame e
  // depois tirar a média, já que a DCT é uma transformação linear).
  outFeatures[0] = rms;
  outFeatures[1] = zcr;
  outFeatures[2] = centroidMean;
  for (int c = 0; c < DSP_N_MFCC; c++) {
    float sum = 0.0f;
    const float *dctRow = &DSP_DCT_MATRIX[c * DSP_N_MELS];
    for (int m = 0; m < DSP_N_MELS; m++) {
      sum += dctRow[m] * melLogMean[m];
    }
    outFeatures[3 + c] = sum;
  }
}
