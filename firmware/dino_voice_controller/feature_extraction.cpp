#include "feature_extraction.h"

#include <math.h>

#include "fft.h"

static float computeRms(const float *signal, int length) {
  double sumSquares = 0.0;
  for (int i = 0; i < length; i++) {
    sumSquares += (double)signal[i] * (double)signal[i];
  }
  return sqrtf((float)(sumSquares / length));
}

static float computeZcr(const float *signal, int length) {
  int crossings = 0;
  int prevSign = signal[0] >= 0.0f ? 1 : -1;
  for (int i = 1; i < length; i++) {
    int sign = signal[i] >= 0.0f ? 1 : -1;
    if (sign != prevSign) crossings++;
    prevSign = sign;
  }
  return (float)crossings / (float)length;
}

// Normaliza o pico de amplitude para targetPeak, preservando a forma
// espectral independente do ganho do dispositivo de gravação (mesma lógica
// de training/features.py normalize_amplitude).
static void normalizeAmplitude(const float *signal, int length, float targetPeak, float *out) {
  float peak = 0.0f;
  for (int i = 0; i < length; i++) {
    float absVal = fabsf(signal[i]);
    if (absVal > peak) peak = absVal;
  }
  if (peak < 1e-6f) {
    for (int i = 0; i < length; i++) out[i] = signal[i];
    return;
  }
  float scale = targetPeak / peak;
  for (int i = 0; i < length; i++) out[i] = signal[i] * scale;
}

void extractFeatures(const float *signal, float outFeatures[FEATURE_DIM]) {
  float rms = computeRms(signal, DSP_CLIP_LENGTH);
  float zcr = computeZcr(signal, DSP_CLIP_LENGTH);

  // Buffer estático (não na pilha): 16000 floats = 64KB seriam grandes demais
  // para a stack de uma task FreeRTOS. Assume uma única task chamando esta
  // função por vez (arquitetura: task de detecção é a única consumidora).
  static float shapeSignal[DSP_CLIP_LENGTH];
  normalizeAmplitude(signal, DSP_CLIP_LENGTH, 0.5f, shapeSignal);

  float centroidSum = 0.0f;
  float melLogSum[DSP_N_MELS];
  for (int m = 0; m < DSP_N_MELS; m++) melLogSum[m] = 0.0f;

  float windowed[DSP_N_FFT];
  float magnitudes[DSP_N_BINS];

  for (int frame = 0; frame < DSP_N_FRAMES; frame++) {
    int start = frame * DSP_HOP_LENGTH;
    for (int i = 0; i < DSP_N_FFT; i++) {
      windowed[i] = shapeSignal[start + i] * DSP_HANN_WINDOW[i];
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
