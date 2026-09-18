#include "fft.h"

#include <math.h>

#define FFT_MAX_N 1024

static void fftRadix2InPlace(float *real, float *imag, int n) {
  // Permutação bit-reversal.
  int j = 0;
  for (int i = 0; i < n - 1; i++) {
    if (i < j) {
      float tr = real[i];
      real[i] = real[j];
      real[j] = tr;
      float ti = imag[i];
      imag[i] = imag[j];
      imag[j] = ti;
    }
    int m = n >> 1;
    while (m >= 1 && j >= m) {
      j -= m;
      m >>= 1;
    }
    j += m;
  }

  // Estágios de combinação (butterfly).
  for (int len = 2; len <= n; len <<= 1) {
    float angleStep = -2.0f * (float)M_PI / (float)len;
    float wr = cosf(angleStep);
    float wi = sinf(angleStep);
    for (int i = 0; i < n; i += len) {
      float curWr = 1.0f, curWi = 0.0f;
      for (int k = 0; k < len / 2; k++) {
        int evenIdx = i + k;
        int oddIdx = i + k + len / 2;
        float evenR = real[evenIdx], evenI = imag[evenIdx];
        float oddR = real[oddIdx], oddI = imag[oddIdx];

        float tR = oddR * curWr - oddI * curWi;
        float tI = oddR * curWi + oddI * curWr;

        real[evenIdx] = evenR + tR;
        imag[evenIdx] = evenI + tI;
        real[oddIdx] = evenR - tR;
        imag[oddIdx] = evenI - tI;

        float nextWr = curWr * wr - curWi * wi;
        float nextWi = curWr * wi + curWi * wr;
        curWr = nextWr;
        curWi = nextWi;
      }
    }
  }
}

void fftComputeMagnitudes(const float *windowedSamples, int n, float *outMagnitudes) {
  static float real[FFT_MAX_N];
  static float imag[FFT_MAX_N];

  for (int i = 0; i < n; i++) {
    real[i] = windowedSamples[i];
    imag[i] = 0.0f;
  }

  fftRadix2InPlace(real, imag, n);

  int nBins = n / 2 + 1;
  for (int k = 0; k < nBins; k++) {
    outMagnitudes[k] = sqrtf(real[k] * real[k] + imag[k] * imag[k]);
  }
}
