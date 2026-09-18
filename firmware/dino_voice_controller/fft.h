// FFT radix-2 (Cooley-Tukey, in-place, iterativa) para uso embarcado.
// Requer que n seja potência de 2 (usamos n = DSP_N_FFT = 512, ver dsp_tables.h).
#pragma once

// Calcula a magnitude do espectro de um sinal já janelado (Hann aplicada
// pelo chamador). windowedSamples tem n amostras; outMagnitudes recebe
// n/2+1 valores (espectro real, bins de 0 a Nyquist).
void fftComputeMagnitudes(const float *windowedSamples, int n, float *outMagnitudes);
