#include "model_inference.h"

#include <math.h>

void modelNormalizeFeatures(const float rawFeatures[MODEL_INPUT_DIM],
                             float normalized[MODEL_INPUT_DIM]) {
  for (int i = 0; i < MODEL_INPUT_DIM; i++) {
    normalized[i] = (rawFeatures[i] - MODEL_FEATURE_MEAN[i]) / MODEL_FEATURE_STD[i];
  }
}

void modelForward(const float normalizedFeatures[MODEL_INPUT_DIM],
                   float outLogits[MODEL_OUTPUT_DIM]) {
  float hidden[MODEL_HIDDEN_DIM];

  // Camada 1: Linear(14 -> 16) + ReLU. MODEL_W1 é [hidden][input] linha-major.
  for (int h = 0; h < MODEL_HIDDEN_DIM; h++) {
    float sum = MODEL_B1[h];
    for (int i = 0; i < MODEL_INPUT_DIM; i++) {
      sum += MODEL_W1[h * MODEL_INPUT_DIM + i] * normalizedFeatures[i];
    }
    hidden[h] = sum > 0.0f ? sum : 0.0f;  // ReLU
  }

  // Camada 2: Linear(16 -> 3). MODEL_W2 é [output][hidden] linha-major.
  for (int o = 0; o < MODEL_OUTPUT_DIM; o++) {
    float sum = MODEL_B2[o];
    for (int h = 0; h < MODEL_HIDDEN_DIM; h++) {
      sum += MODEL_W2[o * MODEL_HIDDEN_DIM + h] * hidden[h];
    }
    outLogits[o] = sum;
  }
}

void modelSoftmax(const float logits[MODEL_OUTPUT_DIM],
                   float outProbabilities[MODEL_OUTPUT_DIM]) {
  float maxLogit = logits[0];
  for (int i = 1; i < MODEL_OUTPUT_DIM; i++) {
    if (logits[i] > maxLogit) maxLogit = logits[i];
  }

  float sumExp = 0.0f;
  for (int i = 0; i < MODEL_OUTPUT_DIM; i++) {
    outProbabilities[i] = expf(logits[i] - maxLogit);
    sumExp += outProbabilities[i];
  }
  for (int i = 0; i < MODEL_OUTPUT_DIM; i++) {
    outProbabilities[i] /= sumExp;
  }
}

int modelArgmax(const float probabilities[MODEL_OUTPUT_DIM]) {
  int best = 0;
  for (int i = 1; i < MODEL_OUTPUT_DIM; i++) {
    if (probabilities[i] > probabilities[best]) best = i;
  }
  return best;
}
