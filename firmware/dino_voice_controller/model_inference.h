// Forward-pass da rede treinada (14 -> 16 -> 3), portado manualmente para C —
// o ESP32 (framework Arduino) não tem runtime ONNX embarcado simples, então
// os pesos (firmware/model_weights.h) são aplicados diretamente como produto
// de matrizes. Ver docs/superpowers/specs/2026-09-16-dino-voice-controller-design.md.
#pragma once

#include "model_weights.h"

// Classes na mesma ordem usada no treino (training/dataset.py CLASSES).
enum ModelClass { CLASS_RUIDO = 0, CLASS_PULAR = 1, CLASS_ABAIXA = 2 };

// Normaliza um vetor de features cru (mesma escala usada no treino: z-score
// com a média/desvio do conjunto de treino).
void modelNormalizeFeatures(const float rawFeatures[MODEL_INPUT_DIM],
                             float normalized[MODEL_INPUT_DIM]);

// Roda o forward-pass (Linear -> ReLU -> Linear) e retorna os logits brutos
// (antes do softmax) em outLogits[MODEL_OUTPUT_DIM].
void modelForward(const float normalizedFeatures[MODEL_INPUT_DIM],
                   float outLogits[MODEL_OUTPUT_DIM]);

// Converte logits em probabilidades (soma 1.0).
void modelSoftmax(const float logits[MODEL_OUTPUT_DIM],
                   float outProbabilities[MODEL_OUTPUT_DIM]);

// Índice (0..MODEL_OUTPUT_DIM-1) da maior probabilidade.
int modelArgmax(const float probabilities[MODEL_OUTPUT_DIM]);
