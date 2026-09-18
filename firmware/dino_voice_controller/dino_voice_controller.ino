// Sketch principal: captura de áudio (I2S) -> extração de features ->
// detecção (modelo) -> atuação dos servos, como 4 tasks FreeRTOS.
// Ver docs/superpowers/specs/2026-09-16-dino-voice-controller-design.md
//
// Este arquivo ainda está em construção — por enquanto só confirma que o
// projeto compila com os módulos de DSP/modelo já incluídos.

#include "feature_extraction.h"
#include "model_inference.h"

void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.println("dino_voice_controller: modulos de DSP e modelo carregados.");
  Serial.printf("FEATURE_DIM=%d MODEL_INPUT_DIM=%d\n", FEATURE_DIM, MODEL_INPUT_DIM);
}

void loop() {
  delay(1000);
}
