// Detector de comandos de voz "pular"/"abaixa" para controlar o jogo do
// dinossauro do Chrome via servos, no ESP32 + microfone INMP441.
//
// Arquitetura (4 tasks FreeRTOS, ver
// docs/superpowers/specs/2026-09-16-dino-voice-controller-design.md):
//   Task 1 (alta prioridade)  — Captura de Áudio (I2S, buffer duplo)
//   Task 2 (prioridade média) — Extração de Features (RMS/ZCR/centroide/MFCC)
//   Task 3 (prioridade baixa) — Detecção (forward-pass do modelo + threshold)
//   Task 4 (prioridade baixa) — Atuação (servos + LED/buzzer)
//
// Sincronização: filas entre as 4 tasks (produtor/consumidor); 2 semáforos
// binários garantem que a Task 1 nunca sobrescreve um buffer que a Task 2
// ainda não terminou de ler (double-buffering correto); 1 mutex protege o
// registro de latência (único recurso realmente compartilhado por todas
// as tasks, fora das filas).

#include <math.h>

#include <driver/i2s.h>
#include <ESP32Servo.h>

#include "feature_extraction.h"
#include "model_inference.h"

// ---------------------------------------------------------------------------
// Pinagem
// ---------------------------------------------------------------------------
#define I2S_WS 13
#define I2S_SCK 14
#define I2S_SD 32
#define I2S_PORT I2S_NUM_0

#define SERVO_PULAR_PIN 18
#define SERVO_ABAIXA_PIN 19
#define LED_PIN 25
#define BUZZER_PIN 26

// Ângulos do servo — CALIBRAR na montagem física real. Mesma lógica pros
// dois (solto em 0, pressiona indo pra 30) — a montagem física do servo de
// "abaixa" foi invertida no hardware, então não precisa mais inverter no
// software.
#define SERVO_PULAR_RELEASED_ANGLE 0
#define SERVO_PULAR_PRESSED_ANGLE 32
#define SERVO_ABAIXA_RELEASED_ANGLE 0
#define SERVO_ABAIXA_PRESSED_ANGLE 30
// Tempo que o braço fica pressionado antes de soltar.
#define SERVO_PULAR_PRESS_HOLD_MS 300
#define SERVO_ABAIXA_PRESS_HOLD_MS 300

#define SAMPLE_RATE 16000
#define I2S_READ_CHUNK_SAMPLES 512
// Dado útil do INMP441 ocupa os bits mais altos da amostra de 32 bits.
// Deslocar 16 bits deixa o valor direto na faixa de um PCM int16 padrão —
// mesma convenção dos .wav do dataset de treino (ver feature_extraction.cpp).
#define I2S_SAMPLE_SHIFT 16

#define CONFIDENCE_THRESHOLD 0.75f
#define COMMAND_COOLDOWN_MS 800

// Piso mínimo de RMS — FIXO, não calibrado por uma única amostra (a
// calibração de 1s testada antes pegou um pico isolado de 0.3281 e ficou
// inutilizável). ATENÇÃO: a margem entre ruído ambiente (~0.02-0.10) e
// fala real medida em bancada (~0.10) é estreita — fale alto e perto do
// microfone pra ficar bem acima do piso. Reajustar aqui se necessário
// depois de medir com o DEBUG no Serial.
#define MIN_RMS_FLOOR 0.07f

// ---------------------------------------------------------------------------
// Mensagens entre tasks
// ---------------------------------------------------------------------------
typedef struct {
  int16_t *buffer;      // ponteiro pra bufferA ou bufferB
  int bufferIndex;      // 0 ou 1 — pra Task 2 saber qual semáforo liberar
  uint32_t captureDoneUs;
} AudioBufferMsg;

typedef struct {
  float features[FEATURE_DIM];
  uint32_t captureDoneUs;
  uint32_t featuresDoneUs;
} FeaturesMsg;

typedef struct {
  int predictedClass;  // CLASS_RUIDO / CLASS_PULAR / CLASS_ABAIXA
  uint32_t captureDoneUs;
  uint32_t featuresDoneUs;
  uint32_t detectionDoneUs;
} CommandMsg;

static QueueHandle_t audioQueue;
static QueueHandle_t featuresQueue;
static QueueHandle_t commandQueue;

// Buffer duplo de captura — "static" (não na pilha), amostras PCM de 16
// bits (mesma convenção do dataset de treino): 16000 int16 = 32KB cada.
static int16_t bufferA[DSP_CLIP_LENGTH];
static int16_t bufferB[DSP_CLIP_LENGTH];
static int16_t *const captureBuffers[2] = {bufferA, bufferB};
static SemaphoreHandle_t bufferFreeSemaphore[2];

// Log de latência compartilhado — único recurso que não passa por fila,
// por isso precisa de mutex (todas as 4 tasks escrevem/leem nele).
typedef struct {
  uint32_t lastCaptureToFeaturesUs;
  uint32_t lastFeaturesToDetectionUs;
  uint32_t lastDetectionToActuationUs;
  uint32_t lastTotalUs;
} LatencyLog;

static LatencyLog latencyLog = {0, 0, 0, 0};
static SemaphoreHandle_t latencyMutex;

static Servo servoPular;
static Servo servoAbaixa;

// ---------------------------------------------------------------------------
// Task 1 — Captura de Áudio (alta prioridade)
// ---------------------------------------------------------------------------
void i2sInstall() {
  const i2s_config_t i2sConfig = {
      .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
      .sample_rate = SAMPLE_RATE,
      .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
      .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
      .communication_format = I2S_COMM_FORMAT_STAND_I2S,
      .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
      .dma_buf_count = 4,
      .dma_buf_len = I2S_READ_CHUNK_SAMPLES,
      .use_apll = false,
      .tx_desc_auto_clear = false,
      .fixed_mclk = 0};
  i2s_driver_install(I2S_PORT, &i2sConfig, 0, NULL);

  const i2s_pin_config_t pinConfig = {
      .bck_io_num = I2S_SCK,
      .ws_io_num = I2S_WS,
      .data_out_num = I2S_PIN_NO_CHANGE,
      .data_in_num = I2S_SD};
  i2s_set_pin(I2S_PORT, &pinConfig);
  i2s_start(I2S_PORT);
}

void audioCaptureTask(void *pvParameters) {
  static int32_t rawSamples[I2S_READ_CHUNK_SAMPLES];
  int activeIndex = 0;
  int sampleCount = 0;

  for (;;) {
    // Espera o buffer atual ficar livre (a Task 2 já terminou de lê-lo) —
    // impede que a captura sobrescreva um buffer ainda em uso.
    xSemaphoreTake(bufferFreeSemaphore[activeIndex], portMAX_DELAY);
    int16_t *activeBuffer = captureBuffers[activeIndex];
    sampleCount = 0;

    while (sampleCount < DSP_CLIP_LENGTH) {
      size_t bytesRead = 0;
      i2s_read(I2S_PORT, rawSamples, sizeof(rawSamples), &bytesRead, portMAX_DELAY);
      int samplesRead = bytesRead / sizeof(int32_t);

      for (int i = 0; i < samplesRead && sampleCount < DSP_CLIP_LENGTH; i++) {
        activeBuffer[sampleCount++] = (int16_t)(rawSamples[i] >> I2S_SAMPLE_SHIFT);
      }
    }

    AudioBufferMsg msg;
    msg.buffer = activeBuffer;
    msg.bufferIndex = activeIndex;
    msg.captureDoneUs = micros();
    xQueueSend(audioQueue, &msg, portMAX_DELAY);

    activeIndex = 1 - activeIndex;
  }
}

// ---------------------------------------------------------------------------
// Task 2 — Extração de Features (prioridade média)
// ---------------------------------------------------------------------------
void featureExtractionTask(void *pvParameters) {
  AudioBufferMsg audioMsg;
  FeaturesMsg featuresMsg;

  for (;;) {
    xQueueReceive(audioQueue, &audioMsg, portMAX_DELAY);

    extractFeatures(audioMsg.buffer, featuresMsg.features);

    // Terminou de ler o buffer — libera pra Task 1 reusar.
    xSemaphoreGive(bufferFreeSemaphore[audioMsg.bufferIndex]);

    featuresMsg.captureDoneUs = audioMsg.captureDoneUs;
    featuresMsg.featuresDoneUs = micros();
    xQueueSend(featuresQueue, &featuresMsg, portMAX_DELAY);
  }
}

// ---------------------------------------------------------------------------
// Task 3 — Detecção (prioridade baixa)
// ---------------------------------------------------------------------------
void detectionTask(void *pvParameters) {
  FeaturesMsg featuresMsg;
  CommandMsg commandMsg;
  uint32_t lastCommandMs = 0;

  for (;;) {
    xQueueReceive(featuresQueue, &featuresMsg, portMAX_DELAY);

    float normalized[MODEL_INPUT_DIM];
    float logits[MODEL_OUTPUT_DIM];
    float probabilities[MODEL_OUTPUT_DIM];

    modelNormalizeFeatures(featuresMsg.features, normalized);
    modelForward(normalized, logits);
    modelSoftmax(logits, probabilities);

    int predicted = modelArgmax(probabilities);
    bool tooQuiet = featuresMsg.features[0] < MIN_RMS_FLOOR;  // features[0] = RMS
    bool confident = probabilities[predicted] >= CONFIDENCE_THRESHOLD;
    bool isCommand = !tooQuiet && confident && predicted != CLASS_RUIDO;

    // DEBUG temporario — remover depois de confirmar que o piso de RMS
    // resolveu o disparo falso em silencio. Mostra RMS/ZCR e as 3
    // probabilidades em toda janela, nao so quando um comando dispara.
    Serial.printf(
        "DEBUG rms=%.4f zcr=%.4f tooQuiet=%d probs=[ruido=%.3f pular=%.3f abaixa=%.3f]\n",
        featuresMsg.features[0], featuresMsg.features[1], tooQuiet, probabilities[0],
        probabilities[1], probabilities[2]);

    uint32_t nowMs = millis();
    bool cooldownExpired = (nowMs - lastCommandMs) >= COMMAND_COOLDOWN_MS;

    if (isCommand && cooldownExpired) {
      lastCommandMs = nowMs;

      commandMsg.predictedClass = predicted;
      commandMsg.captureDoneUs = featuresMsg.captureDoneUs;
      commandMsg.featuresDoneUs = featuresMsg.featuresDoneUs;
      commandMsg.detectionDoneUs = micros();
      xQueueSend(commandQueue, &commandMsg, portMAX_DELAY);
    }

    // Atualiza o log de latência mesmo quando não houve comando acionável,
    // pra registrar o tempo de captura->features->detecção continuamente.
    xSemaphoreTake(latencyMutex, portMAX_DELAY);
    latencyLog.lastCaptureToFeaturesUs = featuresMsg.featuresDoneUs - featuresMsg.captureDoneUs;
    latencyLog.lastFeaturesToDetectionUs = micros() - featuresMsg.featuresDoneUs;
    xSemaphoreGive(latencyMutex);
  }
}

// ---------------------------------------------------------------------------
// Task 4 — Atuação (prioridade baixa)
// ---------------------------------------------------------------------------
void alert() {
  digitalWrite(LED_PIN, HIGH);
  digitalWrite(BUZZER_PIN, HIGH);
  delay(80);
  digitalWrite(LED_PIN, LOW);
  digitalWrite(BUZZER_PIN, LOW);
}

void pressServo(Servo &servo, int releasedAngle, int pressedAngle, int holdMs) {
  servo.write(pressedAngle);
  delay(holdMs);
  servo.write(releasedAngle);
}

void actuationTask(void *pvParameters) {
  CommandMsg commandMsg;

  for (;;) {
    xQueueReceive(commandQueue, &commandMsg, portMAX_DELAY);

    alert();
    if (commandMsg.predictedClass == CLASS_PULAR) {
      pressServo(servoPular, SERVO_PULAR_RELEASED_ANGLE, SERVO_PULAR_PRESSED_ANGLE,
                 SERVO_PULAR_PRESS_HOLD_MS);
    } else if (commandMsg.predictedClass == CLASS_ABAIXA) {
      pressServo(servoAbaixa, SERVO_ABAIXA_RELEASED_ANGLE, SERVO_ABAIXA_PRESSED_ANGLE,
                 SERVO_ABAIXA_PRESS_HOLD_MS);
    }

    uint32_t actuationDoneUs = micros();

    xSemaphoreTake(latencyMutex, portMAX_DELAY);
    latencyLog.lastDetectionToActuationUs = actuationDoneUs - commandMsg.detectionDoneUs;
    latencyLog.lastTotalUs = actuationDoneUs - commandMsg.captureDoneUs;
    uint32_t captureToFeatures = latencyLog.lastCaptureToFeaturesUs;
    uint32_t featuresToDetection = latencyLog.lastFeaturesToDetectionUs;
    uint32_t detectionToActuation = latencyLog.lastDetectionToActuationUs;
    uint32_t total = latencyLog.lastTotalUs;
    xSemaphoreGive(latencyMutex);

    const char *className = commandMsg.predictedClass == CLASS_PULAR ? "PULAR" : "ABAIXA";
    Serial.printf(
        "[%s] captura->features=%uus features->deteccao=%uus deteccao->atuacao=%uus total=%uus\n",
        className, captureToFeatures, featuresToDetection, detectionToActuation, total);
  }
}

// ---------------------------------------------------------------------------
// Setup / loop
// ---------------------------------------------------------------------------
void setup() {
  Serial.begin(115200);
  delay(500);

  pinMode(LED_PIN, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);
  digitalWrite(BUZZER_PIN, LOW);

  servoPular.attach(SERVO_PULAR_PIN);
  servoAbaixa.attach(SERVO_ABAIXA_PIN);
  servoPular.write(SERVO_PULAR_RELEASED_ANGLE);
  servoAbaixa.write(SERVO_ABAIXA_RELEASED_ANGLE);

  i2sInstall();

  audioQueue = xQueueCreate(2, sizeof(AudioBufferMsg));
  featuresQueue = xQueueCreate(2, sizeof(FeaturesMsg));
  commandQueue = xQueueCreate(2, sizeof(CommandMsg));

  bufferFreeSemaphore[0] = xSemaphoreCreateBinary();
  bufferFreeSemaphore[1] = xSemaphoreCreateBinary();
  xSemaphoreGive(bufferFreeSemaphore[0]);  // ambos os buffers começam livres
  xSemaphoreGive(bufferFreeSemaphore[1]);

  latencyMutex = xSemaphoreCreateMutex();

  xTaskCreate(audioCaptureTask, "captura_audio", 4096, NULL, 3, NULL);
  xTaskCreate(featureExtractionTask, "extracao_features", 8192, NULL, 2, NULL);
  xTaskCreate(detectionTask, "deteccao", 4096, NULL, 1, NULL);
  xTaskCreate(actuationTask, "atuacao", 4096, NULL, 1, NULL);

  Serial.println("Sistema iniciado. Fale 'pular' ou 'abaixa'.");
}

void loop() {
  // Tudo roda nas tasks FreeRTOS acima; loop() fica vazio de propósito.
  vTaskDelay(portMAX_DELAY);
}
