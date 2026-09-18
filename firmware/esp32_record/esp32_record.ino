// Sketch de gravação: espera um byte 'r' pela Serial, captura exatamente
// 1s (16000 amostras PCM int16 a 16kHz) pelo INMP441 via I2S, e devolve
// pela Serial um cabeçalho "REC1" seguido dos 32000 bytes brutos.
//
// Usado por training/record_from_esp32.py pra gravar dados de treino
// direto do microfone real do dispositivo (fecha o descasamento de
// domínio entre o mic usado no treino e o mic final do ESP32).
//
// Mesma pinagem dos outros sketches:
//   INMP441 WS  -> GPIO13   INMP441 SCK -> GPIO14   INMP441 SD -> GPIO32

#include <driver/i2s.h>

#define I2S_WS 13
#define I2S_SCK 14
#define I2S_SD 32
#define I2S_PORT I2S_NUM_0

#define SAMPLE_RATE 16000
#define CLIP_LENGTH 16000
#define I2S_READ_CHUNK_SAMPLES 512
// Mesma convenção do dino_voice_controller: desloca pro alcance de um
// PCM int16 padrão (idêntico ao formato dos .wav do dataset de treino).
#define I2S_SAMPLE_SHIFT 16

static int16_t clipBuffer[CLIP_LENGTH];

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

void captureOneClip() {
  static int32_t rawSamples[I2S_READ_CHUNK_SAMPLES];
  int sampleCount = 0;

  while (sampleCount < CLIP_LENGTH) {
    size_t bytesRead = 0;
    i2s_read(I2S_PORT, rawSamples, sizeof(rawSamples), &bytesRead, portMAX_DELAY);
    int samplesRead = bytesRead / sizeof(int32_t);

    for (int i = 0; i < samplesRead && sampleCount < CLIP_LENGTH; i++) {
      clipBuffer[sampleCount++] = (int16_t)(rawSamples[i] >> I2S_SAMPLE_SHIFT);
    }
  }
}

void setup() {
  Serial.begin(115200);
  delay(500);
  i2sInstall();
  Serial.println("READY");
}

void loop() {
  if (Serial.available() > 0) {
    int trigger = Serial.read();
    if (trigger == 'r') {
      captureOneClip();
      Serial.write((const uint8_t *)"REC1", 4);
      Serial.write((const uint8_t *)clipBuffer, CLIP_LENGTH * sizeof(int16_t));
      Serial.flush();
    }
  }
}
