// Teste isolado de captura de áudio via I2S (ESP32 + INMP441).
// Objetivo: confirmar que a pinagem e a leitura do microfone funcionam,
// antes de montar a arquitetura FreeRTOS completa (captura/features/detecção/atuação).
//
// Ligação:
//   INMP441 VDD -> ESP32 3.3V
//   INMP441 GND -> ESP32 GND
//   INMP441 L/R -> ESP32 GND   (seleciona canal esquerdo)
//   INMP441 WS  -> ESP32 GPIO13
//   INMP441 SCK -> ESP32 GPIO14
//   INMP441 SD  -> ESP32 GPIO32
//
// Abra o Serial Monitor (115200 baud) e fale ou bata palmas perto do
// microfone — o valor de RMS impresso deve subir bem acima do nível de
// silêncio quando houver som.

#include <driver/i2s.h>

#define I2S_WS 13
#define I2S_SCK 14
#define I2S_SD 32
#define I2S_PORT I2S_NUM_0

#define SAMPLE_RATE 16000
#define BUFFER_SAMPLES 512

// Dado útil do INMP441 ocupa os bits mais altos da amostra de 32 bits
// lida pelo driver I2S; deslocar 14 bits traz o valor para uma faixa
// numérica prática de imprimir/observar.
#define SAMPLE_SHIFT 14

int32_t raw_samples[BUFFER_SAMPLES];

void i2sInstall() {
  const i2s_config_t i2s_config = {
      .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
      .sample_rate = SAMPLE_RATE,
      .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
      .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
      .communication_format = I2S_COMM_FORMAT_STAND_I2S,
      .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
      .dma_buf_count = 4,
      .dma_buf_len = BUFFER_SAMPLES,
      .use_apll = false,
      .tx_desc_auto_clear = false,
      .fixed_mclk = 0};
  i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);
}

void i2sSetPin() {
  const i2s_pin_config_t pin_config = {
      .bck_io_num = I2S_SCK,
      .ws_io_num = I2S_WS,
      .data_out_num = I2S_PIN_NO_CHANGE,
      .data_in_num = I2S_SD};
  i2s_set_pin(I2S_PORT, &pin_config);
}

void setup() {
  Serial.begin(115200);
  delay(500);

  i2sInstall();
  i2sSetPin();
  i2s_start(I2S_PORT);

  Serial.println("I2S iniciado. Fale ou bata palmas perto do microfone...");
}

void loop() {
  size_t bytes_read = 0;
  i2s_read(I2S_PORT, raw_samples, sizeof(raw_samples), &bytes_read, portMAX_DELAY);
  int samples_read = bytes_read / sizeof(int32_t);

  if (samples_read == 0) {
    return;
  }

  double sum_squares = 0;
  for (int i = 0; i < samples_read; i++) {
    int32_t sample = raw_samples[i] >> SAMPLE_SHIFT;
    sum_squares += (double)sample * (double)sample;
  }
  double rms = sqrt(sum_squares / samples_read);

  Serial.printf("RMS: %.1f\n", rms);
}
