# Detector de Anomalias Acusticas - Controle por Voz do Dino do Chrome

Projeto da ponderada "Detector de Anomalias Acusticas". Em vez de detectar
uma anomalia no sentido estrito (queda, grito, latido), o sistema reconhece
dois comandos de voz em portugues, "pular" e "abaixa", e usa servos motores
para apertar fisicamente as teclas do jogo do dinossauro do Chrome.

O hardware e um ESP32 + microfone I2S INMP441, rodando um firmware com 4
tasks FreeRTOS (captura de audio, extracao de features, deteccao, atuacao).
O modelo e uma rede neural pequena (14 -> 16 -> 3, 291 parametros) treinada
em Python e portada manualmente pra C, ja que o dispositivo nao tem um
runtime ONNX embarcado.

Para o passo a passo de como o projeto foi feito (o que deu certo, o que
deu errado, os bugs no caminho), ver [`docs/diario_de_bordo.md`](docs/diario_de_bordo.md).
Para arquitetura, metodologia e resultados de forma direta, ver
[`docs/relatorio_tecnico.md`](docs/relatorio_tecnico.md). Para o diagrama
das tasks FreeRTOS, ver [`docs/diagrama_rtos.md`](docs/diagrama_rtos.md).

## Estrutura do repositorio

```
firmware/
  dino_voice_controller/   firmware principal (4 tasks FreeRTOS)
  esp32_record/            sketch usado so pra gravar dataset direto do ESP32
  i2s_mic_test/             teste isolado do microfone I2S
training/                  pipeline de treino em Python (features, augmentacao, treino, exportacao)
models/                    modelo treinado (model.onnx) e estatisticas de normalizacao
docs/                      relatorio tecnico, diario de bordo, diagramas, fotos e video
```

## Hardware

- ESP32-WROOM-32U
- Microfone I2S INMP441
- 2x micro servo SG90 (Tower Pro)
- Fonte externa de 5V pros servos (o pino de alimentacao do ESP32 sozinho
  nao da corrente suficiente)

Pinagem usada no firmware (`firmware/dino_voice_controller/dino_voice_controller.ino`):

| Sinal | Pino ESP32 |
|---|---|
| I2S WS (word select) | GPIO 13 |
| I2S SCK (bit clock) | GPIO 14 |
| I2S SD (dado) | GPIO 32 |
| Servo "pular" | GPIO 18 |
| Servo "abaixa" | GPIO 19 |
| LED de alerta | GPIO 2 (LED embutido da placa) |

![Detalhe do circuito na protoboard: ESP32-WROOM-32U, microfone INMP441 e fiacao](docs/images/IMG_1537.jpg)

*Circuito montado na protoboard: ESP32-WROOM-32U, microfone I2S INMP441 e a
fiacao de sinal dos dois servos. Mais fotos da montagem completa (servos
sobre o teclado) em [`docs/relatorio_tecnico.md`](docs/relatorio_tecnico.md#10-demonstracao).*

## Dataset

O dataset de audio esta versionado em `training/data/`:

- `raw/` - dataset final (142 clipes), gravado direto pelo INMP441 do
  dispositivo, o mesmo usado pelo modelo em producao.
- `esp32_raw/` - pasta de staging com as gravacoes brutas do ESP32 antes
  da curadoria.
- `raw_archive_mixed_sources/` - dataset intermediario de etapas
  anteriores do projeto (audio de iPhone/notebook + clipes derivados do
  MLCommons Multilingual Spoken Words Corpus, CC-BY 4.0), descartado do
  modelo final por causa do descasamento de microfone (ver
  [`docs/diario_de_bordo.md`](docs/diario_de_bordo.md)). Mantido no repo
  por documentar esse experimento.

## Rodando o firmware

1. Abra a pasta `firmware/dino_voice_controller/` no Arduino IDE (ou use
   `arduino-cli`).
2. Instale o core do ESP32 e a biblioteca `ESP32Servo`.
3. Compile e envie pro ESP32.
4. Com o dispositivo ligado e o microfone/servos conectados, fale "pular"
   ou "abaixa" perto do microfone com o jogo do dinossauro aberto no
   navegador.

## Rodando o pipeline de treino

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r training/requirements.txt

python -m training.train          # treina e avalia o modelo
python -m training.export_c_model # gera firmware/dino_voice_controller/model_weights.h
python -m training.export_c_dsp   # gera firmware/dino_voice_controller/dsp_tables.h
python -m training.test_detector  # simula anomalias sobre o teste e mede latencia/performance
python -m training.live_test      # testa o modelo ao vivo pelo microfone do computador
```

Rodar os testes:

```bash
pytest
```

## Resultado atual

Avaliacao offline no conjunto de teste (29 clipes gravados pelo INMP441,
nunca vistos no treino): 83% de acuracia geral, com "pular" em 100% de
precisao. Detalhes completos, matriz de confusao e discussao das
limitacoes estao em [`docs/relatorio_tecnico.md`](docs/relatorio_tecnico.md).
