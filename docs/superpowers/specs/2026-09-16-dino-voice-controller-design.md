# Detector de Comandos de Voz para Controle do Jogo do Dinossauro (Chrome)

Data: 2026-09-16
Ponderada: Detector de Anomalias Acústicas (entrega 2026-09-18, checkpoint 2026-09-21)

## Objetivo e justificativa prática

Sistema embarcado ESP32 + INMP441 que detecta em tempo real dois comandos de voz —
"pular" e "abaixa" — e aciona servos que pressionam fisicamente as teclas do jogo do
dinossauro do Chrome (Espaço/Seta-cima para pular, Seta-baixo para abaixar).

Aplicação prática: controle de jogo/dispositivo sem uso das mãos — relevante para
acessibilidade motora ou cenários em que as mãos estão ocupadas. O padrão acústico de
interesse escolhido livremente (conforme permitido pelo enunciado) são os dois comandos
de voz, tratados como três classes acústicas: `pular`, `abaixa`, `ruido` (fundo/silêncio/
outras falas).

## Arquitetura de tarefas (FreeRTOS, via Arduino framework)

Quatro tasks concorrentes (mínimo exigido: 3), conectadas por filas (queues) — sem
polling entre etapas:

1. **Task 1 — Captura de Áudio** (prioridade alta)
   - Lê o microfone I2S continuamente.
   - Usa dois buffers fixos em modo ping-pong: enquanto um buffer é preenchido, o outro
     pode estar sendo consumido pela Task 2. Isso evita necessidade de mutex nesse
     estágio — cada task só acessa um buffer por vez, garantido pela alternância.
   - Ao completar um buffer, envia o ponteiro do buffer via `xQueueSend` para a Task 2.

2. **Task 2 — Feature Extraction** (prioridade média)
   - Recebe o ponteiro do buffer da fila (`xQueueReceive`, bloqueante).
   - Calcula RMS, Spectral Centroid (via FFT) e coeficientes MFCC.
   - Empacota em `struct features_t` e envia por outra fila para a Task 3.

3. **Task 3 — Detecção** (prioridade baixa)
   - Recebe features da fila.
   - Roda o forward-pass do modelo treinado (MLP pequeno, ver seção "Modelo").
   - Aplica threshold de confiança + debounce/hysteresis para evitar múltiplos disparos
     enquanto a palavra ainda está sendo pronunciada.
   - Envia comando (`PULAR` / `ABAIXA` / nenhum) por fila para a Task 4.

4. **Task 4 — Atuação** (prioridade baixa)
   - Recebe o comando da fila.
   - Aciona o servo correspondente (desce braço sobre a tecla física, mantém, sobe).
   - Pisca o LED (cor por classe) e emite beep curto no buzzer — satisfaz o requisito
     "Alertar anomalia via LED/buzzer".
   - Isolar a atuação mecânica (mais lenta que o pipeline de áudio) em sua própria task
     evita que ela bloqueie a captura/detecção.

**Sincronização:** filas (producer/consumer) entre as quatro tasks. O único recurso
verdadeiramente compartilhado por múltiplas tasks é o log de latência (timestamps de
cada etapa, usado para o relatório) — protegido por mutex (`xSemaphoreCreateMutex`),
pois todas as tasks escrevem nele.

## Modelo de detecção

- Features de entrada (14 valores): RMS, Zero-Crossing Rate, Spectral Centroid e ~11
  coeficientes MFCC — extraídos do clipe inteiro (pooling por segmento foi testado e
  descartado, ver "Discussão" do relatório: piorou a acurácia com esse volume de dado).
  Centroide e MFCC são calculados sobre o sinal normalizado por amplitude (pico
  constante), tornando-os invariantes ao ganho do dispositivo de gravação; RMS é
  calculado sobre o sinal bruto, de propósito, para preservar informação de volume
  (distingue fala de ruído/silêncio).
- Rede neural pequena (MLP: 14 → 16 → 3, softmax) treinada em Python, com seed fixa
  (reprodutível).
- Exportado como `.onnx` (entregável exigido).
- Como o ESP32 (Arduino framework) não possui runtime ONNX embarcado simples, o
  forward-pass (poucas centenas de multiplicações) é portado manualmente para C como
  produto de matrizes, gerado a partir dos pesos do modelo treinado. O `.onnx` é o
  artefato de treino/documentação; o C gerado é o que roda de fato no dispositivo.

### Fontes de dados

- **Gravações próprias**: iPhone (Voice Memos) e microfone embutido do notebook
  (`training/record_audio.py`), rotuladas com apoio de transcrição automática
  (Whisper) e conferência manual.
- **MLCommons Multilingual Spoken Words Corpus (MSWC)**: 113 clipes reais adicionais
  ("pula"/"pular": 67 clipes, 33 falantes únicos; "abaixo" — variante mais próxima de
  "abaixa" presente no corpus — 46 clipes, 17 falantes únicos), extraídos por alinhamento
  forçado do Mozilla Common Voice PT-BR. Licença CC-BY 4.0 — atribuição obrigatória no
  relatório. Fonte: https://mlcommons.org/datasets/multilingual-spoken-words/. Usado
  especificamente para combater o viés de domínio de um único microfone (ver risco
  abaixo): são gravações de dezenas de falantes/dispositivos diferentes.

**Risco conhecido:** gravar com microfone do notebook/celular é mais rápido que gravar
direto do INMP441, mas pode haver descasamento de domínio (timbre/ruído diferentes) na
hora da inferência real no ESP32. Mitigação aplicada: normalização de ganho nas features,
augmentation de ganho/deslocamento temporal, e inclusão de dados multi-falante do MSWC —
todas reduziram, mas não eliminaram, um viés residual observado em testes ao vivo
(confusão ocasional entre "pular"/"abaixa" fora do microfone de treino original).
Mitigação pendente: validar a acurácia no dispositivo real assim que o firmware estiver
de pé; se necessário, complementar o dataset com um lote de áudio capturado direto do
INMP441 (sem precisar retreinar do zero).

## Testes

Entregável "código de teste" (script que simula anomalias e mede performance):

1. **Avaliação offline do modelo**: matriz de confusão / acurácia em um conjunto de
   teste separado do treino (métrica para o critério "Detecção de Anomalias — Acurácia").
2. **Latência ponta-a-ponta**: script Python que dispara os comandos (fala ao vivo ou
   playback por alto-falante perto do microfone) enquanto lê via serial os timestamps
   logados pelo ESP32 em cada etapa (captura → features → decisão → atuação), calculando
   a latência de cada estágio e a latência total.

## Entregáveis do projeto

- Repositório com código-fonte (firmware + scripts de treino/teste).
- Diagrama de tarefas RTOS (SVG): tasks, filas, mutex, fluxo de sincronização.
- Modelo de detecção (`.onnx`).
- Relatório técnico: arquitetura RTOS, análise de latência, resultados, discussão.
- Código de teste (avaliação offline + latência).

## Fases de implementação (ordem de execução)

1. **Treino do modelo**: coleta de áudio, extração de features, treino do MLP, export
   ONNX, avaliação offline (acurácia/matriz de confusão).
2. **Firmware ESP32**: as quatro tasks, filas, captura I2S, port do modelo para C,
   log de latência com mutex.
3. **Mecanismo físico**: montagem dos servos sobre o teclado, calibração de
   ângulos/tempos de pressão.
4. **Testes de integração e latência**: script de teste ponta-a-ponta, ajuste de
   thresholds/debounce.
5. **Documentação**: diagrama RTOS (SVG) e relatório técnico.

A Fase 1 (treino do modelo) é executada primeiro, isoladamente, antes do firmware —
decisão do usuário para validar o maior risco do projeto (qualidade dos dados de voz)
antes de investir no restante da implementação.
