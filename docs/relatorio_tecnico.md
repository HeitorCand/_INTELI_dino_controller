# Relatorio Tecnico - Detector de Anomalias Acusticas

**Projeto:** Controle por voz do jogo do dinossauro do Chrome, usando ESP32 +
INMP441 + FreeRTOS + servos.

**Padrao acustico escolhido:** dois comandos de voz, "pular" e "abaixa", mais
uma classe de rejeicao ("ruido": silencio, ruido de fundo, outras falas).
Aplicacao pratica: controle de jogo sem uso das maos.

---

## 1. Objetivo e justificativa

O sistema detecta em tempo real, a partir do audio capturado por um
microfone I2S (INMP441), se a pessoa disse "pular", "abaixa" ou nenhuma das
duas coisas (ruido), e aciona um servo motor que aperta fisicamente a tecla
correspondente do jogo do dinossauro do Chrome. A escolha de comandos de voz
como "anomalia acustica de interesse" foi feita pensando em uma aplicacao de
acessibilidade: controlar um jogo (ou, de forma mais geral, qualquer
dispositivo) sem precisar usar as maos.

## 2. Arquitetura RTOS

A arquitetura completa (4 tasks, filas, semaforos e mutex) esta detalhada em
[`diagrama_rtos.md`](diagrama_rtos.md) e [`diagrama_rtos.svg`](diagrama_rtos.svg).
Resumo:

- **Task 1 - Captura de Audio** (prioridade alta): le o microfone via I2S
  continuamente e preenche um buffer duplo (`bufferA`/`bufferB`, amostras
  PCM int16 de 16 bits). Dois semaforos binarios garantem que um buffer só
  seja reaproveitado depois que a Task 2 terminar de le-lo.
- **Task 2 - Extracao de Features** (prioridade media): recebe o buffer
  cheio por fila, calcula RMS, Zero-Crossing Rate, Centroide Espectral e 11
  coeficientes MFCC (FFT propria de 512 pontos, ver secao 3), e envia o
  vetor de 14 features por outra fila.
- **Task 3 - Deteccao** (prioridade baixa): roda o forward-pass da rede
  neural treinada, aplica um limiar de confianca e um piso minimo de RMS
  (para rejeitar silencio/ruido de baixa energia), alem de um cooldown de
  800ms entre comandos aceitos.
- **Task 4 - Atuacao** (prioridade baixa): aciona o servo correspondente
  (pular ou abaixa) e pisca o LED / aciona o buzzer como alerta.

O unico recurso compartilhado fora do fluxo normal das filas e o registro de
latencia (`LatencyLog`), protegido por um mutex, ja que mais de uma task
escreve nele.

## 3. Pipeline de deteccao (features + modelo)

### 3.1 Extracao de features

Em vez de usar os parametros padrao do `librosa` (FFT de 2048 pontos, 128
bandas mel), foi implementada uma FFT radix-2 propria em C (512 pontos, 26
bandas mel, janela de Hann) dimensionada para caber e rodar em tempo real no
ESP32. A mesma logica foi escrita primeiro em Python
(`training/dsp.py`) e depois portada para C
(`firmware/dino_voice_controller/fft.cpp` e `feature_extraction.cpp`),
verificando numericamente que as duas implementacoes batem (erro relativo na
casa de 1e-4 a 1e-7, testado com tons puros e com clipes reais do dataset).

Vetor de features final (14 valores): RMS, Zero-Crossing Rate, Centroide
Espectral e 11 coeficientes MFCC. RMS e calculado sobre o sinal bruto (carrega
informacao de volume, usada para distinguir fala de silencio); centroide e
MFCC sao calculados sobre uma versao com o pico de amplitude normalizado, o
que os torna invariantes ao ganho do microfone/dispositivo de gravacao.

### 3.2 Modelo

Rede neural pequena (MLP: 14 -> 16 -> 3, ReLU + softmax), treinada em Python
(PyTorch) e exportada em `.onnx`. Como o ESP32 nao tem um runtime ONNX
embarcado simples, o forward-pass foi portado manualmente para C como
produto de matrizes (`model_inference.cpp`), usando os pesos exportados de
`models/model.onnx` (`training/export_c_model.py`). Essa porta foi verificada
numericamente contra o modelo em Python (logits batendo em ~1e-5).

Tamanho do modelo: 291 parametros (~1.16KB em float32) - pequeno o
suficiente para rodar em microssegundos no ESP32, sem necessidade de
quantizacao.

## 4. Coleta de dados e principal desafio encontrado

O maior desafio do projeto nao foi a arquitetura RTOS em si, mas conseguir
dados de treino que generalizassem para o microfone real do dispositivo
final (INMP441 no ESP32). O dataset passou por varias fontes ao longo do
projeto:

1. Gravacoes proprias via iPhone (Voice Memos) e microfone do notebook.
2. Dados publicos do MLCommons Multilingual Spoken Words Corpus (MSWC),
   extraidos por alinhamento forcado do Mozilla Common Voice PT-BR
   (CC-BY 4.0).
3. Gravacoes feitas diretamente pelo ESP32 (script de gravacao via
   Serial, `firmware/esp32_record/esp32_record.ino` +
   `training/record_from_esp32.py`).

Em testes ao vivo, o modelo treinado só com dados de outros microfones
(iPhone, notebook, MSWC) apresentava um viés forte: a fala real captada pelo
INMP441 era classificada quase sempre como a mesma classe, independente do
que a pessoa realmente dissesse. Um experimento controlado (treinar e testar
usando *apenas* dados gravados pelo proprio ESP32) confirmou que o problema
era descasamento de dominio entre microfones, e nao um erro no modelo ou no
pipeline: usando so dados do ESP32, a classe "pular" chegou a 100% de
precisao. O modelo final foi treinado exclusivamente com dados reais do
ESP32 (49 clipes de "pular", 43 de "abaixa", 50 de "ruido").

Outra licao tecnica relevante: uma auditoria manual (com apoio de
transcricao automatica via Whisper) encontrou que uma parte significativa
dos dados do MSWC estava mal alinhada (a palavra transcrita nao era a
palavra realmente dita, ou o audio continha apenas ruido). Remover esses
dados contaminados mudou a acuracia relatada, mas resultou em um modelo mais
confiavel.

## 5. Analise de latencia

Cada etapa do pipeline (captura -> features -> deteccao -> atuacao) tem seu
tempo medido com `micros()`/`millis()` e impresso via Serial. Exemplo real
capturado durante um teste em bancada:

```
[PULAR] captura->features=84838us  features->deteccao=45us  deteccao->atuacao=230030us  total=314901us
```

| Etapa | Tempo tipico | O que inclui |
|---|---|---|
| Captura -> Features | ~85 ms | FFT (512 pontos) + banco de filtros mel + DCT, repetido para ~61 janelas dentro do clipe de 1s |
| Features -> Deteccao | < 1 ms | Forward-pass da rede (291 parametros) - custo desprezivel |
| Deteccao -> Atuacao | ~230 ms a ~1080 ms (depende do comando) | Alerta (LED/buzzer, 80ms) + tempo que o servo fica pressionado (configuravel por comando, hoje 300ms para "pular" e 1000ms para "abaixa") |

Dois pontos importantes para interpretar esses numeros:

- A etapa "Deteccao -> Atuacao" e dominada por uma escolha de projeto (quanto
  tempo o servo fica pressionado), nao por um gargalo de processamento. O
  tempo de calculo real da deteccao (features -> deteccao) e desprezivel.
- O maior componente de latencia *percebida* pelo usuario nao aparece nessa
  tabela: a Task 1 so envia um buffer para processamento depois de
  acumular 1 segundo inteiro de audio (16000 amostras a 16kHz). Ou seja, o
  sistema so pode reagir, na pior das hipoteses, ate ~1 segundo depois do
  inicio da fala, antes mesmo de qualquer processamento comecar.

## 6. Resultados

Avaliacao offline no conjunto de teste (dados reais do ESP32, separados do
treino antes de qualquer aumento de dados, para evitar vazamento de
informacao):

```
              precision    recall  f1-score   support

       ruido       0.75      0.90      0.82        10
       pular       1.00      0.90      0.95        10
      abaixa       0.75      0.67      0.71         9

    accuracy                           0.83        29
```

Matriz de confusao (linhas = classe real, colunas = classe prevista, ordem
ruido/pular/abaixa):

```
[[9 0 1]
 [0 9 1]
 [3 0 6]]
```

"Pular" tem precisao de 100% nesse teste. O erro mais comum e "abaixa" sendo
confundido com "ruido" (3 casos) - um erro relativamente seguro, ja que
resulta em nao acionar nada, em vez de acionar o servo errado.

<!-- ESPACO PARA IMAGEM -->
### [Imagem: montagem fisica do hardware]
*Legenda: foto do ESP32, INMP441, servos, LED e buzzer montados na bancada de
teste.*

<!-- inserir imagem aqui -->

## 7. Discussao

**O que funcionou bem:**
- A separacao em 4 tasks com filas deixou o pipeline facil de depurar
  etapa por etapa (dava pra medir e imprimir o estado em cada fronteira).
- Portar a extracao de features "na mao" para C (em vez de depender de uma
  biblioteca pesada) manteve o uso de memoria baixo (29% da RAM do ESP32) e
  permitiu verificar cada etapa numericamente contra a versao em Python.
- Gravar dados de treino diretamente pelo hardware final (em vez de confiar
  só em gravacoes de outros microfones) foi a mudanca que mais melhorou o
  comportamento em uso real.

**Limitacoes conhecidas:**
- O dataset final (142 clipes reais) e pequeno para um problema de
  reconhecimento de fala, mesmo que restrito a duas palavras. A acuracia de
  83% tem bastante variancia dado o tamanho do conjunto de teste (29
  amostras).
- O modelo ainda confunde "pular" e "abaixa" com alguma frequencia quando a
  fala e mais rapida ou mais baixa.
- A separacao entre fala e ruido de fundo depende de um piso de RMS fixo,
  calibrado manualmente a partir de medicoes em bancada; ele pode precisar
  de reajuste em outro ambiente (mais ou menos ruidoso) ou com outro
  posicionamento do microfone.

**Trabalho futuro:** gravar mais dados diretamente do ESP32 (o experimento
mostrou que e o que mais compensa), e considerar normalizar a duracao/
alinhamento da fala dentro da janela de captura.

## 8. Estrutura do repositorio

- `firmware/` - codigo do ESP32 (Arduino): sketch principal
  (`dino_voice_controller/`), teste isolado de microfone (`i2s_mic_test/`) e
  utilitario de gravacao de dados (`esp32_record/`).
- `training/` - pipeline de treino em Python (extracao de features,
  augmentacao, treino, exportacao para ONNX e para C), com testes
  automatizados (`training/tests/`).
- `models/` - modelo treinado (`model.onnx`), estatisticas de normalizacao e
  relatorio de avaliacao.
- `docs/` - este relatorio, o diagrama RTOS e a spec de arquitetura original.

## 9. Demonstracao

<!-- ESPACO PARA VIDEO -->
### [Video: demonstracao do sistema em funcionamento]
*Legenda: video mostrando o comando de voz "pular"/"abaixa" acionando o jogo
do dinossauro do Chrome em tempo real, via ESP32 + servos.*

<!-- inserir video aqui (arquivo ou link) -->
