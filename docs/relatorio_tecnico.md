# Relatorio Tecnico - Detector de Anomalias Acusticas

**Disciplina:** Ponderada - Detector de Anomalias Acusticas
**Projeto:** controle por voz do jogo do dinossauro do Chrome, usando ESP32
+ INMP441 + FreeRTOS + servos.

Para o historico de desenvolvimento (o que foi tentado, o que deu errado, o
que funcionou), ver [`diario_de_bordo.md`](diario_de_bordo.md). Este
documento cobre a arquitetura, a metodologia e os resultados de forma mais
direta.

---

## 1. Objetivo e justificativa

O enunciado da ponderada permite escolher livremente qual padrao acustico
detectar. A escolha aqui foi reconhecer dois comandos de voz, "pular" e
"abaixa", em vez de uma anomalia no sentido estrito (queda, grito, etc.).
A justificativa e de aplicacao pratica: controlar um dispositivo sem usar
as maos. O sistema aciona um servo motor que aperta fisicamente a tecla
correspondente do jogo do dinossauro do Chrome. Cada comando e tratado
como uma classe acustica a ser reconhecida, junto com uma terceira classe
de rejeicao ("ruido": silencio, ruido de fundo, outras falas).

## 2. Arquitetura RTOS

O sistema roda como 4 tasks FreeRTOS no ESP32 (framework Arduino), descritas
em detalhe com diagrama em [`diagrama_rtos.md`](diagrama_rtos.md):

| Task | Prioridade | Responsabilidade |
|---|---|---|
| 1 - Captura de Audio | alta (3) | Le o I2S continuamente, preenche um buffer duplo (`bufferA`/`bufferB`, PCM int16, 16000 amostras = 1s a 16kHz) |
| 2 - Extracao de Features | media (2) | Recebe o buffer cheio, calcula RMS, Zero-Crossing Rate, Centroide Espectral e 11 coeficientes MFCC |
| 3 - Deteccao | baixa (1) | Roda o forward-pass do modelo, aplica threshold de confianca, piso minimo de RMS e cooldown entre comandos |
| 4 - Atuacao | baixa (1) | Aciona o servo correspondente e o LED de alerta |

Sincronizacao entre tasks:

- 3 filas (`audioQueue`, `featuresQueue`, `commandQueue`) implementam o
  padrao produtor/consumidor entre tasks consecutivas, sem *polling*. Cada
  task fica bloqueada em `xQueueReceive` ate ter dado novo.
- 2 semaforos binarios (`bufferFreeSemaphore[0]` e `[1]`) protegem o
  buffer duplo de captura. A Task 1 so pode escrever num buffer depois que
  a Task 2 sinalizar (via `xSemaphoreGive`) que terminou de le-lo. Sem essa
  protecao haveria condicao de corrida entre a escrita continua do I2S e a
  leitura para extracao de features.
- 1 mutex (`latencyMutex`) protege o unico recurso realmente compartilhado
  fora do fluxo das filas: uma struct (`LatencyLog`) com os timestamps de
  cada etapa, escrita tanto pela Task 3 quanto pela Task 4.

A prioridade da Task 1 e a mais alta porque a captura de audio nao pode
perder amostras. As demais tasks toleram pequenas variacoes de tempo sem
comprometer o pipeline.

## 3. Pipeline de deteccao

### 3.1 Extracao de features

O vetor de entrada do modelo tem 14 valores: RMS, Zero-Crossing Rate,
Centroide Espectral e 11 coeficientes MFCC.

RMS e ZCR sao calculados diretamente no dominio do tempo. Centroide e MFCC
exigem uma FFT. Em vez de usar os parametros padrao do `librosa` (FFT de
2048 pontos, 128 bandas mel, pensados pra uso em desktop), implementei uma
FFT radix-2 propria de 512 pontos com 26 bandas mel, dimensionada pra
rodar em tempo real num microcontrolador. Escrevi a mesma formula duas
vezes: uma em Python (`training/dsp.py`, usada no treino) e outra em C
(`firmware/dino_voice_controller/fft.cpp` e `feature_extraction.cpp`,
usada no dispositivo). Conferi as duas implementacoes numericamente uma
contra a outra, nao contra o `librosa` — o objetivo era garantir que o que
roda no ESP32 calcula exatamente o que o modelo aprendeu no treino, nao
reproduzir uma biblioteca externa. Diferenca maxima observada: da ordem de
1e-4 a 1e-7 em testes com tons puros e com clipes reais do dataset.

Antes de calcular centroide e MFCC, o sinal e normalizado pelo pico de
amplitude, o que deixa essas duas features invariantes ao ganho do
microfone/dispositivo de gravacao. O RMS e calculado sobre o sinal bruto,
sem essa normalizacao, porque e a feature que carrega informacao de
volume, necessaria pra distinguir fala de silencio/ruido de fundo.

### 3.2 Modelo

Rede neural totalmente conectada pequena: 14 (entrada) -> 16 (oculta, ReLU)
-> 3 (saida, softmax). Total de 291 parametros (~1.16KB em float32).
Treinada em Python com PyTorch e exportada para `.onnx`
(`models/model.onnx`).

O ESP32 nao tem um runtime ONNX facil de usar junto com FreeRTOS/Arduino,
entao portei o forward-pass manualmente pra C como produto de matrizes
(`firmware/dino_voice_controller/model_inference.cpp`), usando os pesos
extraidos do `.onnx` (`training/export_c_model.py`). Verifiquei a porta
comparando os logits calculados em C com os calculados pelo
`onnxruntime` em Python, pro mesmo vetor de entrada: diferenca na ordem de
1e-5.

Dado o tamanho da rede (291 parametros), nao fiz quantizacao: o ganho de
memoria seria menor que 1KB, irrelevante frente aos ~520KB de RAM do
ESP32, e o forward-pass ja roda em microssegundos em ponto flutuante.

## 4. Metodologia experimental

### 4.1 Dataset

O dataset final usado no modelo em producao tem 142 clipes de audio de 1
segundo (16kHz, mono, PCM 16 bits), gravados diretamente pelo microfone
INMP441 do dispositivo final, via um utilitario de gravacao proprio
(`firmware/esp32_record/esp32_record.ino` +
`training/record_from_esp32.py`, comunicacao por Serial):

- 49 clipes de "pular"
- 43 clipes de "abaixa"
- 50 clipes de "ruido" (silencio, ruido de fundo, outras falas)

O dataset e ampliado por augmentacao (ruido aditivo, variacao de pitch,
*time-stretch*, variacao de ganho, deslocamento temporal) antes do treino,
sempre aplicada *depois* da separacao treino/teste, pra nao vazar
informacao entre os dois conjuntos (um clipe aumentado nunca aparece em
treino e teste ao mesmo tempo).

Datasets intermediarios (audio gravado por iPhone/notebook e dados
publicos do MLCommons Multilingual Spoken Words Corpus) foram usados em
etapas anteriores do projeto e estao documentados no diario de bordo.
Foram descartados do modelo final porque um experimento controlado (secao
6) mostrou que misturar fontes de microfone diferentes prejudicava a
generalizacao pro hardware real.

### 4.2 Divisao treino/teste

Separacao 80/20 (`training/dataset.py`, funcao `split_raw_clips`), feita
sobre os clipes brutos, antes de qualquer augmentacao, com estratificacao
por classe. Seed fixa (42) em todas as etapas aleatorias (split,
inicializacao dos pesos da rede) pra tornar o resultado reprodutivel.

### 4.3 Codigo de teste

- **Avaliacao offline (treino):** `training/train.py` treina e avalia o
  modelo, gerando a matriz de confusao e o relatorio de precisao/recall/F1
  usados na secao 6 (`models/evaluation.txt`). Essa avaliacao usa o
  argmax direto do modelo, sem os filtros de decisao do firmware.
- **Script de teste (simula anomalias e mede performance):**
  `training/test_detector.py` roda separado do fluxo de treino. Ele pega o
  mesmo conjunto de teste (clipes reais do INMP441, held-out, nunca vistos
  no treino) e simula cada um chegando no detector: mede o tempo de
  extracao de features e de inferencia separadamente (`time.perf_counter`),
  e aplica a mesma logica de decisao do firmware (piso de RMS + limiar de
  confianca) em vez do argmax puro. Gera `models/test_report.txt` com
  latencia por etapa, relatorio de classificacao, matriz de confusao e duas
  metricas voltadas pra deteccao de anomalia: taxa de deteccao (recall
  combinado de "pular"+"abaixa") e taxa de falso positivo ("ruido"
  classificado como comando). Roda com `python -m training.test_detector`.
- **Teste ao vivo:** `training/live_test.py` roda o modelo em tempo real
  pelo microfone do computador (janela deslizante de 1s, classificacao
  continua), util pra depurar rapido sem precisar regravar o firmware.
- **Suite automatizada:** 65 testes unitarios (`training/tests/`, `pytest`)
  cobrindo extracao de features, augmentacao, dataset, treino, exportacao
  pra ONNX/C, o utilitario de gravacao e o script de teste.

## 5. Analise de latencia

Cada etapa do pipeline e medida com `micros()`/`millis()` e registrada numa
struct compartilhada (protegida por mutex, ver secao 2), impressa via
Serial a cada comando detectado. Exemplo real, capturado em bancada:

```
[PULAR] captura->features=84838us  features->deteccao=45us  deteccao->atuacao=230030us  total=314901us
```

| Etapa | Tempo tipico | Composicao |
|---|---|---|
| Captura -> Features | ~85 ms | FFT (512 pontos) + banco de filtros mel + DCT, repetidos pra ~61 janelas dentro do clipe de 1s |
| Features -> Deteccao | < 1 ms | Forward-pass da rede (291 parametros), custo desprezivel frente as demais etapas |
| Deteccao -> Atuacao | ~230 ms a ~1080 ms | Pulso do LED de alerta (80ms) + tempo que o servo fica pressionado (parametro ajustavel por comando: 300ms pra "pular", 1000ms pra "abaixa") |

A etapa dominante em tempo de processamento e a extracao de features
(FFT), mas o maior componente de latencia percebida pelo usuario nem
aparece nessa tabela: a Task 1 so envia um buffer pra processamento
depois de acumular o segundo inteiro de audio. Na pior das hipoteses, o
sistema so comeca a processar ate 1 segundo depois do inicio da fala, e
esse tempo de captura domina a latencia total, mais do que qualquer etapa
de calculo.

## 6. Resultados

Avaliacao offline no conjunto de teste (dados reais do INMP441, 29 clipes,
nunca vistos durante o treino nem gerados por augmentacao a partir de
clipes de treino):

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

"Pular" tem precisao de 100% nesse conjunto de teste (nenhum outro clipe
foi classificado como "pular" por engano). O erro mais frequente e
"abaixa" confundido com "ruido" (3 dos 9 casos), o que no fim das contas e
um erro relativamente seguro do ponto de vista do produto: resulta em
nenhuma acao, em vez de acionar o servo errado.

Essa avaliacao usa o argmax puro do modelo. Rodando o mesmo conjunto de
teste pelo `training/test_detector.py` (secao 4.3), que aplica tambem o
piso de RMS e o limiar de confianca do firmware, a acuracia cai pra 79% no
mesmo conjunto de 29 clipes, porque o piso de RMS (calibrado em bancada, ao
vivo) rejeita 15 dos 29 clipes do teste offline antes mesmo de chegar no
modelo. Isso mostra que o valor de `MIN_RMS_FLOOR` esta mais agressivo pro
volume de gravacao usado no dataset do que pro uso ao vivo real, e fica
registrado como algo a recalibrar (ver limitacoes, secao 7).

Fiz tambem um experimento comparativo, descrito em detalhe no diario de
bordo: treinei o mesmo pipeline com um dataset publico em ingles (Google
Speech Commands, palavras "up"/"down") e obtive 86% de acuracia num
conjunto de teste de 101 amostras. Esse resultado, mais alto e mais
confiavel estatisticamente (dataset maior, gravado por milhares de
falantes), sugere que a principal limitacao do modelo atual e o tamanho
do meu dataset proprio, nao a arquitetura do pipeline em si.

## 7. Discussao

O desafio tecnico central do projeto nao foi a arquitetura RTOS, que
funcionou como planejado desde a primeira versao, e sim garantir que os
dados de treino representassem o microfone do dispositivo final. Um
modelo treinado com audio de outros microfones (celular, notebook,
dataset publico) apresentava um vies forte e consistente pra uma unica
classe quando testado ao vivo pelo INMP441, mesmo com acuracia alta na
avaliacao offline (que usava dados da mesma fonte de microfone do treino).
Um experimento controlado, treinar e testar usando exclusivamente dados
gravados pelo proprio ESP32, eliminou esse vies quase por completo,
confirmando que a causa era descasamento de dominio, e nao um erro de
modelo ou de features.

Outro ponto que apareceu nos testes: em situacoes com silencio digital
"perfeito" (algo que nunca esteve no dataset de treino, ja que toda
gravacao real tem algum ruido de fundo), a rede classificava esse silencio
como um comando valido com mais de 99% de confianca. E um limite conhecido
de redes neurais pequenas sem nenhum mecanismo de rejeicao explicito: alta
confianca em entradas fora da distribuicao vista no treino. A mitigacao
adotada foi um piso minimo de RMS (calibrado empiricamente em bancada, ver
`MIN_RMS_FLOOR` no firmware) que rejeita qualquer janela abaixo de um
limiar de energia antes mesmo de rodar o modelo.

Tambem vale registrar um problema de qualidade de dado publico: uma
auditoria manual (com apoio de transcricao automatica via Whisper) sobre o
Multilingual Spoken Words Corpus revelou uma taxa de erro de alinhamento
maior do que eu esperava, clipes rotulados como uma palavra que na
verdade continham outra coisa, ou so ruido. Isso reforça que dado publico
rotulado automaticamente precisa de auditoria antes de entrar num dataset
de treino, mesmo vindo de uma fonte reconhecida.

Limitacoes conhecidas:

- Dataset final pequeno (142 clipes brutos) pra um problema de
  reconhecimento de fala; a acuracia de 83% tem variancia relevante dado o
  tamanho do conjunto de teste (29 amostras).
- O piso de RMS usado pra rejeitar silencio/ruido e um valor fixo,
  calibrado manualmente num ambiente especifico; pode exigir reajuste em
  outras condicoes de ruido ambiente ou outro posicionamento do microfone.
  O `training/test_detector.py` (secao 4.3) ja mostrou isso na pratica: ele
  rejeitou metade do conjunto de teste offline (15 de 29 clipes) com o
  valor atual de `MIN_RMS_FLOOR`.
- "Pular" e "abaixa" ainda apresentam confusao entre si em falas mais
  rapidas ou mais baixas.

Como trabalho futuro, o que mais ajudaria e ampliar o dataset gravado
diretamente pelo hardware final, ja que foi o que mais melhorou o
resultado ate agora. Tambem daria pra fazer uma segunda rodada de
calibracao do piso de RMS e do limiar de confianca com mais dados de
bancada, e investigar se caracteristicas adicionais de temporalidade
(sem cair no problema de dimensionalidade ja observado com a segmentacao
testada e descartada) ajudam a separar melhor "pular" de "abaixa".

## 8. Estrutura do repositorio

- `firmware/` - codigo do ESP32 (Arduino): sketch principal
  (`dino_voice_controller/`), teste isolado de microfone (`i2s_mic_test/`)
  e utilitario de gravacao de dados (`esp32_record/`).
- `training/` - pipeline de treino em Python (features, augmentacao,
  treino, exportacao pra ONNX e pra C), com testes automatizados.
- `models/` - modelo treinado (`model.onnx`), estatisticas de normalizacao
  usadas pra padronizar as features, e relatorio de avaliacao.
- `docs/` - este relatorio, o diario de bordo, o diagrama RTOS e a spec de
  arquitetura original.

### 8.1 Papel de cada arquivo em `training/`

| Arquivo | Papel |
|---|---|
| `features.py`, `dsp.py`, `augment.py`, `dataset.py`, `train.py` | nucleo do pipeline de treino, usados uns pelos outros |
| `export_c_model.py`, `export_c_dsp.py` | geram os headers `.h` consumidos pelo firmware - scripts de linha de comando |
| `record_audio.py` | nao faz mais parte do fluxo principal (o dataset final e so do ESP32), mas `record_from_esp32.py` reaproveita a funcao `save_clip` dele - ainda e uma dependencia real, nao codigo morto |
| `record_from_esp32.py`, `live_test.py`, `test_detector.py` | utilitarios de linha de comando (`python -m training.<nome>`), nao importados por outros modulos, mas nao sao codigo morto - sao os pontos de entrada usados durante o desenvolvimento e o script de teste pedido pela ponderada |

## 9. Referencias e licencas

- Multilingual Spoken Words Corpus (MLCommons), licenca CC-BY 4.0,
  <https://mlcommons.org/datasets/multilingual-spoken-words/>. Usado numa
  etapa intermediaria do projeto (ver diario de bordo), nao faz parte do
  dataset do modelo final.
- Google Speech Commands Dataset v0.02, licenca CC-BY 4.0. Usado no
  experimento comparativo da secao 6.

## 10. Demonstracao

### Montagem fisica do hardware

![Montagem completa: notebook com o jogo do dinossauro aberto, os dois servos posicionados sobre o teclado e o circuito na protoboard abaixo](images/IMG_1536.jpg)

*Legenda: visao geral da bancada de teste - notebook com o jogo do
dinossauro do Chrome aberto, os dois servos SG90 apoiados sobre o teclado
(um proximo a barra de espaco, outro proximo a seta para baixo) e o
circuito na protoboard, conectado por USB.*

![Detalhe do circuito na protoboard: ESP32-WROOM-32U, microfone INMP441 e fiacao](images/IMG_1537.jpg)

*Legenda: detalhe do circuito - modulo ESP32-WROOM-32U, microfone I2S
INMP441 (modulo redondo) e a fiacao de sinal dos dois servos.*

![Detalhe dos dois servos SG90 montados numa tira de MDF sobre o teclado](images/IMG_1538.jpg)

*Legenda: os dois servos SG90 (Tower Pro) fixados numa tira de MDF apoiada
sobre o teclado do notebook, cada um posicionado para pressionar uma tecla
diferente do jogo.*

![Outro angulo da montagem, mostrando servos e protoboard juntos](images/IMG_1539.jpg)

*Legenda: outro angulo da montagem, mostrando os servos e o circuito
completo na mesma foto.*

### Video: demonstracao do sistema em funcionamento

[Assista ao video (`docs/videos/demonstracao.mp4`)](videos/demonstracao.mp4)

<video src="videos/demonstracao.mp4" controls width="480"></video>

*Legenda: video mostrando o jogo do dinossauro do Chrome sendo controlado
por voz em tempo real, via ESP32 + servos.*
