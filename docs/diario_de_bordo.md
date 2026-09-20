# Diario de Bordo - Detector de Anomalias Acusticas

**Projeto:** controle por voz do jogo do dinossauro do Chrome, usando ESP32 +
INMP441 + FreeRTOS + servos. Comandos: "pular" e "abaixa", mais uma classe
de rejeicao ("ruido": silencio, ruido de fundo, outras falas).

Este documento registra o desenvolvimento na ordem em que ele aconteceu,
incluindo os erros e retrabalhos - a ideia e mostrar o processo real, nao so
o resultado final. Para a arquitetura, metodologia e resultados de forma
mais direta, ver [`relatorio_tecnico.md`](relatorio_tecnico.md).

---

## Definindo a ideia e montando o pipeline de treino

Li o enunciado da ponderada e decidi fugir um pouco dos exemplos dados
(queda, grito, latido). A ideia: detectar os comandos de voz "pular" e
"abaixa" para controlar o jogo do dinossauro do Chrome com um servo motor
apertando o teclado - uma aplicacao de acessibilidade (controlar algo sem
usar as maos).

Defini a arquitetura de 4 tasks FreeRTOS (captura, extracao de features,
deteccao, atuacao) e comecei pelo pipeline de treino em Python, com TDD:
`features.py` (RMS, centroide espectral, MFCC via librosa), `augment.py`
(ruido, pitch shift, time stretch), `dataset.py`, `train.py` (MLP pequeno,
exportacao para ONNX).

Gravei os primeiros audios pelo Voice Memo do iPhone e rotulei com apoio de
transcricao automatica (Whisper), ja que os nomes dos arquivos eram
gerados automaticamente por localizacao, sem indicar a classe.

**Primeiro bug serio:** o `dataset.py` fazia o split treino/teste *depois*
de aplicar augmentacao, entao clipes aumentados do mesmo audio original
podiam cair tanto no treino quanto no teste - a acuracia relatada estava
inflada por vazamento de dados. Corrigido separando por clipe bruto antes
de aumentar.

**Segundo bug:** `record_audio.py` calculava o proximo indice de arquivo
contando quantos arquivos existiam, em vez do maior indice + 1. Se alguem
apagasse uma gravacao ruim no meio, a proxima gravacao sobrescrevia um
arquivo bom sem avisar.

Com o modelo treinado sem seed fixa, a acuracia variava bastante entre
execucoes (0.77 a 0.92) so por causa da inicializacao aleatoria dos pesos -
adicionei `torch.manual_seed(42)` pra tornar o treino reprodutivel.

Fui adicionando mais dados aos poucos (incluindo trechos de conversa longa
mencionando "pular"/"abaixa" de proposito, como exemplos dificeis pra
ensinar o modelo a nao disparar so por ouvir a palavra dentro de uma frase)
e testando ao vivo pelo microfone do notebook (`live_test.py`). A acuracia
foi subindo aos poucos, de 73% para 83%, conforme o dataset crescia e
ficava mais limpo.

## Descasamento de microfone: o problema central do projeto

Comecei a notar um problema recorrente: o modelo treinado com audio do
iPhone e do notebook, quando testado ao vivo, tinha um vies forte para uma
das classes (quase tudo virava "pular", mesmo quando eu falava "abaixa").
Isso e um problema classico de descasamento de dominio: o modelo aprende
caracteristicas do microfone de treino, nao so o conteudo da fala.

Tentei mitigar de varias formas:
- Normalizar a amplitude do sinal antes de calcular centroide/MFCC (deixa
  essas features invariantes ao ganho do microfone).
- Aumentar a augmentacao com variacao de ganho e deslocamento temporal.
- Buscar dados de multiplos falantes/microfones: baixei o Multilingual
  Spoken Words Corpus (MLCommons), que extrai palavras isoladas do Mozilla
  Common Voice por alinhamento forcado. Achei 51 clipes de "pula" e 46 de
  "abaixo" (variante mais proxima de "abaixa" no corpus).

Tambem testei engenharia de features: dividir o audio em segmentos
temporais antes de extrair features (na esperanca de capturar a evolucao
da palavra no tempo). Resultado: piorou a acuracia (74-77% contra 80% sem
segmentacao) - o dataset era pequeno demais pra sustentar mais dimensoes.
Revertido. Ja adicionar zero-crossing rate como feature isolada ajudou
(83%) - a licao foi testar uma mudanca de cada vez, nao empilhar varias.

## Comecando o firmware

Criei a pinagem do ESP32 + INMP441 (I2S) e dos servos, e escrevi um sketch
de teste isolado do microfone (`i2s_mic_test.ino`) antes de montar a
arquitetura completa.

Depois portei o pipeline inteiro pra C:
- FFT radix-2 propria (512 pontos) - librosa usa 2048 pontos e 128 bandas
  mel por padrao, pesado demais pra rodar em tempo real num
  microcontrolador. Reescrevi o calculo de centroide/MFCC em Python
  (`training/dsp.py`) com parametros menores (512 pontos, 26 bandas mel) e
  depois portei a mesma logica pra C, verificando numericamente que as duas
  batiam (erro relativo na casa de 1e-4 a 1e-7).
- Forward-pass da rede neural (291 parametros) como produto de matrizes.
- Ao juntar tudo na arquitetura de 4 tasks, o firmware nao coube na RAM do
  ESP32 (faltavam ~98KB). Resolvido trocando os buffers de captura de
  float para int16 (padrao PCM, metade do tamanho) e eliminando uma copia
  intermediaria de 64KB que nao era necessaria.

## O descasamento de microfone volta, agora no hardware real

Com o firmware rodando de verdade, o mesmo vies apareceu de novo, agora
entre notebook/iPhone (treino) e o INMP441 real (uso) - o problema nao
tinha sido resolvido, so mudado de lugar. Construi um caminho pra gravar
dados direto pelo ESP32 (`esp32_record.ino` + `record_from_esp32.py`,
comunicacao via Serial) e comecei a substituir parte do dataset por
gravacoes reais do hardware final.

Tambem apareceram bugs de hardware: o servo nao se movia porque estava
ligado direto no pino de alimentacao do ESP32 (corrente insuficiente -
precisa de fonte externa de 5V), e o `live_test.py` chegou a classificar
silencio digital puro como comando com 99% de confianca (o modelo nunca viu
esse caso extremo no treino).

## Limpeza de dados e o experimento decisivo

Uma auditoria completa do dataset com Whisper revelou que boa parte dos
dados do MSWC estava mal alinhada (a palavra transcrita nao era a real, ou
o audio so tinha ruido) - removi cerca de 50 clipes contaminados.

Fiz entao um experimento controlado: treinar e testar usando *so* dados
gravados pelo proprio ESP32 (sem misturar iPhone/notebook/MSWC). Resultado:
"pular" chegou a 100% de precisao - confirmou que o descasamento de
microfone era mesmo a causa principal do problema, nao um erro de modelo.
Adotei esse dataset (142 clipes, so do ESP32) como o oficial.

## Um desvio: testando em ingles

Testei tambem um experimento em ingles com o Google Speech Commands
("up"/"down"), um dataset publico feito especificamente pra esse tipo de
tarefa: 86% de acuracia num teste de 101 amostras, bem mais robusto
estatisticamente que qualquer coisa que consegui gravar sozinho. Cheguei a
trocar o projeto pra usar "up"/"down", mas decidimos voltar pro portugues
("pular"/"abaixa") gravado no ESP32, priorizando manter o projeto no
idioma original.

## Ajuste fino no hardware

Por fim, ajustei os parametros fisicos dos servos (angulo de giro, tempo
que ficam pressionados, direcao de rotacao de cada um) direto testando na
bancada, ate o movimento ficar adequado pra apertar as teclas sem forcar
demais o mecanismo.

## Licoes aprendidas

- Descasamento de microfone entre treino e uso real foi o problema tecnico
  central do projeto, muito mais do que a escolha do modelo ou das
  features. Dado real do hardware final vale mais do que dado limpo de
  outra fonte.
- Mudar uma variavel de cada vez (uma feature nova, um parametro) foi o que
  permitiu entender o que realmente ajudava - empilhar varias mudancas
  junto tornou dificil saber o que causou uma melhora ou piora.
- Vale a pena auditar dados de datasets publicos antes de confiar neles;
  alinhamento automatico erra mais do que parece.
- Redes neurais pequenas sao "confiantemente erradas" fora da distribuicao
  de treino (ex: silencio digital puro, nunca visto no treino, classificado
  com altissima confianca). Vale ter uma camada de protecao simples (piso
  de energia, limiar de confianca) alem do modelo em si.

Arquitetura, latencia medida, resultados finais, limitacoes conhecidas e
estrutura do repositorio estao descritos com mais detalhe em
[`relatorio_tecnico.md`](relatorio_tecnico.md).
