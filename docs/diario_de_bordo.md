# Diario de Bordo - Detector de Anomalias Acusticas

**Projeto:** controle por voz do jogo do dinossauro do Chrome, usando ESP32,
INMP441, FreeRTOS e servos. Comandos: "pular" e "abaixa", mais uma classe de
rejeicao ("ruido": silencio, ruido de fundo, outras falas).

Esse documento e o registro de como o desenvolvimento realmente aconteceu,
com os erros e retrabalhos incluidos. Prefiro deixar isso registrado do que
so mostrar a versao final bonitinha. Pra arquitetura, metodologia e
resultados de forma mais direta, tem o [`relatorio_tecnico.md`](relatorio_tecnico.md).

---

## Definindo a ideia e montando o pipeline de treino

Li o enunciado da ponderada e resolvi fugir um pouco dos exemplos dados
(queda, grito, latido). A ideia foi detectar os comandos de voz "pular" e
"abaixa" pra controlar o jogo do dinossauro do Chrome, com um servo motor
apertando o teclado. Pensei nisso como uma aplicacao de acessibilidade:
controlar algo sem usar as maos.

Defini a arquitetura de 4 tasks FreeRTOS (captura, extracao de features,
deteccao, atuacao) e comecei pelo pipeline de treino em Python, seguindo
TDD: `features.py` (RMS, centroide espectral, MFCC via librosa),
`augment.py` (ruido, pitch shift, time stretch), `dataset.py`, `train.py`
(MLP pequeno, exportacao pra ONNX).

Os primeiros audios gravei pelo Voice Memo do iPhone. Como os nomes dos
arquivos saem automaticos por localizacao (nao indicam a classe), rotulei
com ajuda de transcricao automatica via Whisper.

Achei um bug logo no comeco que me preocupou bastante: o `dataset.py`
fazia o split treino/teste *depois* de aplicar augmentacao. Ou seja,
clipes aumentados do mesmo audio original podiam cair tanto no treino
quanto no teste, e a acuracia que eu via estava inflada por vazamento de
dado. Corrigi separando por clipe bruto antes de aumentar.

Outro bug: o `record_audio.py` calculava o proximo indice de arquivo
contando quantos arquivos existiam, em vez de pegar o maior indice e somar
1. Se eu apagasse uma gravacao ruim no meio do caminho, a proxima gravacao
sobrescrevia um arquivo bom sem nenhum aviso.

Sem seed fixa no treino, a acuracia variava bastante entre execucoes (0.77
a 0.92) so por causa da inicializacao aleatoria dos pesos. Botei
`torch.manual_seed(42)` pra parar de ficar adivinhando se uma mudanca
realmente ajudou ou se foi sorte da rodada.

Fui adicionando mais dado aos poucos, inclusive trechos de conversa longa
mencionando "pular"/"abaixa" de proposito (queria ensinar o modelo a nao
disparar so por ouvir a palavra no meio de uma frase), e testando ao vivo
pelo microfone do notebook (`live_test.py`). A acuracia foi subindo aos
poucos: 73%, depois 76%, 78%, 80%, 83%, conforme o dataset crescia e
ficava mais limpo.

## Descasamento de microfone: o problema que dominou o projeto

Reparei num problema que se repetia sempre: o modelo treinado com audio do
iPhone e do notebook, quando eu testava ao vivo, tinha um vies forte pra
uma classe so (quase tudo virava "pular", mesmo quando eu falava
"abaixa"). Isso e descasamento de dominio - o modelo aprende
caracteristicas do microfone de treino, nao so o conteudo da fala.

Tentei mitigar de algumas formas. Normalizar a amplitude do sinal antes de
calcular centroide/MFCC, pra deixar essas features invariantes ao ganho do
microfone. Aumentar a augmentacao com variacao de ganho e deslocamento
temporal. E fui atras de dado de multiplos falantes/microfones: baixei o
Multilingual Spoken Words Corpus (MLCommons), que extrai palavras isoladas
do Mozilla Common Voice por alinhamento forcado. Achei 51 clipes de "pula"
e 46 de "abaixo" (a variante mais proxima de "abaixa" que tinha no
corpus).

Tambem tentei engenharia de features: dividir o audio em segmentos
temporais antes de extrair as features, achando que ia capturar melhor a
evolucao da palavra no tempo. Piorou a acuracia (74-77% contra 80% sem
segmentacao) - o dataset era pequeno demais pra aguentar mais dimensao.
Revertido. Ja o zero-crossing rate como feature isolada ajudou (subiu pra
83%). A licao real aqui foi testar uma mudanca por vez - quando empilhei
varias juntas, ficou dificil saber o que realmente ajudou.

## Comecando o firmware

Defini a pinagem do ESP32 + INMP441 (I2S) e dos servos, e escrevi um
sketch de teste isolado do microfone (`i2s_mic_test.ino`) antes de montar
a arquitetura completa - queria confirmar que o hardware funcionava antes
de complicar.

Depois portei o pipeline inteiro pra C. Duas coisas deram trabalho aqui.
Primeiro, a FFT: o librosa usa 2048 pontos e 128 bandas mel por padrao,
pesado demais pra rodar em tempo real num microcontrolador. Reescrevi o
calculo de centroide/MFCC em Python (`training/dsp.py`) com parametros
menores (512 pontos, 26 bandas mel), e so depois portei a mesma logica
pra C, conferindo numericamente que as duas batiam (erro relativo na casa
de 1e-4 a 1e-7). O forward-pass da rede (291 parametros) foi mais
tranquilo, so produto de matrizes.

Segundo problema: quando juntei tudo na arquitetura de 4 tasks, o
firmware simplesmente nao coube na RAM do ESP32 (faltavam uns 98KB).
Resolvi trocando os buffers de captura de float pra int16 (padrao PCM,
metade do tamanho) e tirando uma copia intermediaria de 64KB que nao
precisava existir.

## O mesmo problema de microfone volta, agora no hardware real

Com o firmware rodando de verdade, o mesmo vies apareceu de novo - so que
agora entre notebook/iPhone (treino) e o INMP441 real (uso). O problema
nao tinha sido resolvido, so tinha mudado de lugar. Montei um jeito de
gravar dado direto pelo ESP32 (`esp32_record.ino` + `record_from_esp32.py`,
comunicacao via Serial) e comecei a trocar parte do dataset por gravacao
real do hardware final.

Nessa fase apareceram uns bugs de hardware tambem: o servo nao se movia
porque tava ligado direto no pino de alimentacao do ESP32 (corrente
insuficiente - precisa de fonte externa de 5V), e o `live_test.py` chegou
a classificar silencio digital puro como comando com 99% de confianca (o
modelo nunca tinha visto esse caso extremo no treino).

## Limpeza de dados e o experimento que resolveu

Fiz uma auditoria completa do dataset com Whisper e descobri que boa
parte dos dados do MSWC estava mal alinhada - a palavra transcrita nao
era a real, ou o audio so tinha ruido. Removi uns 50 clipes contaminados.

Ai fiz o experimento que realmente esclareceu as coisas: treinar e testar
usando *so* dado gravado pelo proprio ESP32, sem misturar iPhone, notebook
ou MSWC. "Pular" chegou a 100% de precisao. Isso confirmou que o
descasamento de microfone era mesmo a causa principal do problema, nao um
erro de modelo ou de features. Adotei esse dataset (142 clipes, so do
ESP32) como o oficial.

## Um desvio: testei em ingles tambem

Testei um experimento em ingles com o Google Speech Commands
("up"/"down"), um dataset publico feito especificamente pra esse tipo de
tarefa. Deu 86% de acuracia num teste de 101 amostras, bem mais robusto
estatisticamente do que qualquer coisa que eu consegui gravar sozinho.
Cheguei a trocar o projeto pra usar "up"/"down" por causa disso, mas
acabei voltando pro portugues ("pular"/"abaixa") gravado no ESP32 -
preferi manter o projeto no idioma original.

## Ajuste fino no hardware

Por ultimo, ajustei os parametros fisicos dos servos direto na bancada:
angulo de giro, tempo que ficam pressionados, direcao de rotacao de cada
um. Foi bastante tentativa e erro ate o movimento ficar adequado pra
apertar as teclas sem forcar demais o mecanismo.

## O que eu levo desse projeto

Descasamento de microfone entre treino e uso real foi o problema tecnico
central do projeto - muito mais do que a escolha do modelo ou das
features. Dado real do hardware final vale mais do que dado limpo vindo
de outro lugar, por mais dado que seja.

Mudar uma variavel de cada vez foi o que realmente permitiu entender o
que ajudava de verdade. Toda vez que empilhei varias mudancas juntas,
ficou dificil saber depois o que causou a melhora (ou a piora).

Dado publico rotulado automaticamente merece uma conferida antes de
entrar no treino, mesmo vindo de uma fonte conhecida - o alinhamento
forcado do MSWC errou mais do que eu esperava.

E redes neurais pequenas sao "confiantemente erradas" fora da
distribuicao de treino. Silencio digital puro, que nunca apareceu no meu
dataset, foi classificado com altissima confianca como se fosse um
comando de verdade. Por isso acabei colocando uma camada de protecao bem
simples (piso de energia, limiar de confianca) alem do modelo em si.

Arquitetura, latencia medida, resultados finais, limitacoes conhecidas e
estrutura do repositorio estao descritos com mais detalhe no
[`relatorio_tecnico.md`](relatorio_tecnico.md).
