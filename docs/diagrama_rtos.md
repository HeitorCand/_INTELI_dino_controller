# Diagrama de Tarefas RTOS - Detector de Comandos de Voz

Arquitetura de 4 tasks FreeRTOS rodando no ESP32, implementada em
`firmware/dino_voice_controller/dino_voice_controller.ino`.

## Visao geral: tasks, filas, semaforos e mutex

```mermaid
graph TD
    MIC(["INMP441 (I2S)"])

    subgraph BUF["Buffer duplo (double buffering)"]
        BA["bufferA - int16_t[16000]"]
        BB["bufferB - int16_t[16000]"]
    end

    SEM0{{"Semaforo binario bufferFreeSemaphore[0]"}}
    SEM1{{"Semaforo binario bufferFreeSemaphore[1]"}}

    T1["Task 1 - Captura de Audio<br/>prioridade ALTA<br/>xTaskCreate(prio=3)"]
    T2["Task 2 - Extracao de Features<br/>prioridade MEDIA<br/>xTaskCreate(prio=2)<br/>RMS, ZCR, Centroide Espectral, MFCC"]
    T3["Task 3 - Deteccao<br/>prioridade BAIXA<br/>xTaskCreate(prio=1)<br/>forward-pass do modelo + threshold + cooldown"]
    T4["Task 4 - Atuacao<br/>prioridade BAIXA<br/>xTaskCreate(prio=1)<br/>aciona servo + LED/buzzer"]

    QA[["Fila audioQueue - AudioBufferMsg"]]
    QF[["Fila featuresQueue - FeaturesMsg"]]
    QC[["Fila commandQueue - CommandMsg"]]

    MUT[("Mutex latencyMutex")]
    LOG["LatencyLog (struct compartilhada)"]

    SERVO_P(["Servo Pular - GPIO18"])
    SERVO_A(["Servo Abaixa - GPIO19"])
    LED(["LED - GPIO25"])
    BUZ(["Buzzer - GPIO26"])

    MIC -->|"i2s_read()"| T1
    T1 -->|"xSemaphoreTake"| SEM0
    T1 -->|"xSemaphoreTake"| SEM1
    T1 -->|"escreve"| BA
    T1 -->|"escreve"| BB
    T1 -->|"xQueueSend (ponteiro + indice + timestamp)"| QA
    QA -->|"xQueueReceive"| T2
    T2 -->|"le"| BA
    T2 -->|"le"| BB
    T2 -->|"xSemaphoreGive (libera buffer p/ Task 1 reusar)"| SEM0
    T2 -->|"xSemaphoreGive"| SEM1
    T2 -->|"xQueueSend (14 features + timestamps)"| QF
    QF -->|"xQueueReceive"| T3
    T3 -->|"xQueueSend (classe detectada + timestamps,<br/>so se confiante e fora do cooldown)"| QC
    QC -->|"xQueueReceive"| T4
    T4 --> SERVO_P
    T4 --> SERVO_A
    T4 --> LED
    T4 --> BUZ

    T3 -.->|"xSemaphoreTake/Give"| MUT
    T4 -.->|"xSemaphoreTake/Give"| MUT
    MUT -.-> LOG

    style T1 fill:#ffd6d6
    style T2 fill:#ffe8b3
    style T3 fill:#d6e8ff
    style T4 fill:#d6ffd6
```

## Para que serve cada mecanismo de sincronizacao

| Mecanismo | Entre quem | Por que existe |
|---|---|---|
| Fila `audioQueue` | Task 1 -> Task 2 | Handoff produtor/consumidor do buffer de audio cheio, sem polling - a Task 2 fica bloqueada em `xQueueReceive` ate ter dado novo. |
| Fila `featuresQueue` | Task 2 -> Task 3 | Mesma logica, entregando o vetor de 14 features ja calculado. |
| Fila `commandQueue` | Task 3 -> Task 4 | So recebe mensagem quando um comando e considerado valido (confianca acima do limiar e fora do cooldown) - desacopla a decisao da atuacao mecanica, que e mais lenta. |
| Semaforos binarios `bufferFreeSemaphore[0]` / `[1]` | Task 1 e Task 2 | Impedem que a Task 1 sobrescreva um buffer que a Task 2 ainda nao terminou de ler. Cada buffer so e reaproveitado depois que a Task 2 devolve o semaforo correspondente. Sem isso haveria condicao de corrida entre escrita (I2S) e leitura (extracao de features) no mesmo buffer. |
| Mutex `latencyMutex` | Task 3 e Task 4 | Unico recurso realmente compartilhado fora do fluxo das filas: o registro de latencia (`LatencyLog`) e lido e escrito por mais de uma task, entao precisa de exclusao mutua pra evitar leitura/escrita simultanea inconsistente. |

## Prioridades das tasks

```mermaid
graph LR
    A["Task 1 - Captura<br/>prioridade 3 (alta)"] --> B["Task 2 - Features<br/>prioridade 2 (media)"] --> C["Task 3 - Deteccao<br/>prioridade 1 (baixa)"] --> D["Task 4 - Atuacao<br/>prioridade 1 (baixa)"]
```

A captura de audio tem a prioridade mais alta porque nao pode perder amostras
do I2S (a janela de 1 segundo e continua). A extracao de features vem em
seguida. Deteccao e atuacao dividem a prioridade mais baixa, ja que podem
tolerar um pequeno atraso sem comprometer a captura.
