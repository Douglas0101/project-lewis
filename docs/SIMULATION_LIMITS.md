# Limites da Simulação — Project-Lewis v1.3-sim-deep

> Este documento descreve os limites conhecidos da validação por simulação Renode e deve ser revisado antes de qualquer decisão baseada nos resultados obtidos sem hardware físico.

---

## 1. Sem Hardware Físico

Toda a validação do firmware `v1.3-sim-deep` foi executada no emulador **Renode 1.15.3** com o modelo de placa STM32F4Discovery. Não houve execução em silício real, portanto:

- Timings absolutos são *representativos*, não garantidos.
- Comportamento de periféricos reais (ADC, DMA, PLL, watchdog independente) pode divergir.
- Consumo de energia **não é medido** nem estimado nesta versão (débito técnico para v1.4).

---

## 2. Latências Determinísticas

O Renode emula ciclos de CPU e periféricos de forma determinística, sem jitter de cache nem variação de temperatura/voltagem. Isso significa que:

- A latência medida de inferência (~16 ms/batimento no modelo atual) é reprodutível entre execuções.
- A latência real no STM32F407 pode variar em função de condições elétricas e térmicas.

---

## 3. Divergência de 1 LSB entre CMSIS-NN e Kernels de Referência

O build ARM linka a biblioteca TFLM otimizada com **CMSIS-NN**. Os testes de bit-exatidão (QG8) comparam o output int8 do firmware contra o interpretador Python usando o resolver `BUILTIN_REF` (kernels de referência).

- CMSIS-NN pode arredondar acumuladores 32-bit de forma otimizada.
- O QG8 aceita divergência de até **1 LSB** (`np.allclose(..., atol=1)`).
- O QG10 compara a saída do firmware contra ground-truth gerado pelo próprio interpretador Python BUILTIN_REF, também dentro da tolerância de 1 LSB.

---

## 4. UART e Timeout de Frame

O teste de fidelidade QG10 envia 500 amostras float32 (2000 bytes) pela UART emulada byte a byte para evitar overflow da FIFO. Para acomodar essa transmissão lenta:

- `UART_FRAME_TIMEOUT_MS` foi configurado para **60 000 ms** no modo `RENODE_SIMULATION`.
- O teste Robot `fidelity.robot` aguarda a resposta por até **60 s**.

No hardware real, a UART a 115200 kbps transmitiria o mesmo frame em ~175 ms; o timeout de 60 s é uma concessão exclusiva da simulação.

---

## 5. Stub de ADC e Sinais de Teste

A aquisição real de ECG ainda não foi integrada. O firmware usa:

- `adc_stub.c` para gerar batimentos sintéticos determinísticos no boot.
- Frames binários pela UART para o QG10.

A correspondência entre o stub em C e a referência Python (`tests/fixtures/adc_stub.py`) é verificada por bit-exatidão.

---

## 6. Detector de R-peak Não está no Caminho Crítico

A função `lewis_detect_r_peaks()` e o comando `PEAK` existem e são validados pelo QG18, mas a segmentação por R-peak **não** é usada para alimentar o modelo no caminho principal de inferência. O modelo recebe janelas de 500 amostras pré-segmentadas (stub ou UART).

---

## 7. Recomendações para Uso em Hardware Real

Antes de promover o firmware para testes em silício:

1. Validar o driver do ADC real e a taxa de amostragem de 500 Hz.
2. Substituir o stub de ADC pela aquisição real e calibrar ganhos.
3. Medir latência e consumo de energia com instrumentação adequada.
4. Ajustar `UART_FRAME_TIMEOUT_MS` para um valor realista (ex.: 1000 ms).
5. Avaliar a robustez do detector de R-peak em sinais reais e ruídos de eletrodos.
