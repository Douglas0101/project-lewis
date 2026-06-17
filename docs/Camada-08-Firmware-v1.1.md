# Project-Lewis — Camada 8: Firmware Embarcado

**Responsável:** Firmware / Embedded ML Engineering  
**Versão:** 1.1 | **Data:** 2026-06-14 | **Arquiteto:** Douglas Souza

---

## 8.1 Objetivo

Implementar o firmware C para classificação de arritmias em STM32F4 usando TensorFlow Lite Micro (TFLM), com pipeline de sinal completo (aquisição → DSP → segmentação → normalização → quantização → inferência → pós-processamento).

O firmware deve ser compilável sem hardware físico, executando em ambiente de emulação realista (Renode) descrito na Camada 9.

---

## 8.2 Hardware Alvo

| Parâmetro | Valor |
| :--- | :--- |
| MCU | STM32F407VG (Cortex-M4F) |
| Flash | 1 MB |
| SRAM | 128 KB + 64 KB CCM |
| FPU | Single-precision hardware |
| Clock | 168 MHz |
| Debug | UART4 (ST-LINK VCP na Discovery) |

---

## 8.3 Estrutura do Firmware

```
firmware/
├── src/
│   ├── app/main.c                 # entry point e loop de inferencia
│   ├── app/command_loop.c         # parser de comandos UART (RUN, SHUTDOWN, PEAK, ...)
│   ├── platform/startup_stm32f4.c # vetor de reset, init .data/.bss, habilita FPU
│   ├── hal/hal.h                  # API de abstracao de hardware
│   ├── hal/simulator/hal_sim.c    # tempo/delay para Renode e host nativo
│   ├── hal/target/uart_stm32f4.c  # UART4 driver para STM32F4
│   ├── hal/native/uart_host.c     # UART stub para host nativo
│   ├── utils/debug.c/h            # prints de debug sem newlib printf
│   ├── dsp/adc_stub.c/h           # gera batimentos de teste
│   ├── dsp/filter.c/h             # filtros bandpass/notch (biquad cascata)
│   ├── dsp/normalizer.c/h         # normalizacao Z-score
│   ├── dsp/r_peak_detector.c/h    # detector leve de R-peaks
│   ├── ml/inference.c/h           # wrapper TFLM
│   └── ml/{model_data,quantization_params}.h  # gerados pelo export
├── renode/                        # scripts de plataforma e testes Renode
├── build/                         # artefatos de build
├── tools/                         # toolchain ARM e Renode (instalados localmente)
└── third_party/tflite-micro/      # fonte do TFLM
```

---

## 8.4 Decisões Arquiteturais

### 8.4.1 Ausência de `printf`

`printf` e semihosting dependem de syscalls que, sem configuração cuidadosa, travam em bare-metal. A pilha de debug implementa funções simples (`lewis_debug_print`, `lewis_debug_print_uint`, `lewis_debug_print_int`, `lewis_debug_print_hex`) que escrevem diretamente na UART4, sem depender do newlib.

### 8.4.2 FPU Habilitada no Reset

O Cortex-M4F requer que o coprocessador de ponto flutuante seja habilitado via `CPACR` antes de qualquer instrução VFP. O `Reset_Handler` configura `CP10/CP11 = Full access` em `0xE000ED88`.

### 8.4.3 UART4 como Canal de Debug

A STM32F4Discovery conecta UART4 ao conversor USB-Serial do ST-LINK. O Renode emula essa UART e permite capturar a saída em arquivo (`CreateFileBackend`).

### 8.4.4 HAL Simulada

A camada `hal/simulator` fornece `millis()` e `delay_ms()` para Renode e host nativo. No STM32F4 emulado o tempo é derivado do SysTick como contador livre decrescente (`LEWIS_SYSTICK_HZ = 168 MHz`), convertendo wraps de 24 bits em milissegundos; no host nativo usa `clock_gettime`/`usleep`. O benchmark de inferência também usa SysTick para medir ciclos de CPU (`lewis_hal_benchmark_start/stop`), o que produz latências representativas no Renode (por exemplo, ~16 ms por batimento no modelo atual).

### 8.4.5 Stub de ADC

`adc_stub.c` gera batimentos sintéticos determinísticos sem `libm` (usa lookup table de seno e exponencial aproximada). A mesma função de geração de sinal é reutilizada nos testes Python via `tests/fixtures/adc_stub.py`, eliminando a duplicação manual entre C e Python e garantindo bit-exatidão comparável no QG8.

---

## 8.5 Pipeline de Inferência

1. **Aquisição:** leitura de 500 amostras @ 500 Hz (no emulador, via stub ou frame UART).
2. **Pré-processamento DSP:** filtro passa-banda 0.5–40 Hz + notch 60 Hz (**implementado**).
3. **Normalização:** Z-score por janela (**implementado**).
4. **Quantização de input:** `int8 = round(float / input_scale) + input_zero_point`.
5. **Inferência TFLM:** `TfLiteInvoke` com arena estática de 64 KB (ativação via `LEWIS_USE_TFLM=1`).
6. **Dequantização de output:** `float = (int8 - output_zero_point) * output_scale`.
7. **Classificação final:** argmax.

> **Nota:** a segmentação por R-peak ainda não está no caminho crítico de inferência. Um detector leve (`lewis_detect_r_peaks`) está disponível via comando `PEAK` e é validado pelo QG18.

A medição de latência é feita com `lewis_hal_benchmark_start()` / `lewis_hal_benchmark_stop()` sobre o SysTick do Cortex-M, capturando ciclos de CPU reais no Renode.

---

## 8.6 Integração com TFLM

O TFLM é adicionado como fonte em `firmware/third_party/tflite-micro/`. O build produz `libtensorflow-microlite.a` via o Makefile oficial do TFLM, que é depois linkado ao firmware.

Operadores registrados (mínimo viável):

- `Conv2D` / `DepthwiseConv2D`
- `FullyConnected`
- `MaxPool2D` / `AveragePool2D`
- `Softmax`
- `Reshape`
- `Quantize` / `Dequantize`

A arena TFLM é um array estático alinhado a 16 bytes de 64 KB. O build padrão de firmware (`LEWIS_USE_TFLM=1`) já linka o interpretador TFLM real; o modo stub (`LEWIS_USE_TFLM=0`) ainda existe apenas para validação estrutural rápida no host nativo.

---

## 8.7 Build

```bash
# Host nativo (validacao rapida de sintaxe/logica)
make firmware-native
./firmware/build/native/lewis

# STM32F4 (elf + bin) com TFLM real
make -C firmware LEWIS_USE_TFLM=1 firmware-build

# Tamanho
arm-none-eabi-size firmware/build/stm32f4/lewis.elf
```

---

## 8.8 Quality Gates

| Gate | Critério | Como Validar |
| :--- | :--- | :--- |
| QG7 | Build sem warnings (`-Werror`); FlatBuffer < 64 KB | `make firmware-build` |
| QG8 | Bit-exatidão int8 vs interpretador Python (BUILTIN_REF) | `pytest -m qg8` |
| QG9 | Latência < 200 ms/batimento @ 168 MHz; RAM TFLM < 64 KB; Flash < 512 KB | Métricas do Renode |
| QG10 | Fidelidade numérica vs ground-truth (`cosine > 0.99`) | `pytest -m qg10` |
| QG13 | Watchdog software de inferência reseta após timeout | `pytest -m qg13` |
| QG16 | Filtros bandpass/notch C vs Python | `pytest -m qg16` |
| QG17 | Pipeline filtrado C vs Python | `pytest -m qg17` |
| QG18 | Detector leve de R-peak C vs AMPT Python | `pytest -m qg18` |

---

## 8.9 Notas de Implementação

- O modelo INT8 gera `model_data.h` e `quantization_params.h` em `models/quantized/`; o build os copia para `firmware/src/ml/`.
- O build padrão de produção usa TFLM real (`LEWIS_USE_TFLM=1`). O stub (`LEWIS_USE_TFLM=0`) permanece disponível para iteração rápida no host nativo.
- Semihosting foi evitado propositalmente por fragilidade em emuladores; a comunicação é via UART4 real do STM32F4.

## 8.10 Limitações Conhecidas / Débito Técnico

1. **`delay_ms()` usa busy-wait**: tanto no Renode quanto no target, `lewis_hal_delay_ms()` é uma espera ocupada baseada em `lewis_hal_millis()`. Isso consome 100% da CPU durante o delay e pode ser substituído por WFI/IRQ de timer em versões futuras.
2. **`millis()` depende do SysTick no Renode**: embora monotônico, o contador é derivado de wraps de 24 bits do SysTick. A leitura de `COUNTFLAG` limpa a flag, portanto aplicações que chamam `millis()` com intervalos muito espaçados (> ~100 ms) não perdem wraps, mas a precisão absoluta depende da configuração de clock do emulador.
3. **Pipeline de sinal completo até a inferência**: filtros passa-banda/notch e normalização Z-score estão integrados. A detecção de R-peak existe como função auxiliar (`lewis_detect_r_peaks`) e comando `PEAK`, mas ainda não é usada para segmentação no caminho principal de inferência.
4. **Watchdog e timeout de inferência**: watchdog software protege a chamada `Invoke()` (QG13). O timeout de frame UART foi estendido para 60 s no modo Renode para acomodar a transmissão byte-a-byte do QG10.
5. **Duplicação C/Python resolvida via fixture**: a geração do sinal de teste foi unificada em `tests/fixtures/adc_stub.py`, consumida pelos testes QG8/QG10. A consistência com `firmware/src/dsp/adc_stub.c` é verificada por comparação de output.
6. **Sem modelagem de energia**: não há estimativa de consumo no relatório de simulação. Débito técnico detalhado em [`DEBITO_TECNICO_Energia_Renode-v1.4.md`](DEBITO_TECNICO_Energia_Renode-v1.4.md); especificação de solução em [`Camada-09-Energia-v1.4.md`](Camada-09-Energia-v1.4.md).
