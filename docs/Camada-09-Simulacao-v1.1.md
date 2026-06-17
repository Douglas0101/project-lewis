# Project-Lewis — Camada 9: Simulação Realista com Renode

**Responsável:** Firmware / QA  
**Versão:** 1.1 | **Data:** 2026-06-14 | **Arquiteto:** Douglas Souza

---

## 9.1 Objetivo

Validar o firmware C do Project-Lewis sem hardware físico, usando emulação fiel de STM32F4 no Renode. A simulação deve produzir métricas transferíveis para o alvo real: tempo de inferência, uso de memória (Flash/RAM), bit-exatidão do output e robustez do pipeline.

---

## 9.2 Por que Renode?

| Aspecto | Renode | QEMU puro | Host nativo |
| :--- | :--- | :--- | :--- |
| Modelos de periféricos | Alto (UART, GPIO, timers, ADC) | Médio (requer configuração) | N/A |
| Emulação de ciclo | Sim | Parcial | Não |
| Suporte a Cortex-M4 | Nativo | Possível, mas verboso | N/A |
| CI headless | Sim | Sim | Sim |
| Medição de latência | Sim | Aproximada | Não representativa |

Renode foi escolhido por ser projetado especificamente para microcontroladores embarcados e possuir modelos prontos para STM32F4Discovery.

---

## 9.3 Setup

```bash
# Instala toolchain ARM e Renode localmente (sem sudo)
make firmware-deps

# Verifica instalacao
firmware/tools/arm-gnu-toolchain-13.3.rel1/bin/arm-none-eabi-gcc --version
firmware/tools/renode-1.15.3/renode --version
```

As ferramentas ficam em `firmware/tools/` e são ignoradas pelo Git (instaladas por ambiente).

---

## 9.4 Arquitetura de Simulação

```
firmware/scripts/run_renode_tests.py
        │
        ├── make stm32f4 (com LEWIS_USE_TFLM=1 quando invocado por firmware-test)
        │
        ├── gera .resc temporario
        │   ├── plataforma STM32F4Discovery
        │   ├── carrega lewis.bin @ 0x0800_0000
        │   └── captura UART4 → arquivo temporario de log
        │
        ├── executa Renode headless por N segundos
        │
        ├── parseia log UART
        │
        └── gera firmware/build/stm32f4/firmware_simulation_report.json
```

---

## 9.5 Script de Plataforma

O script principal é `firmware/renode/stm32f4_discovery.resc`:

```renode
using sysbus
mach create "stm32f4discovery-lewis"
include @scripts/single-node/stm32f4_discovery.resc
sysbus LoadBinary @build/stm32f4/lewis.bin 0x08000000
cpu VectorTableOffset 0x08000000
sysbus.uart4 CreateFileBackend @/tmp/renode_lewis_uart.log true
emulation RunFor "00:00:05"
quit
```

A UART4 da Discovery é a saída padrão do firmware. `CreateFileBackend` grava todos os bytes transmitidos em arquivo texto, que o runner Python consome.

---

## 9.6 Execução Manual

```bash
# Compila e roda interativamente (loop infinito; pare com Ctrl+C)
make firmware-run

# Compila e roda teste automatizado por 5 segundos
make firmware-test

# Rodar com tempo customizado
python firmware/scripts/run_renode_tests.py --run-time 10
```

---

## 9.7 Relatório de Simulação

Exemplo de `reports/firmware_simulation_report.json`:

```json
{
  "model_size_bytes": 25240,
  "beat_count": 3,
  "beat_times_ms": [16, 16, 16],
  "checks": {
    "header": true,
    "model_size": true,
    "inference_init": true,
    "beats": true,
    "end": true
  },
  "uart_log": "firmware/build/stm32f4/renode_uart.log",
  "uart_log_text": "..."
}
```

> **Nota:** os tempos `beat_times_ms` refletem a latência real de inferência do TFLM no Renode. No modelo atual a latência medida é de aproximadamente **16 ms** por batimento (cerca de 16 181 µs), bem abaixo do limite de 200 ms do QG9.

---

## 9.8 Métricas e Interpretação

| Métrica | Fonte | Target | Observação |
| :--- | :--- | :--- | :--- |
| Latência de inferência | Log UART (`Beat N: X ms`) | < 200 ms | Medido no tempo virtual do Renode |
| Tamanho do FlatBuffer | `model_size_bytes` | < 64 KB | Apenas o modelo quantizado |
| Flash total | `arm-none-eabi-size` | < 512 KB | `.text` + `.data` do ELF |
| RAM TFLM | Arena alocada | < 64 KB | Array estático de 64 KB |
| Bit-exatidão | Comparação Python vs UART | 1 LSB de tolerância | Exige TFLM real integrado |
| Fidelidade DSP | Ground-truth vs UART | `cosine > 0.99` | Pipeline filtrado + Z-score |
| Detector R-peak | Comando `PEAK` | Sens/PPV ≥ 90 % | Detector leve C vs AMPT Python |

---

## 9.9 Limitações Conhecidas / Débito Técnico

1. **Clock virtual:** Renode emula ciclos, mas a frequência depende da configuração do `PerformanceInMips`. Os tempos absolutos são representativos, não garantidos contra silício real.
2. **Periféricos simplificados:** ADC e timers ainda não são usados; stubs fornecem sinais de teste.
3. **CMSIS-NN ativo:** o build ARM linka com a biblioteca TFLM otimizada por CMSIS-NN; kernels de referência são usados no host nativo e no interpretador Python (BUILTIN_REF).
4. **Semihosting evitado:** optou-se por UART real para maior fidelidade e menos dependência de configuração do emulador.
5. **`delay_ms()` via TIM2 IRQ:** o firmware usa interrupção de timer de propósito geral para aguardar delays no Renode/alvo, sem depender de busy-wait.
6. **`millis()` depende do SysTick no Renode:** a base de tempo é derivada de wraps de 24 bits do SysTick a 168 MHz. Para os intervalos usados no projeto (< 1 s) a precisão é suficiente, mas aplicações com delays muito longos precisam garantir que `COUNTFLAG` seja amostrado antes de perder wraps.
7. **Duplicação C/Python resolvida via fixture:** a geração do sinal de teste foi unificada em `tests/fixtures/adc_stub.py` e consumida pelos testes QG8/QG10. A consistência com `firmware/src/dsp/adc_stub.c` é verificada por bit-exatidão do output.
8. **Encerramento do Renode:** o runner emite `quit` no script `.resc` (ou tolera timeout) para garantir que o processo `renode --disable-xwt --console` termine após `emulation RunFor`; sem isso o comando `make firmware-test` pode falhar mesmo quando a simulação foi bem-sucedida.
9. **Timeout de frame UART:** para o QG10, o firmware aceita até 60 s para receber um frame de 500 amostras float32 pela UART. Isso evita timeout quando o teste Robot envia bytes lentamente para não saturar a FIFO emulada.

---

## 9.10 Próximos Passos

1. ~~Integrar TFLM real (`LEWIS_USE_TFLM=1`) e substituir stub de inferência.~~ **Concluído.**
2. ~~Implementar pipeline completo de sinal (filtros, segmentador, scaler).~~ **Concluído: filtros + Z-score integrados; detector R-peak leve validado via QG18.**
3. ~~Adicionar testes de bit-exatidão comparando output int8 com interpretador Python.~~ **Concluído (QG8).**
4. ~~Ativar CMSIS-NN e medir ganho de latência.~~ **Concluído (build ARM linka CMSIS-NN).**
5. Estender CI para rodar `make firmware-test` em toda PR.
6. **Modelagem de energia/consumo no Renode**: débito técnico documentado em [`DEBITO_TECNICO_Energia_Renode-v1.4.md`](DEBITO_TECNICO_Energia_Renode-v1.4.md); especificação em [`Camada-09-Energia-v1.4.md`](Camada-09-Energia-v1.4.md).
