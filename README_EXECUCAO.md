# Project-Lewis — Fase 1: Pipeline de Dados + Ciência de Dados

## Setup (primeira vez)
```bash
make env
```

## Download dos datasets
```bash
make download-all
```

## Testes de Quality Gate
```bash
make test
```

## Pipeline completo
```bash
make all
```

## Notas
- Frequência padrão: **500 Hz** (MIT-BIH nativo 360 Hz é resampleado)
- Chapman-Shaoxing: ~8.2 GB, usar tf.data.Dataset com generator
- MIT-BIH+: 226 registros (48 + 78 + 25 + 75)
- Input shape do modelo: **(500, 1)** — 1000 ms @ 500 Hz
- Segmentador gera janelas de **500 amostras** (`window_len=500`,
  `half_len=250`); R-peak próximo ao centro (índice 250). O fallback de
  600 ms gera 300 amostras. Isso alinha segmentação e modelo em `(500, 1)`.
- Ambiente atual validado: Python 3.13 + TensorFlow 2.21
  (`requirements.txt` ainda contém pinos antigos; será reconciliado na frente DevOps)
- Target: STM32F4 (Cortex-M4F, 192 KB SRAM, arena TFLM ~64 KB)

## Fase 2: Firmware + Simulacao Renode (sem hardware)

### Instalar dependencias (toolchain ARM + Renode)
```bash
make firmware-deps
```

### Build nativo (iteracao rapida)
```bash
make firmware-native
./firmware/build/native/lewis
```

### Build para STM32F4
```bash
make firmware-build
```

### Rodar simulacao no Renode
```bash
make firmware-test
```

O comando acima delega para `make -C firmware LEWIS_USE_TFLM=1 firmware-test`,
que compila o firmware, executa no emulador STM32F4Discovery por 5s,
captura a saida UART4 e gera o relatório em
`firmware/build/stm32f4/firmware_simulation_report.json`.
Uma cópia pode ser mantida em `reports/firmware_simulation_report.json` para
referência; os testes de QG7/QG8/QG9 procuram o artefato em ambos os caminhos.
