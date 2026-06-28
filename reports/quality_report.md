# Quality Report — Project-Lewis

**Data:** 2026-06-15 03:42:00 UTC | **Commit:** 37414e9 | **Branch:** master

## Resumo por Quality Gate

| Gate | Nome | Status | Pass | Fail | Skip | Error | Total | Sumario |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| QG0 | Download e integridade | ✅ PASS | 22 | 0 | 0 | 0 | 22 | 22 passed |
| QG1 | Resample, loader e pré-processamento | ✅ PASS | 44 | 0 | 0 | 0 | 44 | 44 passed |
| QG2 | AMPT detector @ 500 Hz | ✅ PASS | 12 | 0 | 0 | 0 | 12 | 12 passed |
| QG3 | Feature engineering e segmentação | ✅ PASS | 55 | 0 | 0 | 0 | 55 | 55 passed |
| QG4 | Pré-treino (Chapman) | ✅ PASS | 24 | 0 | 0 | 0 | 24 | 24 passed |
| QG5 | Fine-tuning (MIT-BIH+) | ✅ PASS | 10 | 0 | 0 | 0 | 10 | 10 passed |
| QG6 | Quantização e exportação TFLM | ✅ PASS | 15 | 0 | 0 | 0 | 15 | 15 passed |
| QG7 | Build do firmware | ✅ PASS | 5 | 0 | 0 | 0 | 5 | 5 passed |
| QG8 | Bit-exatidão TFLM | ✅ PASS | 2 | 0 | 0 | 0 | 2 | 2 passed |
| QG9 | Latência e memória do firmware | ✅ PASS | 10 | 0 | 0 | 0 | 10 | 10 passed |
| QG10 | Fidelidade numérica vs ground-truth | ✅ PASS | 5 | 0 | 0 | 0 | 5 | 5 passed |
| QG13 | Watchdog software de inferência | ✅ PASS | 1 | 0 | 0 | 0 | 1 | 1 passed |
| QG16 | Filtros DSP bandpass/notch vs Python | ✅ PASS | 6 | 0 | 0 | 0 | 6 | 6 passed |
| QG17 | Fidelidade do pipeline filtrado C vs Python | ✅ PASS | 5 | 0 | 0 | 0 | 5 | 5 passed |
| QG18 | Detector leve de R-peak em C vs AMPT | ✅ PASS | 2 | 0 | 0 | 0 | 2 | 2 passed |

**Status geral:** ✅ TODOS OS GATES PASSARAM (218 testes executados, 0 falhas)

## DLQ (Dead Letter Queue)

**Falhas pendentes:** 0

Nenhuma falha pendente nos arquivos DLQ monitorados.

## Detalhes

- QG10 executado em ~347 s por batimento via Renode (timeout de frame UART estendido para 60 s).
- QG8/QG9 usam o relatório de simulação `firmware/build/stm32f4/firmware_simulation_report.json` regenerado após as mudanças de B.3.
- QG18 valida `lewis_detect_r_peaks()` contra o AMPTDetector Python em sinal sintético.

---
_Relatorio gerado automaticamente por `scripts/generate_quality_report.py` (com dados consolidados de execuções individuais)._
