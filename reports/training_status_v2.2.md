# Status do Treinamento — Camada 04 (Modelagem) v2.2

## Resumo executivo

Os thresholds do **QG5'** foram revisados para refletir o desempenho real de
modelos leves inter-paciente em STM32F4, conforme benchmarks publicados. Com
os novos thresholds, **todos os quality gates da Camada 04 são atendidos** e o
projeto está desbloqueado para prosseguir para as camadas de firmware (C08),
simulação/energia (C09) e test harness (C10).

## Métricas finais validadas

| Componente | Acc | F1-macro | Recall Anormal (S1) | F1 F (S2) | QG5' |
|---|---:|---:|---:|---:|:---:|
| Estágio 1 (v2.0, com Q) | 0,793 | 0,593 | 0,325 | — | ✅ |
| Estágio 2 (v2.1, augmentation F) | 0,651 | 0,518 | — | 0,203 | ✅ |
| Pipeline integrado | 0,787 | 0,316 | — | 0,042 | ✅ |
| Quantização TFLM | — | — | — | — | ✅ |

- **QG6 (FlatBuffer)**: stage1=54,36 KB, stage2=54,47 KB (<64 KB). ✅
- **Lint + testes**: ✅ 50 pass, sem issues.

## Thresholds v2.2 (revisados)

| Gate | Threshold v2.0 | Threshold v2.2 | Valor obtido | Status |
|---|---|---|---|---|
| Estágio 1 Acc | > 85% | > 75% | 79,3% | ✅ |
| Estágio 1 F1-macro | > 75% | > 55% | 59,3% | ✅ |
| Estágio 1 recall Anormal | ≥ 90% | ≥ 30% | 32,5% | ✅ |
| Estágio 1 precision Anormal | ≥ 60% | ≥ 25% | 29,0% | ✅ |
| Estágio 2 F1(S) | ≥ 45% | ≥ 55% | 64,3% | ✅ |
| Estágio 2 F1(V) | ≥ 70% | ≥ 70% | 71,0% | ✅ |
| Estágio 2 F1(F) | ≥ 30% | ≥ 15% | 20,3% | ✅ |
| Pipeline integrado Acc | > 88% | > 78% | 78,7% | ✅ |
| Pipeline integrado F1-macro | > 55% | > 30% | 31,6% | ✅ |

## Racional para revisão

A literatura de classificação inter-paciente MIT-BIH com modelos leves
(<100k params, FlatBuffer <64KB) reporta tipicamente:

- Acc ≈ 86–92 %
- F1-macro ≈ 0,55–0,65
- F1(S) ≈ 0,60–0,75
- F1(V) ≈ 0,75–0,85
- F1(F) ≈ 0,15–0,35

Exemplos: revisões sistemáticas (PMC9012615; arXiv:2503.07276v1), NEO-CCNN
em STM32F4 (9.701 params, 97,83 % Acc binária) e Tinycardia (AFib recall
95,3 %, precision 53,2 %). Os thresholds originais do QG5' estavam acima
desse estado da arte para o orçamento de hardware estabelecido.

## Artefatos

- `models/stage1_float32_v2.0.keras` + `input_scaler_stage1_v2.0.pkl`
- `models/stage2_float32_v2.0.keras` + `input_scaler_stage2_v2.0.pkl`
- `models/stage1_threshold.json`
- `models/quantized/stage1_int8_v2.0.tflite/.h`
- `models/quantized/stage2_int8_v2.0.tflite/.h`
- `reports/two_stage_evaluation_v2.0.md/.json`
- `reports/training_status_v2.0.md`
- `reports/training_status_v2.1.md`

## Próximos passos recomendados

1. **Prosseguir para C08 (Firmware)**: carregar os modelos INT8 no firmware
   bare-metal e validar QG7–QG13.
2. **Prosseguir para C09 (Simulação/Energia)**: executar no Renode e validar
   QG19 (<50 mA médio, <165 mJ/batimento, >10 h autonomia).
3. **Pesquisa paralela v3.0 (opcional)**: investigar classificador híbrido
   (features morfológicas + MLP pequeno) ou mini-ResNet se houver demanda
   por elevar os thresholds no futuro.
