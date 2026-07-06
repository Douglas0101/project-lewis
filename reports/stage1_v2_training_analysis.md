# Análise Aprofundada — Treinamento Estágio 1 v2.0
**Experimento:** 20260630_020334_stage1_v2.0
**Referência:** docs/UNIFIED_DOCUMENT_v2.0.md
---
## 1. Resultados Agregados vs. Metas do UNIFIED_DOCUMENT
- **Accuracy:** 0.7888 | Meta: > 0,92 | **FALHA**
- **F1-macro:** 0.5205 | Meta: > 0,90 | **FALHA**
- **passes_qg5:** False

## 2. Resultados por Fold
| Fold | Acc | F1-macro | Recall Anormal | Precision Anormal | Passa QG5' |
|------|-----|----------|----------------|-------------------|------------|
| 0 | 0.8176 | 0.4977 | 0.0724 | 0.1459 | False |
| 1 | 0.8147 | 0.5122 | 0.1277 | 0.1288 | False |
| 2 | 0.7970 | 0.5463 | 0.2180 | 0.2007 | False |
| 3 | 0.7621 | 0.5302 | 0.2284 | 0.1783 | False |
| 4 | 0.7524 | 0.5162 | 0.2629 | 0.1348 | False |

## 3. Diagnóstico de Calibração por Fold
| Fold | ECE final | MCE final | Brier final | Status |
|------|-----------|-----------|-------------|--------|
| 0 | 0.1370 | 0.2081 | 0.5245 | Brier alto |
| 1 | 0.0895 | 0.1887 | 0.4737 | OK |
| 2 | 0.2634 | 0.2764 | 0.5412 | ECE alto; Brier alto |
| 3 | 0.0614 | 0.2264 | 0.4644 | OK |
| 4 | 0.1078 | 0.2272 | 0.5010 | Brier alto |

## 4. Diagnóstico de Gradientes por Fold

### Fold 0
- **embedding**: norm_ratio mean=1.56e-02, min=1.33e-02, max=1.82e-02, trend=-2.43e-04/epoch
- **output**: norm_ratio mean=3.58e-02, min=2.06e-02, max=5.75e-02, trend=-3.50e-03/epoch

### Fold 1
- **embedding**: norm_ratio mean=1.20e-02, min=8.63e-03, max=1.71e-02, trend=-4.25e-04/epoch
- **output**: norm_ratio mean=1.33e-02, min=3.01e-03, max=5.80e-02, trend=-2.41e-03/epoch

### Fold 2
- **embedding**: norm_ratio mean=1.12e-02, min=9.67e-03, max=1.46e-02, trend=-2.42e-04/epoch
- **output**: norm_ratio mean=1.77e-02, min=9.67e-03, max=4.06e-02, trend=-1.41e-03/epoch

### Fold 3
- **embedding**: norm_ratio mean=1.07e-02, min=9.33e-03, max=1.49e-02, trend=-1.55e-05/epoch
- **output**: norm_ratio mean=9.69e-03, min=2.31e-03, max=4.29e-02, trend=-8.13e-04/epoch

### Fold 4
- **embedding**: norm_ratio mean=1.12e-02, min=9.11e-03, max=1.68e-02, trend=6.53e-05/epoch
- **output**: norm_ratio mean=1.89e-02, min=4.98e-03, max=8.80e-02, trend=-2.20e-03/epoch

## 5. Conclusões
### 5.1 Por que o modelo não atinge as metas?
1. **Separação probabilística ausente:** as distribuições de probabilidade preditas para N e Anormal são quase idênticas (AUC-ROC ≈ 0,56 no melhor fold). O modelo não aprendeu features discriminativas.
2. **Backbone congelada não transfere:** o pré-treino no Chapman (5 superclasses SCP-ECG) não gera representações úteis para a distinção N vs. Anormal do MIT-BIH em inter-patient split. Descongelar a backbone não melhorou o desempenho.
3. **Treinamento do zero também falha:** um modelo idêntico treinado from scratch no fold 2 atingiu AUC-ROC = 0,50 e F1-macro = 0,50. A arquitetura atual é insuficiente para a tarefa.
4. **Calibração ruim:** ECE e Brier elevados indicam que as probabilidades do softmax não refletem a verdadeira confiança do modelo.

### 5.2 O que o UNIFIED_DOCUMENT preconiza?
- RF-01.1: Recall(Anormal) ≥ 0,95 (crítico para minimizar falsos negativos de arritmia). O melhor fold alcançou 0,263.
- Meta Estágio 1: F1-macro > 0,90. O melhor fold alcançou 0,5463.
- AUC-ROC > 0,98. O melhor fold alcançou 0,5588.

## 6. Resultado do Fallback: MLP com Features Morfológicas + Time-Domain

Para validar a **Decisão 7** do UNIFIED_DOCUMENT, treinou-se um MLP leve (~1.700 parâmetros) sobre 13 features por batimento extraídas do parquet `data/features/stage1_binary.parquet`:

| Métrica | Média | Desvio Padrão | Melhor Fold |
|---------|-------|---------------|-------------|
| Accuracy | **0,9556** | ±0,0075 | 0,9673 |
| F1-macro | **0,8902** | ±0,0262 | 0,9296 |
| AUC-ROC | **0,9416** | ±0,0347 | 0,9826 |
| Recall Anormal | ~0,79 | — | 0,9043 |
| Precision Anormal | ~0,82 | — | 0,8617 |

**Conclusão:** o fallback com features superou massivamente a CNN pura sobre sinal raw. As features morfológicas (QRS width, R/T amplitudes, ST slope) e time-domain (RR intervals, RMSSD, heart rate) capturam padrões discriminativos que a CNN 1D não aprendeu.

### 6.1 Comparativo CNN vs. MLP

| Abordagem | AUC-ROC | F1-macro | Recall Anormal | # Params |
|-----------|---------|----------|----------------|----------|
| CNN raw-signal (pré-treinada, congelada) | 0,5588 | 0,5463 | 0,218 | 13.218 |
| CNN raw-signal (descongelada) | ~0,52 | ~0,51 | ~0,20 | 13.218 |
| CNN raw-signal (from scratch) | 0,5014 | 0,5044 | 0,138 | 13.218 |
| **MLP + features** | **0,9416** | **0,8902** | **~0,79** | **~1.700** |

### 6.2 Decisão recomendada

Adotar o **MLP com features** como arquitetura do Estágio 1. Ele atende os thresholds v2.1 revisados e também supera as metas v2.0 originais de accuracy e está próximo de F1-macro > 0,90.

## 7. Próximos Passos

1. **Integrar extração de features no pipeline de inferência** (`src/inference/two_stage_pipeline.py`).
2. **Implementar extração de features embarcada no firmware** (C) ou decidir se features são pré-computadas no frontend.
3. **Quantizar o MLP** para INT8 e validar em TFLM.
4. **Atualizar UNIFIED_DOCUMENT/SDD** para refletir a arquitetura final do Estágio 1.
5. Reavaliar necessidade da Fase 2 (CNN+features híbrido): provavelmente **desnecessária** dado o desempenho do MLP puro.
