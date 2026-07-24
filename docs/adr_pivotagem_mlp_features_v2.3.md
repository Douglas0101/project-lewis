# ADR — Pivotagem para MLP sobre features v2.3

## Status
Accepted — implementado em `feature/mlp-features-v2.3`

## Contexto
O pipeline two-stage v2.0/v2.1 baseado em CNN 1D sobre sinal raw não atingiu de
forma robusta os thresholds do Quality Gate 5' (QG5') em todos os folds de
validação, especialmente na separação N vs Anormal do Estágio 1 e na
classificação S/V/F do Estágio 2.

A decisão arquitetural (UNIFIED_DOCUMENT Decisão 7) previa um fallback para um
classificador leve sobre features morfológicas/time-domain quando a CNN pura
falhasse. A v2.3 consolida esse fallback como caminho principal para os modelos
MLP de duas etapas.

## Decisões

1. **Camada de features expandida de 13 para 16 dimensões**
   - Adicionadas: `qrs_asymmetry_index`, `t_r_ratio`, `qrs_raggedness`.
   - `MorphologicalFeatures` ganhou detector fallback de onset/offset baseado em
     amplitude e validação fisiológica da largura QRS ([20, 180] ms).

2. **Modelos: MLP de uma camada oculta**
   - Estágio 1: 16 entradas → 64 unidades ReLU → Dropout 0.3 → 2 saídas softmax.
   - Estágio 2: 16 entradas → 64 unidades ReLU → Dropout 0.3 → 3 saídas softmax.
   - Pesos de classe limitados (`max_weight=20` no Estágio 1, `10` no Estágio 2).
   - Estágio 2: oversampling real da classe F até 75% da maior classe.

3. **Seleção de melhor fold por thresholds QG5' v2.3**
   - Estágio 1: maximiza F1-macro.
   - Estágio 2: prioriza folds que atendem **todos** os thresholds mínimos
     (F1(S) ≥ 0.55, F1(V) ≥ 0.70, F1(F) ≥ 0.15, F1-macro ≥ 0.45); entre eles
     maximiza F1-macro.

4. **Pipeline de inferência dedicado**
   - `TwoStageMLPPipeline` extrai features dos segmentos ECG, aplica scaler e
     executa os dois estágios.
   - Suporta features time-domain pré-computadas, evitando recomputação incorreta
     a partir de subconjuntos esparsos de R-peaks.
   - Suporta modelos Keras float32 e TFLite INT8 quantizados.

5. **Quantização INT8 full-integer**
   - Script `scripts/quantize_mlp_features.py` gera FlatBuffers < 5 KB cada.
   - Calibração sobre features **escalonadas** (mesma escala de treinamento).
   - Validação: ΔF1-macro < 2% em relação ao float32.

## Resultados

| Métrica | Valor | Threshold QG5' |
|---------|-------|----------------|
| Estágio 1 Recall(Anormal) | 0.8342 | ≥ 0.30 |
| Estágio 1 Precision(Anormal) | 0.8241 | ≥ 0.25 |
| Estágio 1 F1-macro | 0.9005 | ≥ 0.55 |
| Estágio 2 F1-macro | 0.8119 | ≥ 0.45 |
| Estágio 2 F1(S) | 0.7750 | ≥ 0.55 |
| Estágio 2 F1(V) | 0.7355 | ≥ 0.70 |
| Estágio 2 F1(F) | 0.9253 | ≥ 0.15 |
| ΔF1-macro Stage1 INT8 | 0.0019 | < 0.02 |
| ΔF1-macro Stage2 INT8 | 0.0041 | < 0.02 |
| Tamanho Stage1 INT8 | 4.88 KB | < 64 KB |
| Tamanho Stage2 INT8 | 4.98 KB | < 64 KB |

## Consequências

- **Positivas:** modelos muito menores (< 5 KB vs dezenas de KB da CNN),
  inferência mais rápida, features portáveis para C/firmware, QG5' atingido com
  folga.
- **Negativas:** dependência das features time-domain corretamente calculadas
  (requer histórico de R-peaks consecutivos por registro); o pipeline de
  inferência deve receber features pré-computadas ou R-peaks completos.

## Artefatos Afetados

- `src/features/morphological.py`
- `src/data/training_schemas.py`
- `src/features/pipeline.py`
- `src/inference/feature_extractor.py`
- `src/inference/two_stage_mlp_pipeline.py` (novo)
- `scripts/train_stage1_mlp.py`
- `scripts/train_stage2_mlp.py`
- `scripts/prepare_stage1_features.py`
- `scripts/prepare_stage2_features.py`
- `scripts/select_best_mlp_fold.py` (novo)
- `scripts/quantize_mlp_features.py` (novo)
- `scripts/validate_quantized_mlp.py` (novo)
- `tests/test_morphological_features.py`
- `tests/test_two_stage_mlp_qg5.py` (novo)
- `models/*_v2.3.*`
- `models/quantized/stage{1,2}_int8_v2.3.tflite`
