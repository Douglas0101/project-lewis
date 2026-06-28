# Status do Treinamento — Camada 04 (Modelagem) v1.1

## Resumo executivo

- **QG4 (pré-treino Chapman):** não atingido. A investigação apontou subajuste
  (underfitting) por descompasso entre rótulos globais de 10 s e segmentos de
  1 s. A estratégia foi abandonar o pré-treino e treinar o backbone do zero no
  MIT-BIH+.
- **QG5 (fine-tuning MIT-BIH+):** não atingido com a arquitetura atual. O
  modelo converge para ~79% de acurácia e F1-macro ~0,36 no fold 0, distante
  dos thresholds de acurácia >93% e F1-macro >0,85.
- **QG6 (quantização TFLM):** atingido para o fold 0. O modelo INT8 ocupa
  **34,3 KB** (< 64 KB) e a degradação de F1-macro vs float é **0,0033**
  (< 0,02).

## Evidências do QG4

| Métrica | Valor |
|---|---|
| Train AUC-ROC macro (melhor época) | ~0,83 |
| Val AUC-ROC macro (por segmento) | ~0,68 |
| Val AUC-ROC macro (por registro) | ~0,71 |
| Val loss | > 0,15 |

Conclusão: o modelo não generaliza para registros de validação porque a tarefa
(pré-treino multi-label a partir de 1 s) é intrinsecamente difícil para o
backbone pequeno.

## Evidências do QG5

Configuração final testada:

- Backbone: Conv1D(16,7) → Conv1D(40,5) → Conv1D(80,3) → Dense(80)
- Params: **19.933** (< 20.000)
- Treino: from-scratch, 5 folds GroupKFold por paciente
- Class weights: `sqrt(balanced_weight / min_weight)` com teto 20
- Augmentation: escala de amplitude, shift temporal, ruído gaussiano
- Seleção do modelo: melhor F1-macro AAMI na validação

Resultado médio aproximado (treinamento interrompido no fold 1):

| Métrica | Valor alvo | Valor obtido |
|---|---|---|
| Acc | > 0,93 | ~0,79 |
| F1-macro | > 0,85 | ~0,36 |
| MCC | > 0,80 | ~0,23 |
| N Se | > 0,96 | ~0,87 |
| S Se | > 0,75 | ~0,03 |
| V Se | > 0,90 | ~0,06 |
| F Se | > 0,60 | ~0,00 |
| Q Se | > 0,70 | ~0,07 |

O tuning de thresholds não conseguiu fechar a lacuna (melhor F1-macro ~0,37).

## Evidências do QG6 (fold 0)

| Métrica | Valor |
|---|---|
| FlatBuffer INT8 | 34,30 KB |
| Limite | 64 KB |
| Δ Acc (INT8 vs float) | +0,018 |
| Δ F1-macro (INT8 vs float) | -0,0033 |

**QG6 passa** no fold 0.

## Artefatos gerados

- `models/finetuned_float32_v1.1.keras`
- `models/input_scaler_v1.1.pkl`
- `models/quantized/finetuned_int8_v1.1.tflite`
- `models/quantized/finetuned_int8_v1.1.h`
- `models/quantized/quantization_params.h`
- `models/quantized/quantization_params.json`

## Próximas alternativas para atingir QG5

1. **Aumentar capacidade do backbone** — exige relaxar o limite de 20k params
   (impacta FlatBuffer e firmware TFLM).
2. **Classificador em duas etapas** — primeiro N vs anormal, depois subtipo.
3. **Features morfológicas + MLP** — usar RR, QRS width etc. (requer extrator
   equivalente em C no firmware).
4. **Revisar o QG5** — os thresholds de sensibilidade por classe podem ser
   irrealistas para um modelo tão compacto no MIT-BIH+ com 5 classes.
