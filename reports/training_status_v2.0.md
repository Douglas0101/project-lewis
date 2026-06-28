# Status do Treinamento — Camada 04 (Modelagem) v2.0

## Resumo executivo

A arquitetura de duas etapas (Estágio 1: N vs Anormal; Estágio 2: S vs V vs F) foi
implementada e validada, mas **os thresholds do QG5' v2.0 não foram atingidos** com o
backbone atual (~38k–50k params).

| Métrica alvo (QG5') | Threshold | Estágio 1 obtido | Estágio 2 obtido | Pipeline integrado |
|---|---|---|---|---|
| Acc | > 0,88 (integrado) | 0,793 | 0,716 | 0,789 |
| F1-macro | > 0,55 (integrado) | 0,593 | 0,580 | 0,325 |
| Recall Anormal (Estágio 1) | ≥ 0,90 | 0,325 | — | — |
| Precision Anormal (Estágio 1) | ≥ 0,60 | 0,290 | — | — |
| F1 S (Estágio 2) | ≥ 0,45 | — | 0,676 | 0,172 |
| F1 V (Estágio 2) | ≥ 0,70 | — | 0,772 | 0,184 |
| F1 F (Estágio 2) | ≥ 0,30 | — | 0,294 | 0,062 |

**QG5'**: ❌ NÃO ATINGIDO  
**QG6 (tamanho TFLM)**: ✅ PASS — stage1=54,36 KB, stage2=54,47 KB (< 64 KB)

## Evidências

### Estágio 1 — N vs Anormal

- Backbone escalado: Conv1D(32,64,96) + Dense(96) → ~38,6k params.
- Treino: GroupKFold 2 splits, 20 épocas, class_weight='balanced'.
- Melhor F1-macro no fold 1: **0,5807** (Acc=0,7855, MCC=0,162).
- Distribuição de probabilidades mostra sobreposição grande entre N e Anormal.
- Mesmo com threshold tuning, precision Anormal não supera ~0,29.

### Estágio 2 — S vs V vs F

- Backbone escalado: Conv1D(32,64,96) + Dense(96) → ~38,7k params.
- Treino: GroupKFold 5 folds, 50 épocas, class_weight com teto 20 para F.
- Melhor fold: F1-macro=0,471; **S e V atingem os thresholds**, mas **F permanece crítico**
  (F1 ~0,01–0,07 na média, F1 ~0,29 no melhor fold).
- A classe F tem apenas ~1.044 amostras; o modelo não consegue aprender padrões estáveis.

### Pipeline integrado

- Bug de indexação corrigido em `src/models/two_stage_pipeline.py` (Estágio 2 agora roda
  apenas sobre amostras classificadas como Anormal pelo Estágio 1).
- Gargalo principal: **Estágio 1** tem baixo recall Anormal (~0,33), então a maioria dos
  batimentos S/V/F nunca chega ao Estágio 2, degradando o F1-macro integrado.

## Artefatos gerados

- `models/stage1_float32_v2.0.keras` + `input_scaler_stage1_v2.0.pkl`
- `models/stage2_float32_v2.0.keras` + `input_scaler_stage2_v2.0.pkl`
- `models/stage1_threshold.json`
- `models/quantized/stage1_int8_v2.0.tflite/.h`
- `models/quantized/stage2_int8_v2.0.tflite/.h`
- `reports/two_stage_evaluation_v2.0.md`
- `reports/two_stage_evaluation_v2.0.json`

## Próximas alternativas para atingir QG5'

1. **Aumentar ainda mais o backbone** — usar ~50k params (limite próximo de 64 KB INT8)
   para reduzir subajuste no Estágio 1.
2. **Excluir a classe Q do treino do Estágio 1** — treinar N vs (S+V+F), tornando a
   fronteira mais nítida.
3. **Features morfológicas + MLP** — adicionar RR, QRS width etc. e implementar extrator
   equivalente em C.
4. **Oversampling/augmentation da classe F** — gerar variações sintéticas dos ~1k batimentos F.
5. **Revisar os thresholds do QG5'** — os targets atuais podem ser irrealistas para o
   hardware alvo com representação puramente raw-signal.

## Testes e lint

- `make lint`: ✅ sem issues.
- `pytest tests/test_model.py tests/test_two_stage_pipeline.py tests/test_quantization.py`: ✅ 41 pass.
