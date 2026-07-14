# Stage 1 — contrato observado de inferência QG5

> Estado: **R01 observado + contrato estrutural R02**. R02 comprovou a saída softmax
> de duas classes e o contrato positivo. R07 ainda verificará a distribuição em runtime.
> Nenhum threshold, teste QG5, modelo, scaler ou dataset foi alterado.

## Ativação do diagnóstico

```bash
STAGE1_DIAGNOSTIC=1 uv run python scripts/diagnose_stage1_qg5.py
```

Sem o valor exato `1`, o script termina com código 2 e não cria artefatos. Com o modo ativo,
os resultados são escritos atomicamente em:

- `artifacts/stage1_recall_investigation/baseline/stage1_qg5_trace.json`;
- `artifacts/stage1_recall_investigation/baseline/stage1_qg5_samples.csv`.

O CSV não contém sinais ECG. Cada entrada completa de `(500, 1)` é representada apenas por
SHA-256 antes e depois do scaler.

## Mapa executado

| Ordem | Transformação | Implementação | Entrada | Saída observada |
| ---: | --- | --- | --- | --- |
| 1 | seleção determinística | `tests/test_two_stage_qg5.py::_load_combined_test_subset`, linha 44 | dois NPZ | `(2048, 500, 1)`, `float32` |
| 2 | cast de entrada | `TwoStageInferencePipeline.predict` | `(2048, 500, 1)` | `(2048, 500, 1)`, `float32` |
| 3 | reshape + scaler + reshape | `TwoStageInferencePipeline._normalize`, linha 140 | `(2048, 500, 1)` | `(2048, 500, 1)`, `float32` |
| 4 | forward Keras | `TwoStageInferencePipeline._forward`, linha 167 | `(2048, 500, 1)` | `(2048, 2)`, `float32` |
| 5 | seleção da coluna 1 | `TwoStageInferencePipeline._run_stage1`, linha 145 | `(2048, 2)` | `(2048,)`, `float32` |
| 6 | decisão `score >= threshold` | `TwoStageInferencePipeline._run_stage1`, linha 145 | `(2048,)` | `(2048,)`, `int64` |
| 7 | roteamento Stage 2 | `TwoStageInferencePipeline.predict` | somente decisões Stage 1 anormais | classe final `N/S/V/F` |
| 8 | binarização da classe final | `tests/test_two_stage_qg5.py::_evaluate_stage1` | `N/S/V/F` | `0=N`, `1=Anormal` |
| 9 | recall | `sklearn.metrics.recall_score(..., pos_label=1)` | `(2048,)` vs `(2048,)` | `0.0661458333` |

A reprodução técnica direta foi comparada com o caminho canônico `pipeline.predict(X)`:

- delta máximo dos scores: `0.0`;
- divergências de decisão: `0`;
- threshold canônico idêntico: `true`.

## Estatísticas por transformação

| Transformação | Min | Max | Média | Desvio | NaN | Inf |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| entrada selecionada | -4.864931 | 9.496250 | 0.645055 | 0.848849 | 0 | 0 |
| entrada escalada | -6.432933 | 12.040265 | 0.654720 | 1.091898 | 0 | 0 |
| saída bruta `(n,2)` | 0.012771 | 0.987229 | 0.500000 | 0.275773 | 0 | 0 |
| coluna 1 interpretada | 0.012771 | 0.650687 | 0.282087 | 0.169011 | 0 | 0 |
| decisão binária | 0 | 1 | 0.062012 | 0.241177 | 0 | 0 |

## Identidade observada da família v2.0

| Papel | Path | SHA-256 |
| --- | --- | --- |
| modelo | `models/stage1_float32_v2.0.keras` | `cd5e2474f294d965d52662f80e12a21024d551749f8b9d787b9c80bd34dbc347` |
| scaler | `models/input_scaler_stage1_v2.0.pkl` | `80f5fabf5aeb790f9bf6c0bd8dfc993133d6d53f5f9787cc53011fb1c2b3acb3` |
| threshold | `models/stage1_threshold_v2.0.json` | `b75e14ab95136d544664a166eaa15fcdcfe38fda821462a781b26265b6966746` |

O threshold efetivamente usado no ponto de decisão foi `0.5800000000000001`.
A compatibilidade histórica entre esses três artefatos ainda deve ser provada em R05/R10.

## Dataset observado

| Arquivo | SHA-256 | Seleção |
| --- | --- | --- |
| `data/features/stage1_binary.npz` | `23adba2139ac691df639b132f4fd52b06fa1d3095d2d3988b011f1382b6a2b35` | primeiros 128 `N` |
| `data/features/stage2_multiclass.npz` | `68fb0a8e9fa3bc3fd06df7af074222c9837b8a04dc644b9f823d26919ba6b04a` | primeiros 640 de cada `S/V/F` |

A concatenação é permutada com `numpy.random.default_rng(42)`. Resultado: 128 normais e
1.920 anormais. Os NPZ usados pelo teste não possuem `record_id` nem `group_id`; por isso,
esses campos são vazios no CSV. O `sample_id` estável é derivado de dataset, classe original
e índice imutável no NPZ, por exemplo `stage2_multiclass:V:123`.

Essa ausência de proveniência por paciente impede confirmar nesta etapa generalização
patient-wise. O subset também é enriquecido para 93,75% de anormais e seleciona as primeiras
ocorrências; portanto não deve ser tratado como evidência única de generalização clínica.

## Classificação conceitual do gate

O teste atual é documentado como `QG5_STAGE1_ABNORMAL_STRESS`. Seu propósito é verificar
se o Stage 1 deixa passar volume suficiente de batimentos anormais para o Stage 2. O gate
`Recall(Anormal) >= 0.30` permanece inalterado.

Este subset não se destina a:

- avaliação balanceada Normal versus Anormal;
- calibração populacional;
- generalização inter-paciente;
- seleção final de threshold.

Devem ser planejados, mas não implementados durante R02–R08:

- `QG5_STAGE1_BINARY_BALANCED`;
- `QG5_STAGE1_PATIENTWISE`.

## Integridade e baseline manual

- IDs duplicados: `0`;
- fontes selecionadas duplicadas: `0`;
- amostras/targets/outputs/predições: `2048/2048/2048/2048`;
- amostras perdidas: `0`;
- ordem preservada: `true`;
- TP/FP/TN/FN: `127/0/128/1793`;
- recall Anormal: `0.0661458333`;
- precision Anormal: `1.0`;
- F1 Anormal: `0.1240840254`;
- specificity: `1.0`;
- false-negative rate: `0.9338541667`;
- balanced accuracy: `0.5330729167`;
- average precision: `0.9136421077`;
- positive prevalence: `0.9375`;
- AP lift: `-0.0238578923`;
- taxa predita Anormal: `0.06201171875`;
- TP mínimo para recall 0,30: `576`;
- anormais adicionais que precisam ser recuperados: `449`.

O sistema prevê somente 127 dos 1.920 anormais. Ele não produz falsos positivos no
threshold atual, mas deixa de detectar 1.793 anormais. A precisão e a especificidade
perfeitas decorrem de um ponto operacional extremamente conservador e não constituem
evidência suficiente de qualidade global.

A cauda de previsões positivas apresenta precisão perfeita no threshold atual, indicando
comportamento extremamente conservador. Entretanto, AP=`0.9136421077` não deve ser
classificada como elevada sem considerar a prevalência positiva de `0.9375`. Como a
prevalência de Anormal é superior à AP observada, `AP lift=-0.0238578923`: por essa métrica
isolada, não há evidência de ganho de ranking sobre a referência de prevalência.

Threshold mismatch continua sendo a hipótese `SUPPORTED_NOT_CONFIRMED`. R07 deverá avaliar
o ranking global e R08 deverá responder quantos dos 128 normais se tornam falsos positivos
para recuperar ao menos 449 anormais adicionais. Nenhum ajuste é autorizado por este
documento.
