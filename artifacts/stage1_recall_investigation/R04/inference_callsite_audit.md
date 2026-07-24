# R04 callsite audit — inference mode

## Methodology

Pesquisamos todos os callsites de inferência no código produtivo e nos testes
usando `ast_grep` e `ripgrep` para:

- `model.predict(`
- `model(`
- `training=True`
- `training=False`
- `predict_on_batch`
- `__call__`

Cada ocorrência foi classificada como **caminho produtivo**, **teste de
pipeline**, **teste de treinamento**, **script de treinamento/otimização**,
**script diagnóstico** ou **R04 forense**.

## Resultado da pesquisa

### `model.predict(...)`

| Arquivo | Linha | Contexto | Classificação | Modo efetivo |
| --- | --- | --- | --- | --- |
| `src/inference/two_stage_pipeline.py` | 177 | `_forward` | **PRODUTIVO** | `predict()` (inference) |
| `src/inference/two_stage_mlp_pipeline.py` | 264 | `_forward` | **PRODUTIVO** | `predict()` (inference) |
| `src/inference/stage1_mlp_runner.py` | 77 | `predict` | **PRODUTIVO** | `predict()` (inference) |
| `src/models/evaluate.py` | 545 | `evaluate_model` | Teste/validação | `predict()` (inference) |
| `src/models/two_stage_pipeline.py` | 78, 106 | predição de treino | Treinamento | `predict()` (inference) |
| `src/models/qg5_gates.py` | 75 | quality gate | Teste | `predict()` (inference) |
| `tests/test_two_stage_mlp_qg5.py` | 68, 99 | teste MLP QG5 | Teste | `predict()` (inference) |
| `tests/test_model.py` | 144, 160 | teste de modelo | Teste | `predict()` (inference) |
| `tests/test_pretrain.py` | 49 | teste pré-treino | Teste | `predict()` (inference) |
| `tests/test_quantization_degradation.py` | 95 | teste quantização | Teste | `predict()` (inference) |
| `scripts/audit_stage1_loader_equivalence.py` | 161, 190 | R03 forense | Diagnóstico | `predict()` (inference) |
| `scripts/build_stage1_loader_fixture.py` | 129 | R03 fixture | Diagnóstico | `predict()` (inference) |
| `scripts/stage1_loader_lane.py` | 161, 190 | R03 lane | Diagnóstico | `predict()` (inference) |
| `scripts/diagnose_stage1_qg5.py` | 335 | pipeline.predict | Diagnóstico | `predict()` (inference) |
| `scripts/optimize_stage1_threshold.py` | 30 | otimização | Otimização | `predict()` (inference) |
| `scripts/select_best_mlp_fold.py` | 131, 201 | seleção fold | Treinamento | `predict()` (inference) |
| `scripts/train_stage1_mlp.py` | 110 | treino MLP | Treinamento | `predict()` (inference) |
| `scripts/train_stage2_mlp.py` | 190 | treino MLP | Treinamento | `predict()` (inference) |
| `scripts/train_stage2_baseline_enhanced.py` | 84 | treino | Treinamento | `predict()` (inference) |
| `scripts/validate_quantized_mlp.py` | 39, 60 | validação | Teste | `predict()` (inference) |
| `scripts/analyze_stage2_thresholds_v11_v13.py` | 38, 65 | análise | Otimização | `predict()` (inference) |
| `scripts/_smoke_stage1.py` | 98 | smoke test | Teste | `predict()` (inference) |

### `model(..., training=...)`

| Arquivo | Linha | Contexto | Classificação | training |
| --- | --- | --- | --- | --- |
| `src/models/slha/warmup.py` | 60 | warmup de treino | Treinamento | `False` |
| `src/callbacks/gradient_monitor.py` | 111 | monitor de gradiente | Treinamento | `False` |
| `scripts/stage1_inference_mode_instance.py` | 87 | R04 forense | Diagnóstico | `False` |
| `scripts/stage1_inference_mode_instance.py` | 97, 173 | R04 forense | Diagnóstico | `True` |
| `scripts/audit_stage1_inference_mode.py` | 137 | R04 forense | Diagnóstico | `False` |

### `predict_on_batch`

Nenhuma ocorrência em `src/` ou `tests/`.

## Caminho QG5 produtivo

O teste QG5 (`tests/test_two_stage_qg5.py::test_two_stage_qg5_end_to_end`) chama
`loaded_pipeline.predict(X)`, que utiliza internamente:

```text
two_stage_pipeline.py::TwoStageInferencePipeline.predict()
  → _run_stage1()
    → _forward(stage1_model, X_norm)
      → model.predict(X, verbose=0)
  → _run_stage2() (condicional)
    → _forward(stage2_model, X_norm)
      → model.predict(X, verbose=0)
```

Nenhuma chamada com `training=True` ocorre no caminho produtivo.

## Conclusão

- Nenhum callsite produtivo utiliza `training=True`.
- Todos os uses de `training=True` estão em scripts diagnósticos R04 ou em código
de treinamento.
- Todos os caminhos de inferência produtiva usam `model.predict(...)`.
- `model.predict(...)` é equivalente a `model(..., training=False)` conforme provado
nas evidências R04.

## Veredito

```text
H11 — modo de inferência incorreto = REJECTED
Nenhum BLOCKED_INFERENCE_MODE_ERROR encontrado.
```
