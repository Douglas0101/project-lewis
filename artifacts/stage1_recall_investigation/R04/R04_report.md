# R04 — auditoria do modo de inferência, Dropout e RNG

## Hipótese falsificável

O pipeline QG5 utiliza modo de inferência. `model.predict(x)` e `model(x, training=False)`
produzem outputs equivalentes e repetíveis. A camada Dropout só é ativada quando
`training=True`.

## Estado de entrada

- R03 checkpoint: `HYPOTHESIS_REJECTED`;
- modelo SHA-256: `cd5e2474f294d965d52662f80e12a21024d551749f8b9d787b9c80bd34dbc347`;
- scaler SHA-256: `80f5fabf5aeb790f9bf6c0bd8dfc993133d6d53f5f9787cc53011fb1c2b3acb3`;
- threshold SHA-256: `b75e14ab95136d544664a166eaa15fcdcfe38fda821462a781b26265b6966746`;
- fixture R03 SHA-256: `59d08ed5ae4a15e55e45f2c09afd3cebcdf4921782c71e8febd649df81d48794`;
- array SHA-256: `f35880b59c6221398b2b02fe57b26dc3271b6dfa22fe93e95281f89bf228c8ae`;
- baseline TP/FP/TN/FN: `127/0/128/1793`;
- baseline Recall: `0.0661458333`;
- threshold: `0.5800000000000001`.

A pré-checagem R04 reproduziu o baseline sem drift.

## Inventário do Dropout

Arquivo: `dropout_inventory.json`.

```text
layer_name = dropout
rate = 0.3
seed = null
noise_shape = null
trainable = true
dtype = float32
module = keras.src.layers.regularization.dropout
class_name = Dropout
has_seed_generator = true
seed_generator_type = keras.src.random.seed_generator.SeedGenerator
non_trainable_variable_count = 1
trainable_variable_count = 10
```

A única variável não treinável do modelo é o estado do `SeedGenerator` do Dropout.

## Fixture compartilhada

Reutilizada de R03:

```text
shape = (11, 500, 1)
dtype = float32
file_sha256 = 59d08ed5ae4a15e55e45f2c09afd3cebcdf4921782c71e8febd649df81d48794
array_sha256 = f35880b59c6221398b2b02fe57b26dc3271b6dfa22fe93e95281f89bf228c8ae
match_manifest = true
```

Não foi recriada nem sobrescrita.

## Instâncias independentes

Cada instância foi carregada em subprocesso Python limpo com:

```python
keras.saving.load_model(path, compile=False, safe_mode=True)
```

### Instance A — `model.predict()`

Três chamadas consecutivas.

```text
pairwise array_equal = true
max_abs_delta = 0.0
argmax_disagreement_count = 0
threshold_disagreement_count = 0
NaN = 0
Inf = 0
```

### Instance B — `model(..., training=False)`

Três chamadas consecutivas.

```text
pairwise array_equal = true
max_abs_delta = 0.0
argmax_disagreement_count = 0
threshold_disagreement_count = 0
NaN = 0
Inf = 0
```

### Instance A vs Instance B

Comparação do primeiro output de cada instância:

```text
array_equal = true
max_abs_delta = 0.0
mean_abs_delta = 0.0
p99_abs_delta = 0.0
argmax_disagreement_count = 0
threshold_disagreement_count = 0
```

`predict()` e `training=False` são bit-exatos equivalentes no modelo/runtime canônico.

### Instance C — `model(..., training=True)`

Três chamadas consecutivas.

```text
pesos treináveis imutáveis = true
estado RNG avançou = true
pairwise max_abs_delta > 0.0
```

Apenas o estado não treinável do `SeedGenerator` avançou; os pesos treináveis
permaneceram idênticos.

### Instance D — recarregamento determinístico

Dois subprocessos independentes com `keras.utils.set_random_seed(42)`.

```text
seed = 42
process_a_first_training_true_sha256 = bd89740b3bd40d72...
process_b_first_training_true_sha256 = bd89740b3bd40d72...
identical = true
```

O estado inicial do RNG é reproduzível entre processos quando seed, ambiente,
modelo e input são iguais.

## Estado RNG

Arquivo: `rng_state_transitions.json`.

```text
predict: initial RNG == final RNG  → false advance
training_false: initial RNG == final RNG  → false advance
training_true: RNG state avançou em todas as 3 chamadas
```

O estado não treinável do Dropout é avançado apenas em `training=True`.

## Pesos treináveis

Arquivo: `model_state_hashes.json`.

```text
trainable_weight_hash_before = <hash>
trainable_weight_hash_after = <hash>
trainable_weights_immutable = true
non_trainable_weights_changed = true
```

Apenas o estado não treinável do RNG muda após forward pass; os pesos
permanecem imutáveis.

## Auditoria de callsites

Arquivo: `inference_callsite_audit.md`.

Resumo:

- todos os caminhos produtivos usam `model.predict(..., verbose=0)`;
- `tests/test_two_stage_qg5.py` chama `loaded_pipeline.predict(X)`;
- o pipeline canônico `src/inference/two_stage_pipeline.py` executa `model.predict(X, verbose=0)`
  em `_forward()`;
- nenhum callsite produtivo utiliza `training=True`;
- os únicos `training=True` estão em scripts diagnósticos R04 e em código de treinamento.

## Testes automatizados

Arquivos:

- `tests/test_stage1_predict_training_false_equivalence.py`
- `tests/test_stage1_dropout_inference_mode.py`
- `tests/test_stage1_inference_state_immutability.py`
- `tests/test_stage1_dropout_rng_state.py`
- `tests/test_stage1_no_training_true_in_production.py`

Resultado:

```text
21 passed
```

## Checagem pós-operatória

- [PASS] modelo preservado;
- [PASS] scaler preservado;
- [PASS] threshold preservado;
- [PASS] datasets preservados;
- [PASS] fixture R03 preservada;
- [PASS] Dropout inventariado;
- [PASS] seed generator identificado;
- [PASS] predict repetível;
- [PASS] training=False repetível;
- [PASS] predict equivale a training=False;
- [PASS] pesos treináveis imutáveis;
- [PASS] RNG não avança em inferência;
- [PASS] comportamento training=True documentado;
- [PASS] nenhum callsite produtivo usa training=True;
- [PASS] QG5 mantém baseline (Recall = 0.0661);
- [PASS] `make lint`;
- [PASS] `uv run pyright src tests` → `0 errors, 0 warnings, 0 informations`;
- [PASS] `git diff --check`;
- [PASS] testes focados passam.

## Métricas QG5 antes/depois

| Métrica | Antes | Depois |
| --- | ---: | ---: |
| TP | 127 | 127 |
| FP | 0 | 0 |
| TN | 128 | 128 |
| FN | 1793 | 1793 |
| Recall | 0.0661458333 | 0.0661458333 |
| Precision | 1.0 | 1.0 |
| Specificity | 1.0 | 1.0 |
| F1 | 0.1240840254 | 0.1240840254 |
| Balanced Accuracy | 0.5330729167 | 0.5330729167 |
| AP | 0.9136421077 | 0.9136421077 |
| Positive Prevalence | 0.9375 | 0.9375 |
| AP Lift | -0.0238578923 | -0.0238578923 |
| Predicted Positive Rate | 0.06201171875 | 0.06201171875 |

## Regressões

Nenhuma regressão numérica, estrutural, de decisão, de baseline ou de lint foi
encontrada.

## Hipótese

`REJECTED`: o pipeline produtivo usa modo de inferência corretamente.

```text
H11 — modo de inferência incorreto = REJECTED
H11' — Dropout ativo na inferência = REJECTED
H11'' — pesos treináveis mutados por forward pass = REJECTED
```

## Checkpoint

`HYPOTHESIS_REJECTED`

## Próxima etapa autorizada

`R05 — contrato modelo–scaler–preprocessing`
