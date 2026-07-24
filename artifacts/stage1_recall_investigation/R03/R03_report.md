# R03 — equivalência direta dos loaders

## Hipótese falsificável

`load_keras_model()` seleciona Keras 3 com `safe_mode=True` e produz estrutura, pesos e
previsões equivalentes a `keras.saving.load_model()` em processo independente.

## Estado de entrada

- HEAD: `7510fdcb9b2046625ed025aa511355cc3756be03`;
- modelo SHA-256: `cd5e2474f294d965d52662f80e12a21024d551749f8b9d787b9c80bd34dbc347`;
- scaler SHA-256: `80f5fabf5aeb790f9bf6c0bd8dfc993133d6d53f5f9787cc53011fb1c2b3acb3`;
- threshold SHA-256: `b75e14ab95136d544664a166eaa15fcdcfe38fda821462a781b26265b6966746`;
- stage1 NPZ SHA-256: `23adba2139ac691df639b132f4fd52b06fa1d3095d2d3988b011f1382b6a2b35`;
- stage2 NPZ SHA-256: `68fb0a8e9fa3bc3fd06df7af074222c9837b8a04dc644b9f823d26919ba6b04a`;
- threshold efetivo: `0.5800000000000001`;
- baseline TP/FP/TN/FN: `127/0/128/1793`.

A pré-checagem reproduziu Recall `0.0661458333`. Não houve baseline drift.

## Implementação

- `src/models/keras_loader.py`: decisão auditável por família, path/hash resolvidos e
  `keras.saving.load_model(..., safe_mode=True)` explícito para Keras 3;
- `scripts/build_stage1_loader_fixture.py`: fixture determinística e não sobrescritível;
- `scripts/stage1_loader_lane.py`: lane independente de referência ou helper;
- `scripts/audit_stage1_loader_equivalence.py`: comparação estrutural, pesos, previsões e
  restauração de compile;
- `tests/test_stage1_loader_direct_equivalence.py`: testes positivos e negativos.

Nenhum threshold, scaler, dataset, modelo ou teste QG5 foi modificado.

## Lanes independentes

Cada lane foi iniciada em subprocesso Python separado com `TF_USE_LEGACY_KERAS=0` e
`KERAS_BACKEND=tensorflow`.

### Lane A — referência

```python
keras.saving.load_model(path, compile=False, safe_mode=True)
```

Resultado: `reference_loader_result.json`.

### Lane B — helper

```python
load_keras_model(path, compile=False)
```

Resultado: `helper_loader_result.json`.

Decisão observada do helper:

```text
artifact_family_detected = KERAS_3_STANDALONE
serialized_module = keras.src.models.functional
selected_loader = keras.saving.load_model
compile = false
safe_mode = true
custom_objects = []
```

O helper não selecionou `tf.keras`/`tf_keras` para o artefato real.

## Fixture compartilhada

- path: `loader_fixture.npz`;
- shape: `(11,500,1)`;
- dtype: `float32`;
- arquivo SHA-256: `59d08ed5ae4a15e55e45f2c09afd3cebcdf4921782c71e8febd649df81d48794`;
- array SHA-256: `f35880b59c6221398b2b02fe57b26dc3271b6dfa22fe93e95281f89bf228c8ae`;
- modelo vinculado: `cd5e2474...dbc347`;
- scaler vinculado: `80f5fabf...b3acb3`.

A seleção é uma união determinística, com duplicatas fundidas, dos seguintes papéis:

- Normal verdadeiro;
- Anormal verdadeiro;
- maior score positivo;
- menor score negativo;
- imediatamente abaixo de `0.58` (`0.5794573426`);
- imediatamente acima de `0.58` (`0.5800524950`);
- quantis p01, p25, p50, p75 e p99.

Não houve seleção manual de exemplos favoráveis.

## Comparação estrutural

As duas lanes produziram:

| Campo | Referência | Helper |
| --- | --- | --- |
| Tipo | Functional | Functional |
| Módulo | keras.src.models.functional | keras.src.models.functional |
| Input | `(None,500,1)` | `(None,500,1)` |
| Output | `(None,2)` | `(None,2)` |
| Parâmetros | 13.218 | 13.218 |
| Camadas | 11 | 11 |
| Weight tensors | 10 | 10 |
| Dtype policy | float32 | float32 |
| Variáveis treináveis | 10 | 10 |
| Variáveis não treináveis | 1 | 1 |

Nomes, classes e módulos das 11 camadas foram idênticos. A equivalência estrutural foi
`true`.

## Comparação dos pesos

`weight_comparison.csv` contém os 10 tensors, associados por posição, camada, shape e dtype.

```text
all_weight_shapes_equal = true
all_weight_dtypes_equal = true
all_weights_array_equal = true
max_abs_weight_delta = 0.0
```

Todos os tensores foram `float32`; nenhum critério dependeu de `variable.name`.

## Comparação das previsões

| Métrica | Valor |
| --- | ---: |
| shape | `(11,2)` |
| dtype | float32 |
| NaN referência/helper | 0/0 |
| Inf referência/helper | 0/0 |
| max abs delta | 0.0 |
| mean abs delta | 0.0 |
| p99 abs delta | 0.0 |
| divergências argmax | 0 |
| divergências threshold 0.58 | 0 |

## Comparação `compile=True`

A lane de referência também carregou o artefato com `compile=True, safe_mode=True`:

```text
optimizer_restored = true
optimizer_class = Adam
loss = sparse_categorical_crossentropy
max_abs_prediction_delta vs compile=False = 0.0
argmax disagreements = 0
threshold disagreements = 0
```

A restauração do optimizer e da loss não alterou inferência.

## Testes positivos

- [PASS] artefato real seleciona `KERAS_3_STANDALONE`;
- [PASS] loader efetivo é `keras.saving.load_model`;
- [PASS] `safe_mode=true`;
- [PASS] duas lanes independentes equivalentes;
- [PASS] pesos e previsões idênticos;
- [PASS] compile true/false equivalentes.

## Testes negativos

- [PASS] archive sintético `tf_keras.src.engine.functional` é reconhecido como
  `TF_KERAS_LEGACY`;
- [PASS] metadata legado seleciona rota `tf.keras.models.load_model` sem afetar o artefato
  Keras 3 real;
- [PASS] fixture e manifest recusam sobrescrita;
- [PASS] teste exige cobertura das classes, decisões, vizinhança do threshold e quantis.

## Checagem pós-operatória

- [PASS] modelo preservado;
- [PASS] scaler preservado;
- [PASS] threshold preservado;
- [PASS] stage1/stage2 NPZ preservados;
- [PASS] helper selecionou Keras 3;
- [PASS] safe mode permaneceu true;
- [PASS] estruturas equivalentes;
- [PASS] pesos equivalentes;
- [PASS] previsões equivalentes;
- [PASS] divergências argmax = 0;
- [PASS] divergências em 0.58 = 0;
- [PASS] `make lint`;
- [PASS] flake8 focado nos scripts R03;
- [PASS] Pyright `0 errors, 0 warnings, 0 informations`;
- [PASS] 8 testes focados R03 + contrato R02;
- [PASS] QG5 manteve o baseline esperado.

As diferenças já existentes em `models/quantized/*` e a diretiva Pyright já existente em
`tests/test_two_stage_qg5.py` estavam presentes na pré-checagem. R03 não as criou.

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

Nenhuma regressão numérica, estrutural, de decisão ou de baseline foi encontrada.

## Hipótese

`REJECTED`: não existe evidência de regressão causada por `load_keras_model()` neste
artefato/runtime.

`H5 — helper altera predições = REJECTED`.

## Checkpoint

`HYPOTHESIS_REJECTED` — todos os critérios de equivalência foram aprovados.

## Próxima etapa autorizada

`R04 — modo de inferência e Dropout`.
