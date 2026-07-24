# R02 — auditoria interna do artefato Stage 1

## Hipótese falsificável

`stage1_float32_v2.0.keras` é um artefato Keras 3 íntegro, com entrada
`(None, 500, 1)`, saída softmax `(None, 2)` e contrato histórico
`0=N/Normal`, `1=Anormal`.

## Estado anterior

- modelo: `models/stage1_float32_v2.0.keras`;
- SHA-256: `cd5e2474f294d965d52662f80e12a21024d551749f8b9d787b9c80bd34dbc347`;
- threshold observado, não alterado: `0.5800000000000001`;
- TP/FP/TN/FN: `127/0/128/1793`.

## Implementação

- `src/models/keras_artifact_inspector.py`: inspetor ZIP-only validado por Pydantic;
- `scripts/inspect_stage1_keras_artifact.py`: geração atômica dos relatórios;
- `tests/test_stage1_keras_artifact_contract.py`: contrato estrutural e imutabilidade;
- `tests/test_stage1_positive_class_contract.py`: prova executável da polaridade;
- `docs/positive_class_contract.md`: cadeia histórica da classe positiva;
- `scripts/diagnose_stage1_qg5.py`: AP agora sempre acompanhada de prevalência e lift;
- `tests/test_stage1_ap_reports_prevalence.py`: proteção da interpretação da AP;
- `config/runtime_identity.json`, `src/runtime_identity.py`,
  `scripts/check_runtime_identity.py` e `tests/test_runtime_identity.py`:
  identidade canônica do runtime;
- correções de tipagem Keras 3 em `src/inference/stage1_mlp_runner.py` e
  `tests/test_bit_exact_python_tflm.py`, eliminando a divergência Pyright.

Nenhum modelo, scaler, threshold, dataset ou requisito QG5 foi alterado.

## Evidências do ZIP

A inspeção ocorreu antes de qualquer desserialização do modelo.

| Componente | Tamanho | SHA-256 |
| --- | ---: | --- |
| arquivo `.keras` | 209.985 bytes | `cd5e2474f294d965d52662f80e12a21024d551749f8b9d787b9c80bd34dbc347` |
| `metadata.json` | 64 bytes | `d7a51ba9e0fc667c1aa067233e22c4a49b774f2004728872dd2305fb99dfa4b9` |
| `config.json` | 9.123 bytes | `6fc94bc111430510445bce4af1770cce954702622d5e941cc21e2dc7b0fece9d` |
| `model.weights.h5` | 200.468 bytes | `a742efbc9a78c4f3635be52b2bb9efd20f032f9a5397ff265c421810f0382a28` |

O ZIP contém somente esses três membros. Cópias somente leitura e o relatório completo estão
em `artifacts/stage1_recall_investigation/R02/`.

### Família e arquitetura

- metadata Keras: `3.14.1`;
- data de salvamento: `2026-07-04@00:43:26`;
- módulo principal: `keras.src.models.functional`;
- classe principal: `Functional`;
- registered name: `Functional`;
- família: `KERAS_3_STANDALONE`;
- nome do modelo: `lewis_backbone`;
- camadas: `11`;
- datasets de peso do modelo: `10`;
- parâmetros do modelo, excluindo slots do Adam: `13.218`.

| Índice | Nome | Classe | Módulo | Configuração relevante |
| ---: | --- | --- | --- | --- |
| 0 | input | InputLayer | keras.layers | `(None,500,1)`, float32 |
| 1 | conv1d_1 | Conv1D | keras.layers | relu |
| 2 | maxpool_1 | MaxPooling1D | keras.layers | — |
| 3 | conv1d_2 | Conv1D | keras.layers | relu |
| 4 | maxpool_2 | MaxPooling1D | keras.layers | — |
| 5 | conv1d_3 | Conv1D | keras.layers | relu |
| 6 | maxpool_3 | MaxPooling1D | keras.layers | — |
| 7 | gap | GlobalAveragePooling1D | keras.layers | — |
| 8 | embedding | Dense | keras.layers | 64, relu |
| 9 | dropout | Dropout | keras.layers | presente |
| 10 | output | Dense | keras.layers | 2, softmax |

Não existe BatchNormalization. Existe um Dropout chamado `dropout`. Não existem Lambda,
camadas customizadas ou referências `custom_objects`.

### Contrato de entrada e saída

- entrada: sequência de 500 amostras, um canal, `float32`;
- saída estrutural: duas unidades softmax, shape `(None, 2)`;
- domínio estrutural: probabilidades complementares que somam 1;
- não são logits na saída serializada;
- não há label mapping dentro do ZIP.

O runtime e a ausência de transformação duplicada serão verificados novamente em R07.

### Configuração histórica serializada

- optimizer: `Adam`;
- learning rate: `0.0010000000474974513`;
- loss: `sparse_categorical_crossentropy`;
- metrics: `accuracy`;
- `run_eagerly=false`, `jit_compile=false`.

Referências recursivas:

- `keras.src`: 1;
- `tf_keras.src`: 0;
- `tensorflow.keras`: 0;
- `tensorflow.python.keras`: 0;
- `custom_objects`: 0.

## Prova da classe positiva

O ZIP não serializa nomes de classes. A polaridade foi provada fora do pipeline de inferência:

1. `scripts/prepare_two_stage_datasets.py:46-108` persiste
   `np.where(y == 0, 0, 1)`, onde AAMI 0 é N;
2. `config/stage1_binary.yaml` usa esse NPZ, duas classes, softmax e
   `sparse_categorical_crossentropy`;
3. `scripts/run_stage1_training.py:64-194` usa `class_names=["N", "Anormal"]` sem
   LabelEncoder ou one-hot intermediário;
4. o commit produtor v2.0 `27ad38b` contém a mesma cadeia;
5. `reports/two_stage_evaluation_v2.0.json` corrobora a ordem N, Anormal.

Conclusão: índice 0 é Normal; índice 1 é Anormal; `pos_label=1` está coerente. A prova completa
está em `docs/positive_class_contract.md`.

## Correção da AP e leitura do stress gate

O gate observado é conceitualmente `QG5_STAGE1_ABNORMAL_STRESS`, não um teste binário
balanceado. A cauda positiva atual tem precisão perfeita, mas o ranking global não pode ser
classificado como bom a partir da AP.

| Métrica | Valor |
| --- | ---: |
| TP | 127 |
| FP | 0 |
| TN | 128 |
| FN | 1.793 |
| Recall | 0.0661458333 |
| Precision | 1.0 |
| Specificity | 1.0 |
| F1 Anormal | 0.1240840254 |
| Balanced Accuracy | 0.5330729167 |
| AP | 0.9136421077 |
| Positive Prevalence | 0.9375 |
| AP Lift | -0.0238578923 |
| Predicted Positive Rate | 0.06201171875 |
| False Negative Rate | 0.9338541667 |

Por AP isoladamente, não há evidência de ganho sobre a referência de prevalência. São
necessários no mínimo 576 TP para recall 0,30, ou 449 anormais adicionais. R08 deverá medir
quantos dos 128 normais se tornam falsos positivos nesse ponto. Nenhum threshold foi escolhido.

## Pyright e runtime

A divergência foi reproduzida com Pyright `1.1.409`, cwd do projeto, sem `[tool.pyright]` ou
`pyrightconfig.json`, usando exatamente `uv run pyright src tests`. As duas chamadas
`predict(..., verbose=0)` foram introduzidas no commit `e28d0ad2`, mas os warnings passaram a
aparecer quando os objetos foram tipados como `keras.Model` pelo loader Keras 3 na árvore de
trabalho atual. A assinatura Keras não anotada tem default `verbose="auto"`, levando Pyright
a inferir incorretamente somente `str`, embora o runtime aceite `0`.

Foram adicionadas supressões locais e justificadas `reportArgumentType`, sem alterar execução.
Resultado final: `0 errors, 0 warnings, 0 informations`.

O runtime oficial foi comparado com `config/runtime_identity.json`. `uv run` usa
`.venv/bin/python3` (resolvido para `/usr/bin/python3.12`), Keras 3.14.1 e TensorFlow 2.21.0.
O `CONDA_PREFIX` externo permanece visível, mas o executável observado coincide com o
manifesto.

## Checagens

- [PASS] inspeção ZIP sem desserialização;
- [PASS] família Keras 3 identificada;
- [PASS] input/output/ativação comprovados;
- [PASS] BatchNorm/Dropout/Lambda/custom objects inventariados;
- [PASS] classe positiva comprovada por cadeia histórica;
- [PASS] SHA original antes e depois idêntico;
- [PASS] `make lint`;
- [PASS] `uv run pyright src tests`: 0 erros, 0 warnings;
- [PASS] 9 testes R02/AP/runtime;
- [PASS] 5 testes bit-exact após correção de tipagem;
- [PASS] QG5 reproduz a mesma falha e métricas, como esperado;
- [NOT RUN] equivalência direta dos loaders em subprocessos independentes (R03);
- [NOT RUN] contrato criptográfico da família modelo–scaler–threshold (R05/R10).

## Hipótese

`CONFIRMED`: o artefato é Keras 3, recebe `(500,1)`, produz softmax de duas classes e o
índice positivo histórico é 1=Anormal.

H5 permanece `LOW_PROBABILITY_BUT_NOT_REJECTED` até R03. H6–H8 permanecem inconclusivas.
H9 permanece inconclusiva como hipótese completa até os testes negativos e AP bidirecional
de R06, embora a polaridade estrutural tenha sido provada. H12 permanece
`SUPPORTED_NOT_CONFIRMED`.

## Checkpoint

`PASS_WITH_WARNINGS`: o label mapping não está embutido no ZIP e ainda não existe manifest
imutável que vincule criptograficamente modelo, scaler e threshold. A evidência histórica é
suficiente para autorizar a auditoria técnica do loader, não para promoção de produção.

## Próxima etapa autorizada

`R03 — equivalência direta dos loaders`.
