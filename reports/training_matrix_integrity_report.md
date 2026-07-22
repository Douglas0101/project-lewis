# `training_matrix_integrity_report`

**Versão:** 1.0  
**Gerado em:** 2026-07-21T03:04:48Z  
**Escopo:** matriz v3 em `artifacts/stage1_matrix_v3/`  
**Geração:** `NEW_EVIDENCE_GENERATION`  
**Estado bloqueador:** `TRAINING_BLOCKED_BY_DATA_PROVENANCE`  
**Estado final permitido:** `REVIEW_REQUIRED`

## 1. Decisão executiva

A matriz contém **100/100 arquivos de célula** nas combinações esperadas:
4 famílias × 5 outer folds × 5 seeds. Essa completude é **estrutural**, não confirmatória.

O número de células confirmatórias válidas é **0/100**. Nenhuma célula contém o bundle mínimo
co-produzido exigido pelo protocolo, a identidade dos dados efetivamente carregados não coincide
com o `data_hash` declarado, a ontologia declarada está divergente da fonte atual, a árvore Git
não representa os arquivos usados e há evidência direta de sobrescrita de uma célula sem
atualização do ledger.

Nenhum novo treinamento foi iniciado nesta auditoria. As métricas existentes podem ser descritas
somente como resultados exploratórios observados. Elas não autorizam seleção, promoção,
implantação ou alegação clínica.

## 2. Classificação da geração

**Categoria:** `DERIVED_MATHEMATICALLY`

A execução é `NEW_EVIDENCE_GENERATION` porque os artefatos v3 usam ontologia, política de Q,
relógio de anotação, datasets derivados, splits e famílias diferentes dos artefatos legados.
Logo, os números históricos de MLP e fusão só podem ser usados como referência condicionada, não
como comparação confirmatória ou hard gate.

Conclusão clínica máxima permitida após futura validação completa:

```text
PATTERNS_SUPPORTED_WITHIN_OBSERVED_DATA_AND_DEFINED_POPULATION
```

Estado atual dos padrões:

```text
UNKNOWN_OR_INSUFFICIENT_EVIDENCE
```

## 3. Snapshot autenticável disponível

### 3.1 Repositório

| Item | Valor observado | Categoria |
| --- | --- | --- |
| Branch | `develop` | `OBSERVED` |
| `HEAD` | `886f1b398bf34f0e6c5b14d9409822f7c2574bf9` | `OBSERVED` |
| Árvore de `HEAD` | `bca19d19472e39f0806dd8cce5a862e316a7967c` | `OBSERVED` |
| Estado antes de escrever este relatório | 72 paths dirty: 13 modificados e 59 não rastreados | `OBSERVED` |
| Runner v3 | não rastreado por Git | `OBSERVED` |
| Matriz v3 | não rastreada por Git | `OBSERVED` |
| Specs/calibração/domain audit v3 | não rastreados por Git | `OBSERVED` |

`source_revision=HEAD` seria falso: o commit não contém o runner, as specs nem a matriz v3.

### 3.2 Hashes dos dados e contratos

SHA-256 foi calculado sobre os bytes completos dos arquivos.

| Componente | SHA-256 observado |
| --- | --- |
| `data/features/finetuning_mitbih_family.parquet` | `dc4069b94a5ff44743f47970e1542a9bce6521caafc2e2e568403fd8d39f8111` |
| `data/features/stage1_binary.npz` | `8ac6969f1fc5e61cd6470f3f3b7baf908bd63bbb3ae9c37efc30bb8c463065f7` |
| `data/features/stage1_binary.parquet` | `390b1be321aeb85baefc4500d8bf4a2282401273034ea0a073a6c86056fae892` |
| `data/features/training_manifest.json` | `71060aef63ad40318c29f6c58d7ec1ca2aad4aed9567613ec7c49f8d20b5a730` |
| `src/features/ontology_v3.py` atual | `051f4d8b476e4dde2e32b125b302b18b091e4d582d97fc217a50b1d5ab0f78a1` |
| preprocessing composto atual | `0b8792fd9d080b76f33cc31cf1cca2e47da2278b9a3e2934ce3d04bec0220c58` |
| `scripts/run_stage1_matrix_v3.py` atual | `5bd915a6f3e23e34632127d3b506cb8c700c74707db8b77a648f6ed49e0b06e0` |
| split index v3 (bytes) | `9f83175ef67dfbfa2152b3cde65cda82f5f99deb97c4c5e29331dd0cf91b37be` |

O `data_hash` das 100 células é o hash de `finetuning_mitbih_family.parquet`. Porém o runner
carrega `stage1_binary.npz` e `stage1_binary.parquet` em
`scripts/run_stage1_matrix_v3.py:98-105`. Os dois inputs efetivos não são hasheados nem ligados à
célula. Portanto, `data_hash` não autentica os bytes usados para treinar.

O split declara `ontology_hash=af4278e8…1280`, enquanto a fonte atual é `051f4d8b…78a1`.
O preprocessing composto declarado coincide com a fonte atual, mas isso não resolve a divergência
da ontologia nem a ausência das demais identidades.

## 4. Integridade das 100 células

### 4.1 Cobertura combinatória

| Família | Células esperadas | Células presentes | Ledger `DONE` |
| --- | ---: | ---: | ---: |
| A — CNN waveform | 25 | 25 | 25 |
| B — MLP features | 25 | 25 | 25 |
| C — CNN + features | 25 | 25 | 25 |
| D — beat + quality e treino separado de ritmo | 25 | 25 | 25 |
| **Total** | **100** | **100** | **100** |

**Categoria:** `OBSERVED`.

### 4.2 Campos de identidade por célula

| Campo exigido | Presença | Resultado |
| --- | ---: | --- |
| `cell_id`, `fold`, `seed`, `family` | 100/100 | presente |
| `data_hash` | 100/100 | presente, mas ligado ao arquivo errado |
| `split_hash` | 0/100 | há `fold_manifest_sha256`, sem o contrato solicitado |
| `ontology_hash` | 100/100 | divergente da fonte atual |
| `preprocessing_hash` | 100/100 | coincide com a composição atual |
| `feature_hash` | 0/100 | ausente |
| `config_hash` | 0/100 | ausente |
| `model_hash` | 0/100 | ausente; modelos não foram preservados |
| `scaler_hash` | 0/100 | ausente; scalers/imputers não foram preservados |
| `calibrator_hash` | 0/100 | ausente; calibração não foi executada |
| `threshold_hash` | 0/100 | ausente; só há valor numérico |
| `metrics_hash` | 0/100 | ausente |
| `environment_hash` | 0/100 | ausente |
| `source_revision` | 0/100 | ausente |

### 4.3 Verificação criptográfica do ledger

- SHA-256 dos bytes do arquivo coincide com o ledger: **0/100**.
- O hash interno, calculado sobre JSON canônico antes de inserir o próprio campo `sha256`,
  recomputa: **100/100**.
- Hash interno coincide com o ledger: **99/100**.
- Divergência: `d_f1_s17`.

Para `d_f1_s17`:

| Evidência | Ledger | Arquivo atual |
| --- | --- | --- |
| `completed_utc` / `created_utc` | `2026-07-19T10:56:05.554953+00:00` | `2026-07-20T04:57:52.345670+00:00` |
| hash | `f81dffec…be37` | `71d83c63…dbf4` |
| ROC-AUC | `0.8805307616` | `0.8805307616` |

O runner grava a célula de modo incondicional em `run_cell` e o resume valida apenas
`status + data_hash` em `campaign` (`scripts/run_stage1_matrix_v3.py:443-532,550-606`). A
célula foi sobrescrita e o ledger permaneceu apontando para a versão anterior. Isso viola o
bloqueio explícito contra célula sobrescrita.

### 4.4 Identidade temporal da fonte

O runner atual tem mtime `2026-07-19T15:13:14.759669+00:00`. Foram produzidas 83 células antes
desse instante e 17 depois; todas as 17 posteriores pertencem à família D. Mtime não prova
alteração semântica, mas, sem `source_revision`/`source_hash` por célula, também não é possível
provar uma única identidade de fonte para a matriz.

## 5. Splits e cadeia de custódia

### 5.1 O que foi demonstrado

- Os cinco manifests de beat contêm 201 identificadores únicos de `record_id`.
- Cada `record_id` aparece em exatamente um conjunto de outer validation.
- Contagem por fold: 40, 41, 40, 40, 40 registros.
- Não há duplicação de `record_id` entre os conjuntos de outer validation.

**Categoria:** `OBSERVED` para separação por registro.

### 5.2 O que não foi demonstrado

Os builders criam `groups` a partir de `record_id` e copiam esse valor para
`val_patient_ids` (`scripts/build_frozen_splits_v3.py:49-135`). Não existe um manifest versionado
`record_id → patient_id`. Portanto:

```text
PATIENT_DISJOINTNESS_NOT_PROVEN
```

A separação por paciente é `NOT_SUPPORTED`; somente a separação por registro é observada.

### 5.3 Linhagem por amostra

`stage1_binary.parquet` contém 461.600 linhas e 201 `lineage_path` válidos, todos resolvíveis.
Esses lineages são por registro, não por amostra. Nos 201 JSONs não aparecem os campos exigidos:

- `dataset_id`;
- `patient_id`;
- `segment_id`;
- `source_sampling_rate` e `target_sampling_rate` como contrato de amostra;
- `annotation_index_native`;
- `annotation_time_seconds`;
- `annotation_index_target`;
- `class_original` e `class_canonical`;
- `quality_label`;
- `split` e `fold`.

O parquet também não contém a maioria desses campos. A origem de cada linha não pode ser
reconstruída até o par anotação-nativa → anotação-500-Hz apenas a partir do artefato treinado.

Consequência fail-closed:

```text
TRAINING_BLOCKED_BY_DATA_PROVENANCE
```

## 6. Protocolo científico: conformidade observada

| Regra | Evidência | Status |
| --- | --- | --- |
| Mesmas 25 combinações fold–seed por família | 100 JSONs | `OBSERVED / PASS_ESTRUTURAL` |
| Outer split por registro | manifests v3 | `OBSERVED / PASS_POR_REGISTRO` |
| Outer split por paciente real | sem mapping paciente–registro | `NOT_SUPPORTED / BLOCKER` |
| Scaler waveform fit somente no inner-train | `_scale_waveform`, linhas 154–161 | `OBSERVED` na fonte atual |
| Imputer/scaler features fit somente no inner-train | `_scale_features`, linhas 164–172 | `OBSERVED` na fonte atual |
| Inner CV completo | usa somente a primeira partição de `GroupKFold(4)` | `NOT_SUPPORTED` |
| Tuning de hiperparâmetros inner | parâmetros fixos | `NOT_SUPPORTED` |
| Estratégia de balanceamento selecionada inner | class weights fixos em todas as famílias | `NOT_SUPPORTED` |
| Early stopping inner | validação interna única | `OBSERVED` na fonte atual; incompleto |
| Calibração em partição independente | sem partição/calibrador | `NOT_SUPPORTED / BLOCKER` |
| Threshold congelado antes do outer | escolhido no inner-val | `OBSERVED` na fonte atual |
| Threshold com busca adequada | 99/100 no limite superior 0,70 | `HYPOTHESIS_REQUIRING_TEST` |
| Outer test avaliado uma vez | caminho normal faz uma avaliação | `SUPPORTED_INFERENCE`, não autenticado |
| Célula não sobrescrevível | `d_f1_s17` contradiz | `OBSERVED / FAIL` |
| Modelo/scaler/calibrador/threshold co-produzidos | nenhum preservado | `OBSERVED / BLOCKER` |
| Predictions/embeddings/history preservados | ausentes | `OBSERVED / BLOCKER` |

Nessas linhas, `OBSERVED` significa observação direta da fonte atual. Sem identidade de fonte por
célula, o comportamento não pode ser atribuído com certeza a todas as execuções passadas.

## 7. Métricas existentes — descritivas, não confirmatórias

Os valores abaixo são derivados aritmeticamente das 25 células por família. Denominador:
25 combinações fold–seed; os cinco seeds reutilizam os mesmos pacientes em cada fold e não são
25 populações independentes.

| Família | ROC-AUC média ± DP | Recall Anormal | Precision Anormal | F1-macro | MCC | ECE-15 | P10 ROC-AUC | Pior fold ROC-AUC médio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 0,8985 ± 0,0435 | 0,7812 | 0,5313 | 0,7742 | 0,5770 | 0,1499 | 0,8348 | fold 2: 0,8375 |
| B | 0,9655 ± 0,0194 | 0,8908 | 0,7097 | 0,8755 | 0,7613 | 0,0745 | 0,9358 | fold 2: 0,9334 |
| C | 0,9665 ± 0,0126 | 0,8503 | 0,7766 | 0,8875 | 0,7812 | 0,0881 | 0,9539 | fold 0: 0,9586 |
| D | 0,9032 ± 0,0379 | 0,7780 | 0,5383 | 0,7744 | 0,5784 | 0,1498 | 0,8482 | fold 1: 0,8592 |

As linhas B e C reproduzem aritmeticamente a referência histórica fornecida no prompt. Isso
confirma a origem numérica da referência; não autentica o protocolo nem demonstra equivalência.

Ausências métricas nas células:

- NPV, log-loss/NLL, calibration intercept/slope e classwise-ECE;
- métricas por paciente, dataset, qualidade e classe canônica S/V/F;
- denominadores por paciente para cada erro;
- predictions por amostra necessárias ao bootstrap agrupado e à análise de erros.

## 8. Suporte observado e concentração

Fonte: `finetuning_mitbih_family.parquet`, 469.723 batimentos, 201 `record_id`.

O suporte efetivo abaixo usa apenas o índice de concentração de Kish por registro:

\[
n_{\mathrm{Kish}} = \frac{(\sum_i n_i)^2}{\sum_i n_i^2}
\]

Ele mede concentração entre registros, mas **não** corrige autocorrelação temporal, janelas
sobrepostas ou duplicatas. Portanto é limite descritivo, não tamanho amostral clínico efetivo.

| Classe | Registros rotulados como pacientes | Batimentos | `n_Kish` | Maior registro | Top-5 registros | Folds com suporte |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| N | 200 | 406.440 | 186,460 | 1,04% | 4,35% | 5/5 |
| S | 141 | 16.934 | 28,895 | 10,74% | 30,22% | 5/5 |
| V | 174 | 37.182 | 63,955 | 4,18% | 16,52% | 5/5 |
| F (`FUSION`) | 45 | 1.044 | 3,935 | 35,63% | 81,99% | 5/5 |
| Q (`Q_OR_UNKNOWN`) | 34 | 8.123 | 4,086 | 25,63% | 99,03% | 5/5 |

A matriz Stage 1 exclui Q e avalia N vs S/V/F. A presença nominal de F em 5/5 folds não remove
a concentração extrema. Para inferência robusta sobre F, o estado é:

```text
INSUFFICIENT_CLASS_SUPPORT
```

### 8.1 Cobertura por dataset

| Dataset | Registros | Batimentos | `n_Kish` | Maior registro | Top-5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| INCART | 75 | 175.789 | 71,431 | 2,22% | 10,01% |
| MITDB | 48 | 109.440 | 46,222 | 3,07% | 14,32% |
| SVDB | 78 | 184.494 | 74,469 | 2,31% | 9,64% |

AFDB não participa do classificador de batimentos; aparece em tarefa de ritmo separada.
Os 78 headers locais de SVDB declaram 128 Hz. Os lineages observados registram MITDB 360 Hz,
SVDB 128 Hz e INCART 257 Hz, todos convertidos para 500 Hz. Como frequência, lead e dataset são
fortemente acoplados, o probe de origem é obrigatório antes de interpretar embeddings como
fisiologia.

### 8.2 Qualidade no Stage 1

Denominador: 461.600 batimentos.

| Flag | Batimentos | Registros | Fração |
| --- | ---: | ---: | ---: |
| `qf_flatline` | 0 | 0 | 0,0000% |
| `qf_clip` | 610 | 20 | 0,1321% |
| `qf_off_center` | 30.822 | 178 | 6,6772% |

Não há `quality_label` canônico nem análise de métricas condicionadas a essas flags.

## 9. Evidências confirmatórias ausentes

| Entregável solicitado | Estado atual |
| --- | --- |
| 1. `training_matrix_integrity_report` | **PRODUZIDO por este arquivo; matriz reprovada** |
| 2. `arithmetic_trend_report` | ausente; histories/checkpoints não preservados |
| 3. `pattern_coverage_matrix` | parcial nesta auditoria; morfologia/ritmo/duplicatas sem qualificação completa |
| 4. `patient_support_report` | parcial; só existe `record_id`, sem identidade de paciente autenticada |
| 5. `dataset_shortcut_report` | ausente; há biblioteca e testes sintéticos, não execução real |
| 6. `counterfactual_invariance_report` | ausente; há biblioteca e testes sintéticos, não execução real |
| 7. `fold_heterogeneity_report` | ausente; métricas agregadas não explicam pacientes/qualidade/erros |
| 8. `paired_candidate_comparison` | ausente; sem predictions e bootstrap pareado por paciente |
| 9. `calibration_report` | ausente; nenhum calibrador foi ajustado |
| 10. `external_validation_report` | ausente |
| 11. `robustness_report` | ausente; controles negativos e LODO não executados |
| 12. `artifact_lineage_bundle` | ausente; nenhum modelo/scaler/calibrador/threshold preservado |
| 13. `risk_of_bias_assessment` | apenas especificação/risk register; sem avaliação da matriz |
| 14. `training_decision_report` | não pode promover; somente `REVIEW_REQUIRED` |

Consequências adicionais:

```text
CALIBRATION_INCOMPLETE
EXTERNAL_VALIDATION_REQUIRED
INSUFFICIENT_EVIDENCE
```

## 10. Verificação de software executada

Comando:

```bash
uv run --locked pytest -q \
  tests/test_matrix_families_v3.py \
  tests/test_rebuild_v3_infra.py \
  tests/test_temporal_alignment_v3.py \
  tests/test_ontology_v3.py
```

Resultado observado: **53 passed em 7,02 s**, Python 3.12.3.

Esses testes validam unidades e contratos sintéticos. Eles não validam os 100 bundles, pois os
bundles não existem, nem executam calibração, LODO, bootstrap por paciente ou validação externa.

## 11. Gates obrigatórios antes de qualquer nova célula

1. Ratificar formalmente D1–D7, o pivot da prioridade E06.5 e os parâmetros ainda marcados
   `PROPOSED_REQUIRES_HUMAN_RATIFICATION`.
2. Congelar árvore Git limpa ou snapshot de fonte autenticado; registrar hash por célula.
3. Regenerar splits contra a ontologia final e persistir mapping real paciente–registro.
4. Criar manifest imutável com os dez hashes exigidos e assinatura/digest detached.
5. Persistir os campos de custódia por amostra, inclusive índices nativo/tempo/500 Hz.
6. Ligar `data_hash` aos bytes exatos de NPZ/parquet realmente carregados e verificar alinhamento
   linha a linha entre ambos.
7. Implementar preflight fail-closed e smoke autenticado antes da campanha.
8. Fazer inner CV completo; separar treino, seleção, calibração e threshold por pacientes.
9. Persistir por célula: modelo, scaler, imputer, calibrador, threshold, predictions, embeddings,
   histories, denominadores e hashes.
10. Bloquear sobrescrita; cada tentativa deve usar novo `cell_id`/generation id.
11. Executar controles de labels permutados, metadados, região mascarada e identidade paciente.
12. Executar probe de dataset com IC, condicionado à classe/qualidade, LODO e contrafactuais.
13. Executar bootstrap pareado com 10.000 reamostragens por paciente e validação externa separada.
14. Registrar revisão humana independente antes de qualquer continuidade operacional.

## 12. Tabela de afirmações epistemológicas

| Afirmação | Categoria epistemológica | Evidência | Denominador | Incerteza | Limitação |
| --- | --- | --- | ---: | --- | --- |
| Há 100 combinações estruturais | `OBSERVED` | cells + ledger | 100 células | baixa | não prova validade |
| Há 0 células confirmatórias | `DERIVED_MATHEMATICALLY` | campos/bundles obrigatórios | 100 células | baixa | contrato precisa ratificação formal, mas também é exigido pelo prompt |
| Outer folds são disjuntos por registro | `OBSERVED` | cinco manifests | 201 IDs | baixa | registro não prova paciente |
| Outer folds são disjuntos por paciente | `NOT_SUPPORTED` | mapping ausente | — | não quantificável | identidade paciente indisponível |
| O `data_hash` não representa os inputs carregados | `OBSERVED` | hashes + `_load_stage1` | 3 arquivos | baixa | stage1 pode ser derivado do family parquet, mas a derivação não está ligada |
| A ontologia declarada diverge da fonte atual | `OBSERVED` | `af4278e8…` vs `051f4d8b…` | 1 contrato | baixa | conteúdo histórico não preservado |
| `d_f1_s17` foi substituída sem atualizar ledger | `OBSERVED` | timestamp + hashes | 1 célula | baixa | motivo da reexecução desconhecido |
| MLP e fusão reproduzem a referência aritmética | `DERIVED_MATHEMATICALLY` | médias/DP das células | 25 por família | dependência fold–seed | artefatos não autenticados |
| F tem suporte efetivo baixo e concentrado | `DERIVED_MATHEMATICALLY` | Kish e frações | 1.044 beats/45 registros | autocorrelação não modelada | paciente real não autenticado |
| O modelo aprendeu padrões fisiológicos | `NOT_SUPPORTED` | probes/LODO/contrafactuais ausentes | — | não quantificável | sem modelos/embeddings preservados |
| O modelo generaliza externamente | `NOT_SUPPORTED` | validação externa ausente | — | não quantificável | LODO também ausente |
| A calibração é adequada | `NOT_SUPPORTED` | nenhum calibrador | — | não quantificável | Brier/ECE brutos não substituem calibração |
| Nova campanha pode continuar | `NOT_SUPPORTED` | blockers acima | 14 gates | alta | requer ratificação e reconstrução |

## 13. Estado final

Não retornar:

```text
TRAINING_MATRIX_VALIDATED
PATTERNS_SUPPORTED_WITHIN_DEFINED_SCOPE
CONTROLLED_PROCESS_CONTINUATION_APPROVED
```

Retornar:

```text
NEW_EVIDENCE_GENERATION
TRAINING_BLOCKED_BY_DATA_PROVENANCE
INSUFFICIENT_CLASS_SUPPORT
CALIBRATION_INCOMPLETE
EXTERNAL_VALIDATION_REQUIRED
INSUFFICIENT_EVIDENCE
REVIEW_REQUIRED
```

A decisão humana independente ainda não foi registrada. Nenhum candidato é promovível.
