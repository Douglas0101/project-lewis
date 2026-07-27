# E07R — plano de correção, revalidação e auditoria

- **Data:** 2026-07-26
- **Governança:** `AUTONOMOUS_PREAUTHORIZED`
- **Responsável:** `AUTONOMOUS_GOVERNANCE_PREAUTH`
- **Status inicial:** E07 `BLOCKED`, 0/150 células
- **Regra de execução:** validade científica prevalece sobre desempenho

## 1. Contexto e causa

O Stage 2 v2.4 congelado agrupava por `record_id`. A fonte oficial PhysioNet comprova que MIT-BIH 201 e 202 pertencem ao mesmo indivíduo, mas os manifests legados os separavam em outer folds 3/5 e inner fold 1. Os resultados E06.5 permanecem evidência histórica record-disjoint, porém não podem fundamentar E07 patient-disjoint.

A remediação será aditiva: nenhum byte legado será reescrito. O Stage 2 v2.4 pré-DQ-01/02 e o Stage 2 r4 marcado `NOT_AUTHORIZED_FOR_TRAINING` permanecem evidência histórica. Uma geração r5, derivada byte-a-byte do parent r4 com binding ordenado NPZ↔Parquet, será a única fonte ativa para E06.5-PD/E07-PD. Splits antigos serão marcados em quarentena documental content-addressed.

## 2. Versões e namespaces

| Domínio | Versão/path planejado |
| --- | --- |
| Evidência MIT-BIH | `docs/physionet_mitdb_patient_statement.md` |
| Mapeamento MIT-BIH | `data/metadata/physionet_mitdb_patient_mapping.json` |
| Stage 2 autorizado | `data/features/v3.1.0-r5-stage2-pd/` |
| Identidade consolidada | `data/metadata/stage2_patient_identity_v4.0.json` |
| Splits novos | `data/splits/stage2_multiclass_patient_disjoint_v4.0/` |
| Quarentena legada | `experiments/stage2_v2.4_research/quarantine/splits_record_disjoint_leakage_era_v2.3/` |
| Integridade E07R | `experiments/stage2_v2.4_research/integrity/` |
| Features PD | `experiments/stage2_v2.4_research/manifests/features/e065pd-v4.0/` |
| E06.5-PD | `experiments/stage2_v2.4_research/E06_5_PD/e065pd-audit-v1/` |
| E07-PD | `experiments/stage2_v2.4_research/E07_PD/e07pd-audit-v1/` |

Todos os paths de runs e checkpoints serão write-once. Uma mudança de source/config/split exige novo ID, nunca `--force` sobre célula `DONE`.

## 3. Custódia Stage 2 r5

Antes dos splits, gerar de forma write-once `v3.1.0-r5-stage2-pd` a partir de `data/features/v3.1.0-r4/finetuning_mitbih_family.{npz,parquet}`. A transformação é filtro ordenado e determinístico para classes S/V/F e datasets com identidade confirmatória MIT-BIH+INCART.

O NPZ r5 deve conter `X`, `y`, `sample_id` e `waveform_sha256`, todos pickle-free e alinhados ao parquet. O produtor deve validar:

- hashes exatos do parent r4;
- `sample_id` único e ordem NPZ = ordem Parquet;
- `waveform_sha256` NPZ = Parquet e recomputado sobre cada `(500,1)` float32;
- label S/V/F e `y` 0/1/2 alinhados;
- clocks nativo/target e índices de anotação presentes;
- cobertura MIT-BIH+INCART sem SVDB no fitting confirmatório;
- manifest de transformação, hashes dos outputs e publicação exclusiva.

SVDB permanece `DOMAIN_SENSITIVITY`, com artefato/manifest separado e sem participação em fitting, seleção primária, S2/S4 ou métricas por paciente enquanto sua identidade estiver não autenticada.

## 4. Estratégia de identidade

### 4.1 MIT-BIH

Criar o mapping requerido com política:

```text
record 201 -> mitdb:subject:201_202
record 202 -> mitdb:subject:201_202
other record -> mitdb:subject:<record_id>
```

A união 201/202 é sustentada pela citação oficial. O mapping completo será validado contra todos os records MIT-BIH presentes no parquet congelado.

### 4.2 INCART

Consumir o mapping `IDENTITY_VERIFIED` já publicado em:

`data/splits/groupkfold_5_stratified/v3.1.0/patient_identity_manifest.json`

O arquivo deve manter SHA-256 `c8f31ddeda6825c2983e06fce31b344935c0a419d90386359c1d378279f62bab` antes do uso. Apenas os 74 records INCART presentes no Stage 2 serão projetados para o mapping consolidado; seus `patient_group_id` autenticados serão preservados.

### 4.3 SVDB

A identidade é `IDENTITY_UNVERIFIED`. Os 78 records recebem `patient_id=null` e uma barreira não biológica `partition_barrier_id=svdb:conservative-unverified-cohort`. Eles permanecem fora do fitting confirmatório e podem ser usados somente em análise de sensibilidade de domínio claramente separada. Nenhum resultado SVDB será chamado de métrica por paciente.

## 5. Geração dos splits v4.0

1. Ler o parquet r5 autorizado e validar seu SHA-256.
2. Resolver `dataset + record_id → patient_id` para todos os records confirmatórios MIT-BIH+INCART presentes.
3. Validar cobertura total, unicidade e evidência de cada paciente.
4. Gerar outer folds com `StratifiedGroupKFold`, 5 folds, `shuffle=true`, seed 42.
5. Para cada outer train, gerar quatro candidatos inner patient-disjoint com a mesma política de `select_inner_split`: seed `42 + outer_fold_index + 1` (43–47).
6. Preservar o protocolo legado de seleção inner escolhendo o primeiro split determinístico retornado pelo splitter; publicar também os quatro candidatos para auditoria.
7. Priorizar disjointidade; aceitar desbalanceamento ou `LOW_SUPPORT` em vez de quebrar grupos.
8. Persistir índices, patient IDs, record IDs, counts S/V/F, suporte 208/213, seeds e hashes.

Validações por fold:

- interseções train/validation/test de patient IDs vazias;
- interseções de record IDs vazias;
- cada patient ID aparece em uma única partição;
- todos os records de um patient ID permanecem juntos;
- 201 e 202 permanecem juntos;
- cobertura e cardinalidade equivalem ao dataset congelado;
- índices não duplicados e dentro do range;
- structural zeros e low support são reportados, não ocultados.

## 6. Quarentena dos splits legados

Os arquivos antigos não serão movidos nem apagados. O path de quarentena conterá `quarantine_manifest.json`, que lista paths originais, tamanhos e hashes, com:

```text
status = QUARANTINED_NOT_DELETED
reason = PATIENT_LEAKAGE_RECORD_DISJOINT_ONLY
active_for_e07r = false
```

O preflight falhará se qualquer workflow PD apontar para os manifests legados.

## 7. Fechamento de integridade

### 7.1 Freeze manifest

Criar `e07r_freeze_manifest.json` com:

- quatro pins E06.5 e hashes herdados;
- checkpoint E07 bloqueado;
- evidence/identity/splits novos;
- árvores `models/`, `backup_v2.3/` e quarentena v3.1;
- source commit aplicável;
- `pinned`, `writable`, `quarantine`, tamanho e SHA-256;
- algoritmo canônico de tree hash.

### 7.2 Proteção

- quatro pins: `0444`;
- arquivos em árvores críticas: `0444` quando seguro;
- diretórios críticos: `0555` quando seguro;
- `chattr +i`: somente se suportado sem privilégio adicional e sem risco; caso contrário registrar `NOT_AVAILABLE`;
- proteção lógica em todo writer Stage 2 por rejeição de paths congelados e evento JSON de violação.

A proteção não pode alterar bytes. Qualquer hash pós-permissão divergente bloqueia.

### 7.3 Preflight E07R

O preflight deve validar no início de cada célula, imediatamente antes de `DONE`, após a matriz e antes de agregação/seleção:

- schema e hashes do preauth/freeze manifest;
- quatro pins e checkpoint bloqueado;
- geração Stage 2 r5 e binding ordenado completos;
- mapping confirmatório completo;
- splits patient/record-disjoint;
- 201/202 juntos;
- manifests legados inativos;
- H6/H11/H12 sem mudança semântica;
- `models/`, backup e quarentena v3.1 intactos;
- permissões somente leitura;
- source/runtime/config identities;
- espaço em disco e determinismo CPU;
- testes requeridos verdes.

## 8. Testes

Aplicar TDD para:

1. contrato e proveniência do mapping;
2. união 201/202;
3. cobertura integral dos records confirmatórios r5 e separação explícita do SVDB;
4. outer patient-disjoint;
5. inner patient-disjoint;
6. record disjoint e cobertura de índices;
7. quarentena legada inativa;
8. freeze manifest e hashes;
9. proteção contra overwrite e evento de violação;
10. preservação de `models/`, backup e quarentena v3.1;
11. sampling train-only e independente de validação/teste;
12. regressões `np.int64` JSON-native e `E06.5 → E06_5`;
13. determinismo e resume/write-once.

Validação obrigatória: testes focados, `make lint`, Pyright, suíte pytest completa e `make test-e2e`. Uma falha sem correção segura produz `BLOCKED`.

## 9. Revalidação E06.5-PD

Matriz: baseline/H6/H11/H12 × folds 1–5 × seeds 17/29/43/71/101 = 100 células.

Fatores congelados:

- dataset r5 content-addressed e derivado do parent r4 sem reinterpretar bytes legados;
- splits v4.0 patient-disjoint;
- MLP-128, CE natural, argmax cru;
- imputação/scaling train-only;
- templates fitados somente em grupos patient-disjoint de treino;
- early stopping inner-only;
- CPU determinística, serial.

Métricas: F1(S/V/F), macro, precision/recall/AP(F), mínimo/std, zero-F1, 208/213, por fold/seed/paciente e matrizes de confusão.

H6-PD fica pré-comprometido como única representação elegível para E07-PD antes de observar qualquer outer test v4. H11/H12 são revalidações descritivas e não podem substituir H6 para E07 usando esses mesmos outer tests. Se H6 não superar baseline de forma robusta, registrar `NO_ROBUST_GAIN` e parar antes de E07-PD.

A inferência primária usa bootstrap pareado por paciente sobre predições OOF, com seeds aninhadas/mediadas dentro de cada fold. As 25 células e win rate permanecem descritivas; seeds não são tratadas como 25 pacientes independentes.

## 10. E07-PD — braços pré-registrados

Todos os braços usam somente a partição de treino e o mesmo orçamento `N_budget = 3 × max(class_count_train)` em inner e outer fit, exceto S0 natural. Nenhum parâmetro será ajustado após observar validação/teste.

| Braço | Semântica congelada |
| --- | --- |
| S0 | dados naturais, sem resampling |
| S1 | slots iguais por classe até `max(class_count_train)`; a probabilidade por linha da classe é `1/(3 × n_class_train)`; nenhum class weight adicional |
| S2 | bootstrap uniforme por patient ID e depois uniforme por linha do paciente; `N_budget` |
| S3 | bootstrap de linhas com `p_i ∝ 4.0` para linha F e `p_i ∝ 1.0` para linha S/V, normalizado sobre o treino; `N_budget` |
| S4 | slots iguais por classe; dentro da classe, patient ID uniforme e linha uniforme; `N_budget` |
| S5 | bootstrap uniforme de linhas do treino, preservando distribuição em expectativa; `N_budget` |

S5 controla o efeito do orçamento/replicação. Seeds do sampler derivam exclusivamente da seed da célula e da partição. O manifest registra fórmula, pesos, contagens, patient shares, hashes de índices e `validation_or_test_sampled=false`.

Matriz: S0–S5 × folds 1–5 × seeds 17/29/43/71/101 = 150 células.

## 11. Métricas e estatística E07-PD

Obrigatórias por célula:

- F1(F), F1(S), F1-macro;
- precision/recall/AP(F);
- confusion matrix e zero-F1;
- por fold, seed, patient e escopo 213;
- margens F e taxa negativa;
- suporte F e pacientes F por fold.

Exploratórias sem virar gate:

- Brier F one-vs-rest;
- ECE com bins pré-fixados;
- curva de calibração;
- confiança em acertos/erros F;
- proxy OOD train-only baseado em distância robusta às features H*-PD de treino.

Análise:

- bootstrap pareado por patient ID sobre predições OOF, 10.000 repetições, com seeds aninhadas/mediadas, contra S0;
- Holm para S1–S5 usando estatística cluster-aware;
- efeito absoluto/relativo e win rate;
- fold 5 preservado;
- Pareto F1(F) × F1(S) × 213 × macro × zero-F1 × fold 5.

Classificação: `INSUFFICIENT`, `PARTIAL_GAIN`, `STRONG_NOT_PUBLICATION_READY` ou `INVALID`. F1(F) < 0,50 mantém publicação `HOLD` em qualquer caso.

## 12. Critérios de aceite e parada

A continuidade exige cumulativamente:

- geração Stage 2 r5 autorizada, mapping e splits novos válidos e congelados;
- zero overlap patient/record em outer e inner;
- 201/202 sempre juntos;
- hashes e permissões válidos;
- árvores protegidas intactas;
- testes verdes;
- E06.5-PD completo e candidato H*-PD válido.

Parar imediatamente para leakage, hash divergente, uso de split legado, teste inseguro, alteração em produção, uso de teste no sampler, ausência de candidato PD ou gate relaxado. Em `BLOCKED`, não gerar métricas E07-PD e emitir checkpoint/relatório específicos.

## 13. Saídas e decisão

Gerar somente artefatos aditivos previstos na missão. O checkpoint final será assinado por `AUTONOMOUS_GOVERNANCE_PREAUTH`, manterá:

```text
publication_ready = false
models_untouched = true
gates_relaxed = false
next_authorized_stage = NONE
```

Mesmo um ganho forte permanece pesquisa interna; nenhuma publicação externa ou promoção está autorizada.
