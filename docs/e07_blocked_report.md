# E07 — relatório de bloqueio científico

- **Data:** 2026-07-26
- **Status:** **BLOCKED**
- **Momento do bloqueio:** preflight, antes de alterar código, executar testes de implementação ou iniciar qualquer célula E07
- **Regra acionada:** parada imediata por leakage de paciente
- **Checkpoint:** `experiments/stage2_v2.4_research/E07_blocked_20260726.json`
- **SHA-256 do checkpoint:** `d8a684c0e29a91033a60fa10c4f9257c7ece1059a19ebe5472b2accc70749f63`

## 1. Resumo executivo

O E07 não pode ser executado honestamente sobre os splits congelados atuais. O protocolo Stage 2 v2.4 usa `record_id` como chave de grupo e prova apenas que registros não se sobrepõem. A base MIT-BIH, porém, contém dois registros da mesma pessoa: 201 e 202. Os splits congelados colocam esses registros em lados opostos de partições outer e inner.

Assim, uma mesma pessoa aparece simultaneamente em treino/teste nos folds outer 3 e 5 e em treino/validação no fold inner 1. Qualquer comparação S0–S5 herdaria leakage biológico e seria classificada como **MELHORA APARENTE/INVÁLIDA**, independentemente da métrica observada.

Nenhum braço de sampling foi implementado ou executado. Nenhum modelo foi promovido. `models/`, backup e quarentena permanecem intocados.

## 2. Evidência do leakage

### 2.1 Contrato de identidade disponível

O artefato congelado `data/features/stage2_multiclass.parquet` contém:

- `record_id`;
- `dataset`;
- chaves de batimento e features.

Não contém:

- `patient_id`;
- `subject_id`;
- `group_id` autenticado.

O loader em `src/stage2_research/data.py` materializa explicitamente:

```text
groups = frame["record_id"]
group_key = "record_id"
```

Portanto, `overlap_groups == []` nos manifests significa **record-disjoint**, não necessariamente patient-disjoint.

### 2.2 Identidade oficial dos registros 201/202

A documentação oficial do MIT-BIH/PhysioNet afirma:

> “Records 201 and 202 came from the same male subject.”

Fonte primária: <https://physionet.org/physiobank/database/html/mitdbdir/intro.htm>.

O parquet congelado contém ambos:

- MIT-BIH 201: 328 amostras Stage 2;
- MIT-BIH 202: 75 amostras Stage 2.

### 2.3 Partições afetadas

| Nível | Fold | Treino | Partição fora do treino | Veredito |
| --- | ---: | --- | --- | --- |
| outer | 3 | registro 201 | registro 202 em outer test | **LEAKAGE** |
| outer | 5 | registro 202 | registro 201 em outer test | **LEAKAGE** |
| inner | 1 | registro 201 | registro 202 em inner validation | **LEAKAGE** |

Os registros estão separados como grupos, mas pertencem à mesma pessoa. O fold 5, que deveria ser diagnosticado sem mascaramento, está diretamente afetado.

## 3. Arquivos afetados pela causa

Nenhum arquivo foi alterado para produzir o defeito nesta tentativa. Os artefatos que materializam ou consomem o contrato insuficiente são:

- `data/features/stage2_multiclass.parquet` — não possui identidade de pessoa;
- `src/stage2_research/data.py` — usa `record_id` como grupo;
- `experiments/stage2_v2.4_research/splits/outer_splits_v2.4.json` — separa 201/202 nos folds outer 3/5;
- `experiments/stage2_v2.4_research/splits/inner_splits_v2.4.json` — separa 201/202 no fold inner 1.

Esses artefatos são congelados para a missão atual e **não foram corrigidos, reinterpretados nem sobrescritos**.

## 4. Integridade observada antes do bloqueio

### 4.1 Quatro pins E06.5

| Path | SHA-256 esperado/observado | Estado |
| --- | --- | --- |
| `data/features/stage2_multiclass.npz` | `68fb0a8e9fa3bc3fd06df7af074222c9837b8a04dc644b9f823d26919ba6b04a` | PASS |
| `data/features/stage2_multiclass.parquet` | `870b386e160790cce5a102ccc447e736c5f5381e76ef9fd5fba7cde3ee9d3802` | PASS |
| `data/features/finetuning_mitbih_family.npz` | `6b38eaf8e118ea25190459c83c90272a14f5dfd08c6b9e47d3e61a8aa236a54a` | PASS |
| `data/features/finetuning_mitbih_family.parquet` | `9efe6681cdef7c48cabc26992bcfba3011315d580531af9007fbbfd532330c26` | PASS |

### 4.2 Evidência E06.5 e splits

| Artefato | SHA-256 dos bytes |
| --- | --- |
| `experiments/stage2_v2.4_research/E065_recovery_20260725.json` | `74067c68ff735d450c1e36d3468e85b6022ba6dc8e0b7b5e2de2a9413ead90ca` |
| `experiments/stage2_v2.4_research/reports/e065_verify.json` | `6fc6a4d1e41dd7d722bd6e50b499eb0d9aa20f56b898316239402f9e9572c2af` |
| `experiments/stage2_v2.4_research/selections/representation_selection.json` | `4296aa322a64107656149742f01913369b25c9b5fca98e9d033803c4b32f7538` |
| `experiments/stage2_v2.4_research/splits/outer_splits_v2.4.json` | `f496e93784e4622dcab4587bcade3223b26660c17c6a98f79fa007e5ee066845` |
| `experiments/stage2_v2.4_research/splits/inner_splits_v2.4.json` | `1eb6e059a60cacb4a977662e20b4e0203a9701aa4848b78100d07ef319eb903e` |
| `docs/e07_execution_plan.md` | `668dcdbc778ba096f2bb39c6d719fbf389dfdb80736d8fa1e4bdaf64340c7c2f` |

### 4.3 Árvores protegidas

| Árvore | Arquivos | Tree SHA-256 | Git status |
| --- | ---: | --- | --- |
| `models/` | 35 | `32cb3d59014775008ee627d08308a2f81f34ce9ea32f93c9630f349fe2b183dd` | limpo |
| `data/features/backup_v2.3/` | 7 | `5531577f700914c221ef7fb8f9c76f5672b49a534270d82389669823ff61a56b` | limpo |
| `data/features/quarantine_v31_working_20260718/` | 4 | `dfd12b3e56d6d32167e0c25ab228a575b9eb2fd0e24df92470fefcb94a5779f7` | limpo |

Algoritmo reproduzível do tree hash: ordenar todos os arquivos regulares pelo path POSIX relativo e, para cada arquivo, alimentar SHA-256 com `relative_path + NUL + file_sha256_lowercase + NUL + size_bytes_decimal + LF`, em UTF-8. Nenhum symlink estava presente nas três árvores.

O commit `886f1b398bf34f0e6c5b14d9409822f7c2574bf9` existe no repositório.

## 5. Salvaguardas de sobrescrita — estado honesto

A proteção corretiva ainda não foi implementada porque a instrução corrente da missão, seção 8, exige parada imediata ao detectar leakage. No momento da inspeção, os quatro paths pinados tinham modo `0664`, portanto continuavam graváveis pelo owner/group.

Isso não significa que uma remediação futura exclusivamente de integridade seja cientificamente proibida; significa apenas que ela requer autorização humana separada e não autoriza treino E07. Esta tentativa não continuou após o blocker.

Consequências:

- hashes atuais: válidos;
- nova sobrescrita: ainda não impedida pelo fluxo solicitado;
- manifesto `e065_freeze_manifest.json`: não criado;
- testes de write violation: não adicionados;
- conclusão: lacuna de integridade permanece aberta e não deve ser mascarada como PASS.

## 6. Testes e execução experimental

- Células planejáveis pela instrução corrente: 6 braços S0–S5 × 5 folds × 5 seeds = 150.
- O contrato corrente diverge do desenho legado com SMOTE; qualquer futura autorização deve registrar explicitamente qual contrato prevalece antes de pré-registrar os braços.
- Células iniciadas: **0**.
- Células concluídas: **0**.
- Testes executados nesta tentativa E07: **0**; o bloqueio ocorreu antes da implementação.
- Evidência herdada E06.5: 116 testes passados, registrada no checkpoint E06.5, **não revalidada nesta tentativa**.
- Parsing JSON: **PASS**.
- Validação ad hoc de um subconjunto estrito com Pydantic v2: **PASS**; nenhum schema canônico de checkpoint E07 foi introduzido porque a implementação parou.

Não há métricas E07, IC95, deltas, zero-F1 antes/depois nem Pareto legítimos para reportar.

## 7. Decisão

```text
E07 = BLOCKED
PUBLICAÇÃO = HOLD
MODELS = INTOCADOS
GATES = NÃO AFROUXADOS
PRÓXIMA ETAPA AUTORIZADA = NENHUMA
```

Não é permitido tratar os resultados E06.5 record-disjoint como prova patient-disjoint para contornar o bloqueio. Também não é permitido corrigir silenciosamente os splits congelados e continuar a mesma campanha.

## 8. Ação recomendada — condicional, não autorizada

A governança humana deve, em uma missão futura e separada:

1. fornecer um mapeamento `dataset + record_id → patient_id` autenticado e versionado para todas as linhas;
2. validar cobertura, unicidade e proveniência dessa identidade;
3. autorizar explicitamente uma nova geração imutável de splits outer/inner patient-disjoint;
4. reconhecer que novos splits alteram o protocolo congelado e exigem renovar a evidência comparável de representação antes do E07;
5. manter os artefatos E06.5 atuais como evidência histórica, sem reescrita.

Esta recomendação não autoriza E08, E09, publicação, promoção ou qualquer execução adicional.
