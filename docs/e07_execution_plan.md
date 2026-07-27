# Plano de execução E07 — auditoria de sampling

- **Data:** 2026-07-26
- **Escopo:** fechamento corretivo E06.5 + E07 analítico sobre H6 congelado
- **Estado do plano:** **BLOCKED antes da implementação**
- **Motivo:** o preflight científico encontrou sobreposição de uma mesma pessoa entre partições dos splits congelados; o critério de parada imediata por leakage foi acionado.

## 1. Escopo autorizado

Se todos os gates de preflight fossem satisfeitos, o trabalho ficaria limitado a:

1. validar e proteger, sem reescrever, os artefatos críticos do E06.5;
2. executar somente E07, alterando exclusivamente o sampler train-only;
3. manter H6, dataset, folds, seeds, arquitetura MLP-128, CE, argmax e gates congelados;
4. produzir evidência analítica, sem promoção, publicação, exportação ou escrita em `models/`.

E08, E09, refit, calibração operacional, seleção de threshold e publicação permanecem fora do escopo.

## 2. Premissas e congelamentos

| Item | Contrato congelado |
| --- | --- |
| Representação | H6, feature manifest selecionado `50762aa120d6ad0f…` |
| Folds | 1–5, manifests outer `d02d284c532da7eed8367c74b89539d79a0a580ddebe11d1730fde88f79f1593` e inner `8dc223b7c08869a88c5a495952f3d473fb8c992fc961a3c04f41f9320fa54b04` |
| Seeds | 17, 29, 43, 71, 101 |
| Dataset | quatro paths pinados no preflight E06.5, hashes listados na seção 3 |
| Arquitetura/loss/decisão | `minimal_mlp_128`; CE multiclasse; softmax argmax cru |
| Seleção/fit | inner-only; outer test proibido para fitting, sampling, priors ou seleção |
| Gates | QG5' e target F1(F) ≥ 0,50 inalterados |
| Produção | `models/` e v2.3 intocados; nenhum modelo E07 promovido |

## 3. Checklist de integridade pré-execução

### 3.1 Paths pinados validados

| Artefato | SHA-256 esperado e observado | Estado |
| --- | --- | --- |
| `data/features/stage2_multiclass.npz` | `68fb0a8e9fa3bc3fd06df7af074222c9837b8a04dc644b9f823d26919ba6b04a` | PASS |
| `data/features/stage2_multiclass.parquet` | `870b386e160790cce5a102ccc447e736c5f5381e76ef9fd5fba7cde3ee9d3802` | PASS |
| `data/features/finetuning_mitbih_family.npz` | `6b38eaf8e118ea25190459c83c90272a14f5dfd08c6b9e47d3e61a8aa236a54a` | PASS |
| `data/features/finetuning_mitbih_family.parquet` | `9efe6681cdef7c48cabc26992bcfba3011315d580531af9007fbbfd532330c26` | PASS |

O commit de regeneração `886f1b398bf34f0e6c5b14d9409822f7c2574bf9` existe no repositório. Os bytes v3.1-era permanecem em `data/features/quarantine_v31_working_20260718/`.

### 3.2 Baselines de árvores protegidas

- `models/`: 35 arquivos; tree SHA-256 `32cb3d59014775008ee627d08308a2f81f34ce9ea32f93c9630f349fe2b183dd`.
- `data/features/backup_v2.3/`: 7 arquivos; tree SHA-256 `5531577f700914c221ef7fb8f9c76f5672b49a534270d82389669823ff61a56b`.
- `data/features/quarantine_v31_working_20260718/`: 4 arquivos; tree SHA-256 `dfd12b3e56d6d32167e0c25ab228a575b9eb2fd0e24df92470fefcb94a5779f7`.
- `git status --porcelain` não apontou alteração nesses paths.

### 3.3 Gate de leakage — FAIL

O parquet congelado contém apenas `record_id` como identidade de grupo; não contém `patient_id`, `subject_id` nem `group_id`. O loader confirma `group_key: record_id`.

A documentação oficial PhysioNet do MIT-BIH declara literalmente: **“Records 201 and 202 came from the same male subject.”** Fonte: <https://physionet.org/physiobank/database/html/mitdbdir/intro.htm>.

Nos splits congelados:

| Fold | Registro da mesma pessoa no treino | Registro da mesma pessoa fora do treino | Violação |
| --- | --- | --- | --- |
| outer 3 | 201 | 202 em outer test | train/test patient overlap |
| outer 5 | 202 | 201 em outer test | train/test patient overlap |
| inner 1 | 201 em inner train | 202 em inner validation | train/validation patient overlap |

Logo, `overlap_groups == []` prova somente disjointidade por **registro**, não por pessoa. O requisito “nenhum paciente simultaneamente em treino/validação/teste” falha com evidência positiva, e não apenas por ausência de metadados.

## 4. Braços E07 planejados, mas não executados

Todos seriam determinísticos, train-only, 5 folds × 5 seeds, com H6 fixo:

| Braço | Estratégia congelável proposta |
| --- | --- |
| S0 | distribuição natural; controle de reprodução E06.5 |
| S1 | bootstrap ponderado por frequência inversa de classe, pesos calculados somente no treino |
| S2 | balanceamento uniforme por paciente/grupo autenticado, com diagnóstico de contribuição máxima por paciente |
| S3 | upweight/reamostragem exclusiva de F com fator explícito e versionado, estimado sem consultar validação/teste |
| S4 | amostragem estratificada por paciente autenticado × rótulo, somente no treino |
| S5 | controle negativo aleatório, mesmo orçamento e cardinalidade, sem correção estrutural |

Este conjunto segue a instrução corrente, que exige exatamente S0–S5. Ele diverge do desenho legado do documento mestre (que incluía SMOTE); antes de qualquer futura execução, a governança deve registrar explicitamente a precedência do contrato corrente. Como o preflight bloqueou, nenhum braço ou parâmetro foi congelado operacionalmente.

O fator S3 deveria ser pré-registrado antes de qualquer métrica de validação/teste. Nenhum braço alteraria threshold, loss, arquitetura, features ou representação.

## 5. Métricas e artefatos planejados

Por braço/fold/seed:

- F1(F), F1(S), F1-macro, precision(F), recall(F), AP/PR-AUC(F);
- matriz de confusão; média, desvio, mínimo e zero-F1 runs;
- escopos 208, 213 e fora de 208/213;
- métricas por paciente autenticado e contagens F por paciente/fold;
- margem `logit_F - max(logit_S, logit_V)`, taxa de margens negativas e confiança em erros F;
- Brier, ECE e curva de calibração apenas descritivos, sem ajustar calibrador/threshold no outer test;
- proxy OOD pré-registrado antes dos resultados, sem transformá-lo em gate.

Todos os arquivos novos seriam publicados em paths E07 write-once, com manifestos e SHA-256. Checkpoints `.keras` permaneceriam somente sob `experiments/`, nunca em `models/`.

## 6. Análise estatística planejada

1. bootstrap pareado de 10.000 repetições sobre as 25 células fold×seed;
2. deltas absolutos e relativos contra S0;
3. comparações pareadas por fold/seed;
4. correção de Holm entre S1–S5;
5. efeito em zero-F1, pior fold e fold 5;
6. classificação prática congelada: insuficiente, ganho parcial, ganho forte ainda não publicável ou melhora inválida;
7. Pareto F1(F) × F1(S) × escopo 213 × macro × zero-F1 × fold 5.

## 7. Testes planejados

- manifesto/hash e árvores protegidas;
- bloqueio de escrita em paths pinados com evento JSON de violação;
- `models/`, backup e quarentena intactos;
- regressões `np.int64 → int/string JSON-native` e `E06.5 → E06_5` preservadas;
- determinismo e semântica de S0–S5;
- sampler recebe somente índices de treino e não acessa estatísticas de validação/teste;
- disjointidade por paciente autenticado em outer e inner splits;
- ausência de NaN/Inf, cardinalidades e proveniência de amostras;
- suíte original, lint, Pyright, testes focados e gates aplicáveis.

## 8. Critérios de aceite

A execução só poderia iniciar se, cumulativamente:

- todos os hashes críticos e árvores protegidas conferissem;
- pins estivessem protegidos e o preflight E07 fosse write-once, sem sobrescrever o preflight E06.5;
- uma identidade de pessoa autenticada estivesse disponível para todas as linhas;
- outer e inner splits fossem disjuntos por essa identidade;
- H6/dataset/gates permanecessem congelados;
- os seis braços e suas fórmulas fossem pré-registrados;
- todos os testes de integridade/leakage passassem.

O gate de identidade/split falhou; portanto os demais critérios não autorizam execução.

## 9. Riscos

- leakage biológico oculto por agrupamento por registro;
- estimativa de sampling contaminada por validação/teste;
- sobrescrita de pins por produtores externos;
- interpretação de ganho concentrado em 208/213 como generalização;
- mascaramento do fold 5;
- tuning do fator S3 após observar resultados;
- mudança de source/config exigindo novo preflight sem reescrever evidência E06.5;
- uso de proxy OOD pós-hoc.

## 10. Critérios de parada

Parada imediata para hash divergente, alteração em `models/`/backup/quarentena, escrita em pin, leakage, uso de estatística de teste, remoção do fold 5, mudança em H6/dataset/gates, teste inseguro ou ambiguidade científica.

**Critério acionado:** leakage por paciente nos splits congelados (outer folds 3/5 e inner fold 1).

## 11. Decisão deste plano

- **E07:** BLOCKED antes de implementação/treino.
- **Sampling executado:** nenhum.
- **Modelos promovidos:** nenhum.
- **Próxima etapa autorizada:** nenhuma.
- **Condição futura, não autorizada nesta missão:** governança humana deve fornecer mapeamento record→patient autenticado e aprovar uma nova geração de splits. Isso altera o protocolo congelado e exigiria nova evidência comparável; não pode ser feito como “correção” silenciosa do E06.5/E07.
