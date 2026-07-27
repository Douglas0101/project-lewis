# Relatório de Causa-Raiz — Research Branch Stage 2 v2.4 (classe F, fusion beat AAMI)

- **Documento:** fechamento da research branch v2.4 do Stage 2
- **Data:** 2026-07-26
- **Doc mestre:** `docs/STAGE 2 RESEARCH BRANCH v2.4.md`
- **Histórico do ciclo E00–E09:** `docs/stage2_v2.4_research_report.md` (não sobrescrito)
- **Campanha final:** `experiments/stage2_v2.4_research/E06_5/e065-audit-v3/` (100 células DONE)
- **Veredito de publicação:** `RESEARCH_CANDIDATE_NOT_PUBLICATION_READY` (detalhes em `docs/stage2_v2.4_publication_readiness.md`)
- **Checkpoint final da branch:** `PASS` (fechamento auditado) — release E06.5 = `E06_5_PASS_REPRESENTATION_SELECTED`; target de publicação **NÃO atingido**

## Resumo executivo

A branch v2.4 determinou, com protocolo de auditoria determinístico (4 candidatos × 5 folds × 5 seeds = 100 runs, CPU-only, sem seleção por outer test), que a causa dominante do platô inter-paciente de F1(F) no Stage 2 é a **causa A — escassez e concentração da classe F por paciente/registro** (70,31% de todos os beats F estão nos registros 208 e 213; mediana de 3 beats F por grupo), com teto de representação comprovado como causa secundária (B). A engenharia de representação (H6/H11/H12) produziu ganho real e pareado de ≈ +0,10 de F1(F) sobre o baseline (~3×: 0,042 → 0,137–0,147) e 6–7× fora de 208/213, mas o teto absoluto ficou em F1(F)=0,147 — abaixo do gate QG5' Estágio 2 (≥ 0,15) e muito abaixo do target de publicação (≥ 0,50). No fold mais duro (fold 5, outer test 100% fora de 208/213) todos os candidatos colapsam (F1(F) ≤ 0,015; 99,44% de margens F negativas), classificado como `OPTIMIZATION_COLLAPSE`. Nenhum artefato em `models/` foi tocado; v2.3 segue como produção.

---

## 1. ETAPA EXECUTADA

```text
Fechamento da research branch v2.4: E00 → E09 → E06R (triagem H1–H12) → E06.5 (auditoria multi-seed)
```

Esta etapa de fechamento consolida o ciclo E00–E09 (2026-07-11, relatado em `docs/stage2_v2.4_research_report.md`), a reabertura E06R (2026-07-14, triagem de 12 hipóteses de representação) e a campanha de auditoria E06.5 (concluída em 2026-07-26T00:46Z com `e065-audit-v3`), incluindo o evento de recuperação do dataset de 2026-07-25.

## 2. HIPÓTESE

"O platô inter-paciente de F1(F) do Stage 2 é dominado por uma causa identificável entre A–I; se a causa fosse representação (B), uma representação corrigida elevaria F1(F) inter-paciente ao target de 0,50 sob protocolo de auditoria."

**Resultado:** a primeira parte foi **confirmada** (causa dominante A identificada e quantificada); a segunda parte foi **refutada** (representação corrigida elevou F1(F) para 0,147, não para 0,50).

## 3. ALTERAÇÕES IMPLEMENTADAS

Nesta etapa de fechamento, apenas documentação:

- `docs/stage2_v2.4_root_cause_report.md` (este documento, novo)
- `docs/stage2_v2.4_publication_readiness.md` (novo)

Nenhum arquivo em `src/`, `tests/`, `config/` ou `models/` foi modificado. Documentos preexistentes (`docs/stage2_v2.4_research_report.md`, `docs/stage2_research_cli.md`, `docs/stage2_e065_robustness_report.md`, `docs/stage2_fold5_root_cause.md`) **não** foram sobrescritos e são citados como fontes.

## 4. VARIÁVEL CONCEITUAL ALTERADA

Na campanha de auditoria final (E06.5): **representation** (única variável entre as células).

- `baseline`: 16 features tabulares v2.4 (schema `stage2-dataset-v2.4`, hash de features `244fe5cb…`)
- `H6`: contexto RR causal + templates de classe N/V/F (banco F com 8 templates; hash `50762aa1…`)
- `H11`: idem H6, banco F com 16 templates (hash `8d899c1b…`)
- `H12`: idem H6, banco F com 24 templates (hash `fe36ea00…`)

Fixos em todas as células: dataset (`stage2_multiclass.npz` SHA-256 `68fb0a8e…`), splits congelados (outer `d02d284c…`, inner `8dc223b7…`), loss = cross-entropy multiclasse (`method: ce_control`), sampling = distribuição natural, decisão = argmax do softmax, arquitetura MLP minimal, seeds {17, 29, 43, 71, 101}, folds 1–5.

## 5. CHECAGEM PÓS-OPERATÓRIA

Evidência: `experiments/stage2_v2.4_research/reports/e065_verify.json` (status `E06_5_PASS_REPRESENTATION_SELECTED`), `experiments/stage2_v2.4_research/manifests/preflight.json` (status `PREFLIGHT_PASS`), `experiments/stage2_v2.4_research/manifests/e065_smoke_gate.json` (`E06_5_SMOKE_PASS`, smoke canônico de 4 células fold 1/seed 17).

```text
[PASS] testes (bateria E06/pesquisa: 8 arquivos de teste, exit 0 — preflight validation_commands)
[PASS] lint/estática (make lint + pyright src tests, exit 0 — preflight validation_commands)
[PASS] shapes (signal_shape [55161, 500, 1]; contratos Pydantic por run)
[PASS] NaN/Inf (contratos de dados por run; nenhum run descartado — verify.no_seed_discarded)
[PASS] dataset manifest (stage2_npz_sha256 68fb0a8e… idêntico ao pino; manifest_hash 168224d0…)
[PASS] feature schema (4 manifests pinados; seleção gravada com selected_feature_manifest_hash)
[PASS] split integrity (outer/inner manifest hashes encadeados em todos os runs)
[PASS] no group overlap (outer_overlap_count=0; template_source_group_overlap=0 nos 5 folds)
[PASS] artifacts isolated (nada em models/; campaign root experiments/stage2_v2.4_research/E06_5/)
[PASS] no outer-test selection (outer_test_used_for_selection=false; verify.no_best_fold_selection)
[PASS] scaler/template fit train-only (scaler_leakage_count=0; template_leakage_count=0)
[PASS] determinismo (CPU-only, deterministic_requested=true; save/reload Δ=0,0; prediction_equivalence=true)
[PASS] 100/100 células DONE (verify.100_runs_complete; 100 marcadores DONE em e065-audit-v3)
[NOT RUN] auditoria CPU/GPU cruzada (E11) — ambiente de auditoria sem GPU; fora do escopo desta etapa
[NOT RUN] E07 (sampling), E08 (long-tail), E09 (decisão/calibração), E10 (limites de estabilidade) no protocolo de auditoria
```

## 6. RESULTADOS

Médias sobre 25 runs por candidato (5 folds × 5 seeds), computadas de `experiments/stage2_v2.4_research/E06_5/e065-audit-v3/<candidato>/fold_*/seed_*/metrics.json` e idênticas ao agregado oficial `summary.json`:

| Métrica | baseline | H6 (selecionado) | H11 | H12 |
| --- | ---: | ---: | ---: | ---: |
| F1(F) mean ± std | 0,0423 ± 0,0442 | 0,1373 ± 0,0947 | **0,1470 ± 0,0960** | 0,1449 ± 0,0944 |
| F1(F) min / max | 0,000 / 0,121 | 0,000 / 0,303 | 0,000 / 0,302 | 0,000 / 0,316 |
| Runs com F1(F)=0 | 11/25 (folds 2,4,5) | 2/25 (fold 5) | 2/25 (fold 5) | 2/25 (fold 5) |
| macro-F1 mean | 0,4445 | 0,4794 | **0,4824** | 0,4823 |
| precision(F) mean | 0,188 | 0,416 | **0,429** | 0,429 |
| recall(F) mean | 0,028 | 0,091 | **0,099** | 0,097 |
| AP(F) mean | 0,178 | **0,313** | 0,312 | 0,312 |
| F1(F) fora de 208/213 | 0,013 | 0,080 | 0,092 | **0,097** |
| F1(S) mean | 0,489 | 0,501 | 0,500 | 0,502 |
| F1(V) mean | 0,802 | 0,800 | 0,800 | 0,800 |

Comparação pareada (mesmos folds/seeds; `summary.json` → `paired_vs_baseline`):

| Par | Δ F1(F) mean | IC95 | win fraction |
| --- | ---: | --- | ---: |
| H11 − baseline | +0,1047 | [+0,0641; +0,1460] | 0,80 |
| H12 − baseline | +0,1026 | [+0,0641; +0,1414] | 0,80 |
| H6 − baseline | +0,0950 | [+0,0567; +0,1338] | 0,80 |
| H11 − H6 | +0,0097 | [−0,000003; +0,020360] | 0,56 |

Leitura honesta: os três candidatos de representação superam o baseline com IC95 exclusivamente positivo (ganho real de representação), mas **H11 e H12 não superam H6 além da variância entre seeds** (IC95 do Δ H11−H6 inclui ~0/negativo). Por isso a seleção congelada elegeu **H6** (`selections/representation_selection.json`, `selected_name=H6`, política "não promover complexidade sem ganho estável"; ver também `docs/policies/authenticated_research_decision_v1.md` §3.2: "H11 não substitui H6 se o ganho não exceder variabilidade entre seeds").

## 7. ANÁLISE POR FOLD

F1(F) médio por fold (5 seeds; computado dos `metrics.json`):

| Fold | baseline | H6 | H11 | H12 | Conteúdo F do outer test |
| --- | ---: | ---: | ---: | ---: | --- |
| 1 | 0,054 | 0,289 | 0,279 | 0,287 | contém registro 208 (372 F) |
| 2 | 0,094 | 0,094 | 0,089 | 0,094 | misto |
| 3 | 0,063 | 0,114 | 0,131 | 0,134 | misto |
| 4 | 0,000 | 0,179 | 0,222 | 0,195 | misto |
| 5 | 0,000 | 0,012 | 0,015 | 0,015 | **99 F / 11 pacientes, 0 beats de 208/213** |

O desempenho dos candidatos no fold 1 (onde 208 está no teste) é ~19–25× o do fold 5 (onde todo F de teste é de pacientes "não dominantes"). A variância **entre folds** (0,012–0,289 em H6) domina a variância entre seeds dentro de cada fold.

**Fold 5 — causa-raiz dedicada** (`fold_audits/e065-audit-v3/fold5_report.json`, status PASS, classificação `OPTIMIZATION_COLLAPSE`):

- Partições: outer_train 945 F/34 pacientes (208 e 213 presentes, 372+362); inner_train 559 F/28 (208); inner_validation 386 F/6 (213); outer_test 99 F/11 pacientes — **todos fora de 208/213**.
- `negative_margin_fraction = 0,9944` sobre 1.980 observações F verdadeiras (99 beats × 4 candidatos × 5 seeds): em 99,44% das vezes a margem da classe F foi negativa — o score de F ficou abaixo do de outra classe.
- Confusão fold 5 (`fold5_F_confusion.csv`): dos 99 F reais por run, 85–96 são preditos V e 3–12 preditos S; no máximo 2/99 preditos F (baseline: 0/99 em todas as seeds).
- Conclusão do audit: mesmo com 208/213 integralmente no treino, o modelo não transfere F para os demais pacientes; no regime mais duro a otimização colapsa para "quase nunca prever F" — não é falha de integridade (status PASS), é evidência científica negativa.

## 8. ANÁLISE DA CLASSE F

Separação obrigatória 208 / 213 / demais grupos F:

- **Distribuição** (`E01_patient_distribution/f_concentration_report.json`): 1.044 beats F (1,89% de 55.161), 45 grupos com F e 152 sem; registro 208 = 372 F (35,63% de todo F), registro 213 = 362 F (34,67%) — **acumulado 70,31%**; top-3 (I18, INCART) eleva a 75,67%; mediana de 3 beats F por grupo (média 23,2, desvio 74,9); índice Herfindahl-like 0,254 (diagnóstico interno).
- **Escopos por run** (média ± std sobre 25 runs, do campo `scopes` dos `metrics.json`, H11): F1(F) no recorte 208 = 0,059 ± 0,121; no recorte 213 = 0,018 ± 0,039; fora de 208/213 = 0,092 ± 0,082. Os recortes 208/213 concentram zeros porque cada registro só aparece no outer test de um fold — nos demais folds está no treino e não é medido; o contraste limpo é por fold (seção 7): fold 1 (208 no teste) = 0,279–0,289 vs fold 5 (nenhum dos dois no teste) = 0,012–0,015.
- **Histórico E05** (`docs/stage2_v2.4_research_report.md`): leave-group-out com as 16 features — treinar sem 208 e testar em 208 → F1(F)=0,02; treinar sem 213 e testar em 213 → F1(F)=0,40; mutual information F-vs-resto < 0,03; features informativas dominadas por RR.
- **Ganho fora de 208/213 (E06.5):** baseline 0,013 → H6 0,080 / H11 0,092 / H12 0,097 (6–7,7×). Material, porém em nível absoluto ≤ 0,10.

## 9. REGRESSÕES

- **Métricas:** nenhum candidato regridiu o baseline em F1(F) pareado (Δ mínimo −0,052 pontual em 1 run de 25 para H6; mediana do Δ +0,068). F1(S) ficou essencialmente estável (0,489 → 0,500–0,502) e F1(V) estável (~0,80); macro-F1 subiu 0,445 → 0,479–0,482. Não houve regressão de métrica agregada.
- **Evento de integridade do dataset (2026-07-18 → 2026-07-25):** a geração v3.1 (trabalho do commit `ed53dd8`) sobrescreveu em 2026-07-18 12:56 (-03:00) os 4 paths raiz pinados pelo preflight E06.5 de 2026-07-16 (`stage2_multiclass.npz/.parquet`, `finetuning_mitbih_family.npz/.parquet`). Detectado em 2026-07-25 01:05 (-03:00) como `BLOCKED` (preflight exit_code=6, hash mismatch `07841722…` vs pino `68fb0a8e…`). Recuperação byte-exata validada por SHA-256: 3 arquivos restaurados de backup de 2026-07-11 e `finetuning_mitbih_family.parquet` regenerado deterministicamente (código congelado `886f1b3`, `uv.lock` idêntico ao pino, match byte-exato); bytes da era v3.1 preservados em quarentena (`data/features/quarantine_v31_working_20260718/`). Três rebinds da auditoria (`e065-audit-v1`, `-v2`, `-v3`) produziram **métricas idênticas** (baseline 0,042329 / H6 0,137336 / H11 0,147036 / H12 0,144944 nas três), divergindo apenas no `aggregate_hash` (metadados de rebinding). Registro completo: `experiments/stage2_v2.4_research/E065_recovery_20260725.json`.
- **Artefatos v2.3:** nenhum sobrescrito; `git status` de `models/` limpo ao final da campanha (última alteração em `models/` = commit `886f1b3` de 2026-07-16, restrito a `models/quantized/` de firmware).

## 10. HIPÓTESE

| Hipótese | Classificação | Evidência |
| --- | --- | --- |
| Branch: causa dominante identificável | **SUPPORTED** | Causa A dominante; ver Anexo B |
| Branch: representação corrigida atinge target | **REJECTED** | Melhor média F1(F)=0,147 < 0,50 |
| E06.5: sinal de representação existe | **SUPPORTED (parcial)** | Δ pareado +0,095…+0,105, IC95 > 0; status `REPRESENTATION_SIGNAL_CONFIRMED / TARGET_NOT_MET` |
| E06R H1–H12 (cada uma "melhora materialmente fora de 208/213") | **REJECTED** (12/12 `PASS_HYPOTHESIS_REJECTED`) | Ganho material fora de 208/213 apenas em H6/H8/H10/H11/H12; nenhuma atingiu target; manifests `E06_reopened/*/E06R_*_manifest.json` |

## 11. CHECKPOINT

```text
PASS (fechamento da branch, com target de publicação NÃO atingido)
```

- Release E06.5: `E06_5_PASS_REPRESENTATION_SELECTED` (`reports/e065_verify.json`; todos os 8 checks verdadeiros).
- Estado científico (`docs/policies/authenticated_research_decision_v1.md` §3.2): `REPRESENTATION_SIGNAL_CONFIRMED / TARGET_NOT_MET / ROBUSTNESS_VALIDATION_REQUIRED` — a validação de robustez multi-seed foi executada (100 runs) e o status de release é válido; o `TARGET_NOT_MET` permanece.
- Publicação: **negada** — `RESEARCH_CANDIDATE_NOT_PUBLICATION_READY` (sem afrouxar gates, sem reduzir target; ver `docs/stage2_v2.4_publication_readiness.md`).
- Resultados negativos preservados: ciclo E00–E09 arquivado; 100 runs de `e065-audit-v3` completos com manifests, predições e hashes; fold 5 auditado e classificado.

## 12. PRÓXIMA ETAPA AUTORIZADA

```text
E07 — AUDITORIA DE SAMPLING (via e07-run)
```

Exatamente uma etapa, conforme o doc mestre e a policy §3.2 ("E07/E08 permanecem bloqueados até release válido de E06.5" — a release é válida desde 2026-07-26T00:46Z, logo E07 está tecnicamente desbloqueada). Escopo obrigatório: representação **H6 congelada** (feature hash `50762aa1…`), loss CE, matriz 5 folds × 5 seeds idêntica à de E06.5, comparando `natural` (baseline E06.5 já medido), `random oversampling`, `SMOTE` (com auditoria de geometria) e `patient-aware sampling`, com seleção inner-only e gates formais. A avaliação detalhada da evidência sobre se sampling pode fechar a lacuna está em `docs/stage2_v2.4_publication_readiness.md` §4 — expectativa honesta: ganho parcial provável, fechamento da lacuna 0,147→0,50 não suportado pela evidência existente.

---

## Anexo A — Respostas objetivas às 11 perguntas finais do doc mestre

**1. Por que v11–v16 ficaram no platô de F1(F)?**
Todas as variantes (focal, class-weight, SMOTE, MLP, thresholds) operaram sobre a mesma representação de 16 features, cujo sinal F é dominado por 2 registros (70,31% de todo F em 208/213). O QG5 balanceado (até ~683 exemplos/classe) mascarava o problema: F1(F)=0,9379 no subset balanceado vs 0,1627–0,214 inter-paciente. O platô ~0,21–0,24 era o teto da representação+distribuição, não do otimizador. Evidência: `E04_qg5_gates/qg5_v2.4_report.json` (PATIENTWISE FAIL para v14), `docs/stage2_v2.4_research_report.md` §Diagnóstico, `E05` (LGO sem 208 → 0,02).

**2. Quanto da dificuldade é explicada pela concentração 208/213?**
A maior parte. 70,31% dos beats F estão em 2 registros; 3º maior grupo tem só 56 F; mediana 3 F/grupo. Quando 208 está no teste (fold 1), os candidatos atingem F1(F)≈0,28–0,29; quando o teste é 100% fora de 208/213 (fold 5: 99 F/11 pacientes), caem para ≈0,012–0,015 com 99,44% de margens negativas — mesmo com 208/213 integralmente no treino. O baseline zera 11/25 runs. Evidência: `E01_patient_distribution/f_concentration_report.json`, `fold_audits/e065-audit-v3/fold5_report.json`, `summary.json` (`outside_208_213_F1_F`).

**3. As 16 features contêm fronteira generalizável para F?**
Não. Baseline de auditoria: F1(F)=0,0423±0,0442, 11/25 runs zero, AP_F=0,178, recall_F=0,028, fora de 208/213 = 0,013. E05 já havia medido MI < 0,03 e colapso leave-group-out. Evidência: `summary.json` (candidato `baseline`), `docs/stage2_v2.4_research_report.md` §Diagnóstico.

**4. Context features melhoram pacientes não dominantes?**
Sim, materialmente, mas insuficiente. Fora de 208/213: 0,013 → 0,080 (H6) / 0,092 (H11) / 0,097 (H12) — 6–7,7×. A triagem E06R marcou ganho material fora de 208/213 em H6/H8/H10/H11/H12. Porém o nível absoluto ≤ 0,10 e o fold 5 permanece ≈0,01: o ganho não se sustenta no regime mais duro. Evidência: `summary.json`, `E06_reopened/*/E06R_*_manifest.json`.

**5. Qual estratégia long-tail é realmente consistente?**
Nenhuma demonstrou consistência sob gates. No ciclo anterior (protocolo sem gates formais): reamostragem por paciente elevou F1(F) para 0,4654±0,1018; focal+class-weight sobre o reamostrado não adicionou nada (0,453±0,082 — abaixo da reamostragem isolada). No protocolo de auditoria E06.5, estratégias long-tail foram mantidas fixas (CE natural) para isolar representação — E07/E08 do doc mestre ainda não foram executados nesse regime. Evidência: `E07_label_audit/baseline_resampled/baseline_enhanced_metrics.json`, `docs/stage2_v2.4_research_report.md` §Resultados.

**6. SMOTE ajuda ou prejudica neste dataset?**
Indeterminado no protocolo de auditoria (NOT RUN — aguarda E07). A evidência histórica é cética: ganhos sintéticos/balanceados não transferiram entre pacientes (subset balanceado F1(F)=0,9379 vs inter-paciente 0,1627–0,214 na linhagem v14). A auditoria de geometria SMOTE exigida pelo doc mestre nunca foi executada sob o protocolo congelado. Evidência: `E04_qg5_gates/qg5_v2.4_report.json`.

**7. Os thresholds legados criam decisões frágeis?**
A auditoria da regra de decisão (E09A–D do doc mestre) não foi executada (NOT RUN); na campanha E06.5 a decisão é argmax do softmax puro (`method: ce_control`), portanto os thresholds legados S=0,5/V=0,5/F=0,8 não influenciaram nenhum número desta campanha. A evidência indica que o gargalo é de **ranqueamento**, não de decisão: AP_F=0,31 (H11), recall_F=0,099 no argmax, e no fold 5 os F reais são absorvidos por V (85–96/99). Nenhuma política de threshold corrige um ranqueador que coloca F abaixo de V/S em 99,44% dos casos. Evidência: `metrics.json` (campos `method`, `AP_F`), `fold5_F_confusion.csv`.

**8. Qual é a variabilidade entre seeds?**
Desvio-padrão de F1(F) sobre as 25 células: baseline 0,044; H6 0,095; H11 0,096; H12 0,094 — dominado pela variância **entre folds**. Dentro de um fold, o spread entre 5 seeds é pequeno (ex.: H6 fold 1: 0,271–0,303; fold 5: 0,000–0,020). O Δ H11−H6 = +0,0097 com IC95 [−0,000003; +0,020360] é da ordem da oscilação entre seeds ⇒ `GAIN_WITHIN_TRAINING_VARIANCE`, e H6 foi mantido. Evidência: `summary.json`, `selections/representation_selection.json`, `fold5_seed_comparison.csv`.

**9. CPU/GPU alteram decisões do modelo de forma material?**
Não medido (NOT RUN): o ambiente de auditoria é CPU-only (`physical_devices: [CPU:0]`), e a campanha rodou em modo determinístico (`deterministic_requested=true`). A reprodutibilidade intra-CPU foi provada por célula (`save_reload_max_abs_delta=0,0`, `prediction_equivalence=true`, `deterministic=true` em todos os 100 runs). A comparação cruzada CPU/GPU (E11) permanece em aberto e sem evidência de impacto material. Evidência: `manifests/preflight.json` (runtime), amostras de `metrics.json`.

**10. O target inter-paciente F1(F)≥0,50 foi realmente atingido?**
**Não.** Melhor média: 0,1470 (H11); candidato selecionado H6: 0,1373. Também não foi atingido o gate QG5' Estágio 2 F1(F)≥0,15. No ciclo anterior, o melhor resultado (0,4654±0,1018, reamostragem por paciente, sem gates formais) igualmente ficou abaixo de 0,50. Evidência: `summary.json` (`publication_target_F1_F: 0.5`), `E07_label_audit/baseline_resampled/baseline_enhanced_metrics.json`.

**11. O candidato v2.4 está pronto para publicação ou permanece research-only?**
**Research-only:** `RESEARCH_CANDIDATE_NOT_PUBLICATION_READY`. H6 é a representação de referência selecionada para continuar a pesquisa (E07), não um modelo publicável. Nenhum artefato v2.4 foi promovido a `models/`; v2.3 segue como produção. Detalhamento gate a gate: `docs/stage2_v2.4_publication_readiness.md`.

## Anexo B — Classificação das causas A–I (REGRA FINAL DE ENGENHARIA)

| Causa | Veredito | Justificativa quantitativa |
| --- | --- | --- |
| **A. Concentração da classe F por paciente/registro** | **DOMINANTE** | 70,31% de F em 208/213; mediana 3 F/grupo; 45/197 grupos com F; fold 5 (teste 100% fora de 208/213) colapsa todos os candidatos (F1(F) ≤ 0,015; 99,44% margens negativas); baseline zera 11/25 runs. O desempenho aparente em folds que contêm 208/213 (até 0,30) não se transfere. |
| **B. Representação insuficiente das 16 features** | **SECUNDÁRIA COMPROVADA (teto)** | Corrigir a representação deu +0,095…+0,105 F1(F) pareado (IC95 > 0) e 6–7,7× fora de 208/213 — logo a representação era de fato um limitante; mas o teto absoluto 0,147 mostra que representação tabular sozinha não fecha a lacuna. Status: `REPRESENTATION_SIGNAL_CONFIRMED / TARGET_NOT_MET`. |
| C. Viés do classificador long-tail | NÃO DOMINANTE | No ciclo anterior, focal+class-weight (0,453±0,082) não superou reamostragem isolada (0,465±0,102); trocar a loss não moveu o platô. |
| D. Sampling inadequado | **PENDENTE (única alavanca não auditada)** | Historicamente a maior alavanca (+0,25 no protocolo anterior), mas nunca medida sob o protocolo de auditoria E06.5 nem sob gates formais ⇒ objeto da E07 autorizada. Sampling não altera a geometria que produz 99,44% de margens negativas no fold 5. |
| E. Regra de decisão/threshold frágil | NÃO DOMINANTE | Campanha usa argmax puro; gargalo é ranqueamento (AP_F=0,31), não threshold. Auditoria E09A–D: NOT RUN. |
| F. Má calibração | NÃO DOMINANTE | O déficit é de discriminação (recall_F=0,099); calibração não cria separação F vs S/V. Métricas de calibração no protocolo de auditoria: NOT RUN. |
| G. Instabilidade de otimização | CONDICIONAL/SECUNDÁRIA | Variância entre folds domina a entre seeds; no fold 5 o audit classificou `OPTIMIZATION_COLLAPSE` (otimização converge para "não prever F" no regime sem 208/213). Δ H11−H6 dentro da variância entre seeds. |
| H. Insuficiência de modelagem morfológica | NÃO CONFIRMADA | H1 (morfologia QRS direta) e H4 (QRS amostrado) rejeitadas sem ganho material fora de 208/213; templates de classe (H5) idem. Morfologia tabular não foi a alavanca; a hipótese de morfologia bruta (E12 híbrido) não foi testada (condição de entrada discutida em `docs/stage2_v2.4_publication_readiness.md`). |
| **I. Combinação comprovada de causas** | **VEREDITO FINAL: combinação com A dominante** | A (dados) domina; B contribui com teto comprovado; G manifesta-se condicionalmente no fold mais duro; D pendente. Nenhuma alavanca de modelagem isolada (C, E, F, H) demonstrou efeito dominante. |

## Anexo C — Linha do tempo completa com checkpoints

| Data (UTC/-03) | Etapa | Checkpoint | Resultado-chave |
| --- | --- | --- | --- |
| 2026-07-11 | E00 snapshot forense | PASS | Baseline v14 congelado; reprodução Δ=0,0; hashes dos artefatos v2.3 registrados (`E00_baseline_snapshot/`) |
| 2026-07-11 | E01 distribuição F | PASS | 1.044 F (1,89%); 70,31% em 208/213; 45 grupos com F (`f_concentration_report.json`) |
| 2026-07-11 | E02 manifests imutáveis | PASS | Dataset/feature manifests + validador com falha explícita |
| 2026-07-11 | E03 protocolo de split | PASS | StratifiedGroupKFold selecionado com justificativa; zero overlap |
| 2026-07-11 | E04 redesenho QG5 | PASS | 5 sub-gates criados; v14: PATIENTWISE FAIL (0,1627<0,50) → `RESEARCH_CANDIDATE_NOT_PUBLICATION_READY` |
| 2026-07-11 | E05 separabilidade | PASS | RR-dominadas, MI<0,03; LGO sem 208 → 0,02, sem 213 → 0,40 |
| 2026-07-11 | E06 features p/ F (33 dim) | PASS_HYPOTHESIS_REJECTED | Features enhanced não melhoraram F1(F) |
| 2026-07-11 | E07 rótulos/reamostragem | PASS | Reescrita não justificada; reamostragem por paciente → F1(F)=0,4654±0,1018 (sem gates formais) |
| 2026-07-11 | E08 MLP+focal+class-weight | PASS | F1(F)=0,453±0,082; target 0,50 não atingido |
| 2026-07-11 | E09 publication guard | PASS | `DO_NOT_PUBLISH_v2.4`; nada publicado em `models/`; relatório do ciclo arquivado |
| 2026-07-14 | E06R triagem H1–H12 | PASS_HYPOTHESIS_REJECTED (12/12) | Ganho material fora de 208/213 só em H6/H8/H10/H11/H12; H7 reprovado no smoke; head `886f1b3` (`E06_reopened/`) |
| 2026-07-15 → 2026-07-26 | E06.5 smokes v1–v8 | iterações de pipeline | Gate canônico de 4 células converge para `E06_5_SMOKE_PASS` (v8, 2026-07-26T00:21Z) |
| 2026-07-16 | Preflight E06.5 (pino) | — | 4 paths de dados pinados por SHA-256 no HEAD `886f1b3` |
| **2026-07-18 12:56 (-03)** | **Incidente:** geração v3.1 sobrescreve os 4 paths pinados | (latente) | Origem do mismatch; commit `ed53dd8` |
| **2026-07-25 01:05 (-03)** | **Preflight BLOCKED** (exit_code=6, hash mismatch) | BLOCKED | `07841722…` ≠ pino `68fb0a8e…` em `stage2_multiclass.npz` |
| **2026-07-25** | **Recuperação do dataset** (`E065-RECOVERY-20260725`) | RECOVERY_VALIDATED_PENDING_PREFLIGHT → validado | Restauração byte-exata (backup 2026-07-11) + regeneração determinística no worktree `886f1b3`; quarentena dos bytes v3.1; **3 rebinds (audit-v1/v2/v3) com métricas idênticas** |
| 2026-07-26T00:20Z | Preflight final | PREFLIGHT_PASS | lint+pyright+pytest exit 0; zero leakage; CPU determinístico (`preflight.json`) |
| 2026-07-26T00:45Z | Auditoria fold 5 | PASS / `OPTIMIZATION_COLLAPSE` | 99 F/11 pacientes fora de 208/213; 99,44% margens negativas |
| 2026-07-26T00:45Z | Seleção de representação | — | **H6** selecionado (Δ H11−H6 dentro da variância entre seeds) |
| 2026-07-26T00:46Z | Agregação 100 runs | — | `summary.json` (baseline 0,0423; H6 0,1373; H11 0,1470; H12 0,1449) |
| 2026-07-26T00:46Z | Verify final | **E06_5_PASS_REPRESENTATION_SELECTED** | 8/8 checks; release E06.5 válida ⇒ E07 desbloqueada |

## Anexo D — Fontes primárias

- Doc mestre: `docs/STAGE 2 RESEARCH BRANCH v2.4.md`
- Ciclo E00–E09: `docs/stage2_v2.4_research_report.md`; manifests `experiments/stage2_v2.4_research/E0*/E*_manifest.json`
- Gates E04: `experiments/stage2_v2.4_research/E04_qg5_gates/qg5_v2.4_report.json`
- Concentração F: `experiments/stage2_v2.4_research/E01_patient_distribution/f_concentration_report.json`
- Reamostragem ciclo anterior: `experiments/stage2_v2.4_research/E07_label_audit/baseline_resampled/baseline_enhanced_metrics.json`
- Triagem E06R: `experiments/stage2_v2.4_research/E06_reopened/*/E06R_*_manifest.json`
- Campanha E06.5: `experiments/stage2_v2.4_research/E06_5/e065-audit-v3/` (100 células DONE + `summary.json`)
- Fold 5: `experiments/stage2_v2.4_research/fold_audits/e065-audit-v3/fold5_report.json` (+ CSVs/parquets)
- Seleção: `experiments/stage2_v2.4_research/selections/representation_selection.json`
- Verify: `experiments/stage2_v2.4_research/reports/e065_verify.json`
- Preflight/smoke: `experiments/stage2_v2.4_research/manifests/preflight.json`, `manifests/e065_smoke_gate.json`
- Recuperação do dataset: `experiments/stage2_v2.4_research/E065_recovery_20260725.json`
- Gates científicos: `docs/policies/authenticated_research_decision_v1.md` §3.2; `AGENTS.md` (QG5')
- Documentos do orquestrador (não sobrescritos): `docs/stage2_research_cli.md`, `docs/stage2_e065_robustness_report.md`, `docs/stage2_fold5_root_cause.md`
