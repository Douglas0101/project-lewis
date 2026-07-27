# Publication Readiness — Research Branch Stage 2 v2.4 (classe F)

- **Data:** 2026-07-26
- **Escopo:** veredito de publicação da research branch v2.4 do Stage 2 (S/V/F), ao fechamento da campanha E06.5 (`e065-audit-v3`)
- **Documento irmão:** `docs/stage2_v2.4_root_cause_report.md` (causa-raiz, 11 perguntas, causas A–I, linha do tempo)
- **Gates de referência:** `AGENTS.md` (QG5' Estágio 2), `docs/policies/authenticated_research_decision_v1.md` §3.2, `docs/STAGE 2 RESEARCH BRANCH v2.4.md` (E04/E13)

## 1. Veredito

```text
RESEARCH_CANDIDATE_NOT_PUBLICATION_READY
```

Sem afrouxar gates e sem reduzir o target de publicação (`mean F1(F) >= 0,50`, inalterado desde o doc mestre e reafirmado na policy §3.2). O candidato selecionado **H6** é uma *representação de referência para pesquisa continuada*, não um modelo publicável. Nenhum artefato v2.4 foi promovido a `models/`; **v2.3 permanece como linha de produção**.

## 2. Tabela de gates

### 2.1 QG5' Estágio 2 por candidato (protocolo de auditoria E06.5)

Thresholds (policy §3.2 / `AGENTS.md`): F1(S) ≥ 0,55; F1(V) ≥ 0,70; F1(F) ≥ 0,15; F1-macro ≥ 0,45.
Valores = médias sobre 25 runs (5 folds × 5 seeds), computadas dos `metrics.json` de `experiments/stage2_v2.4_research/E06_5/e065-audit-v3/<candidato>/` e idênticas ao agregado oficial `summary.json`. Protocolo: distribuição natural, CE, argmax, CPU determinístico.

| Candidato | F1(S) (≥0,55) | F1(V) (≥0,70) | F1(F) (≥0,15) | F1-macro (≥0,45) | Resultado |
| --- | ---: | ---: | ---: | ---: | --- |
| baseline | 0,489 ✗ | 0,802 ✓ | 0,042 ✗ | 0,445 ✗ | **FAIL** (3/4 gates) |
| H6 (selecionado) | 0,501 ✗ | 0,800 ✓ | 0,137 ✗ | 0,479 ✓ | **FAIL** (2/4 gates) |
| H11 | 0,500 ✗ | 0,800 ✓ | 0,147 ✗ | 0,482 ✓ | **FAIL** (2/4 gates) |
| H12 | 0,502 ✗ | 0,800 ✓ | 0,145 ✗ | 0,482 ✓ | **FAIL** (2/4 gates) |

Nenhum candidato passa no QG5' Estágio 2. Observações honestas:

- F1(F) do melhor candidato (H11, 0,1470) fica abaixo do gate 0,15 por 0,003 — registrado como falha, sem arredondamento.
- F1(S) também falha em todos os candidatos (0,489–0,502 < 0,55); a branch investigou F, mas o gate de S permanece igualmente não atendido no protocolo de auditoria com distribuição natural.
- O flag `publication_eligible=true` presente em cada `metrics.json` refere-se à elegibilidade de **integridade da célula** (run válido para agregação), não à prontidão de publicação do modelo.

### 2.2 Target de publicação

| Gate | Threshold | Melhor resultado | Status |
| --- | ---: | ---: | --- |
| Target final de publicação (doc mestre E13; policy §3.2) | mean F1(F) ≥ 0,50 | 0,1470 (H11); 0,1373 (H6 selecionado) | **FAIL** — lacuna de 0,353 (o melhor resultado é 29% do target) |
| Referência histórica ciclo E00–E09 (sem gates formais) | mean F1(F) ≥ 0,50 | 0,4654 ± 0,1018 (reamostragem por paciente, E07 do ciclo anterior); 0,453 ± 0,082 (focal+class-weight, E08 do ciclo anterior) | **FAIL** (também abaixo) |

### 2.3 Sub-gates E04 (QG5_SMOKE_BALANCED / PATIENTWISE / STABILITY / CALIBRATION / REPRODUCIBILITY)

Status para os **candidatos E06.5** (coluna "E06.5") e, como referência, para o **v14 legado** medido em E04 (`experiments/stage2_v2.4_research/E04_qg5_gates/qg5_v2.4_report.json`, 2026-07-11):

| Sub-gate | Candidatos E06.5 | v14 legado (E04) | Observação |
| --- | --- | --- | --- |
| QG5_SMOKE_BALANCED | **NOT RUN** | PASS (diagnostic_only; F1(F)=0,9379 no subset balanceado) | O smoke canônico E06.5 (`e065-smoke-v8`, 4 células fold 1/seed 17) PASSOU, mas é gate de integridade de pipeline — não é o diagnóstico balanceado E04 e não autoriza publicação |
| QG5_PATIENTWISE | **FAIL** (medido: tabela 2.1; melhor F1(F)=0,147 < 0,15 e < 0,50) | FAIL (F1(F)=0,1627 < 0,50) | Gate real de generalização inter-paciente; reprova todos |
| QG5_STABILITY | **NOT RUN** (como gate) | PASS (recording-only) | Limites de estabilidade nunca foram definidos (E10 não executado); variabilidade registrada: std F1(F) ≈ 0,095; worst fold F1(F) = 0,000 (baseline) / 0,012–0,015 (candidatos, fold 5) |
| QG5_CALIBRATION | **NOT RUN** | PASS (recording-only; log_loss 0,745, Brier 0,153) | `metrics.json` da campanha não inclui log_loss/Brier/reliability; decisão = argmax sem calibrador |
| QG5_REPRODUCIBILITY | **PASS** | PASS | 100/100 células DONE; determinístico CPU; save/reload Δ = 0,0; hashes de dataset/split/features encadeados; `e065_verify.json` 8/8 checks; zero leakage (outer/scaler/template) |

Status composto E06.5: 1 FAIL (patientwise), 1 PASS (reproducibility), 3 NOT RUN — qualquer leitura de "publicável" está excluída já pelo PATIENTWISE.

## 3. O que falta objetivamente para publicação

1. **Lacuna de F1(F):** 0,147 → 0,50 (déficit absoluto 0,353; o melhor candidato atinge 29% do target). Mesmo o gate intermediário QG5' F1(F) ≥ 0,15 não é atendido (0,147).
2. **Recall de F:** recall_F médio = 0,099 (H11) — o sistema perde ~90% dos fusion beats reais inter-paciente; precision_F = 0,429 e AP_F = 0,312 indicam ranqueamento F insuficiente, não apenas threshold mal ajustado.
3. **Colapso no fold mais duro:** fold 5 (outer test com 99 F de 11 pacientes, todos fora de 208/213) produz F1(F) ≤ 0,015 em todos os candidatos e 99,44% de margens F negativas (`OPTIMIZATION_COLLAPSE`). Publicação exigirá demonstrar generalização exatamente nesse regime — hoje é o pior resultado, não o melhor.
4. **F1(S) abaixo do gate:** 0,500–0,502 vs 0,55 exigido (todos os candidatos, protocolo natural).
5. **Sub-gates não executados:** STABILITY (sem limites definidos), CALIBRATION (sem métricas no protocolo), SMOKE_BALANCED (não aplicado aos candidatos). Antes de qualquer conversa de publicação, os 5 sub-gates precisam estar RUN para o candidato final.
6. **Refit e artefatos finais (E13):** protocolo de refit documentado, `final_training_manifest.json`, validação pós-publicação em processo limpo — nada disso foi iniciado, corretamente, pois os gates não passaram.

## 4. Próxima etapa autorizada (exatamente uma)

```text
E07 — AUDITORIA DE SAMPLING (via e07-run)
```

**Por que E07 e por que agora:** a policy §3.2 condicionava E07/E08 a uma release válida de E06.5; essa release existe desde 2026-07-26T00:46Z (`e065_verify.json` = `E06_5_PASS_REPRESENTATION_SELECTED`). Logo E07 está **tecnicamente desbloqueada** e é a única variável conceitual seguinte na sequência do doc mestre (representation → sampling → long-tail loss → decisão/calibração). Escopo congelado proposto: representação **H6 fixa** (feature hash `50762aa1…`), CE, mesma matriz 5 folds × 5 seeds e mesmos manifests de E06.5; braços: `natural` (já medido em E06.5), `random oversampling`, `SMOTE` (com auditoria de geometria de sintéticos) e `patient-aware sampling`; seleção inner-only; classificação BENEFICIAL/NEUTRAL/UNSTABLE/HARMFUL por braço; gates formais aplicados.

**Sampling pode fechar a lacuna? Avaliação à luz das evidências:**

- A favor: no ciclo anterior, reamostragem por paciente foi a maior alavanca já observada — elevou o baseline de F1(F) ≈ 0,21 para 0,4654 ± 0,1018 (`E07_label_audit/baseline_resampled/baseline_enhanced_metrics.json`). E08 (focal+class-weight) não adicionou nada sobre ela (0,453 ± 0,082), indicando que a alavanca era o sampling, não a loss.
- Contra / limites: (a) esse resultado veio de protocolo **sem gates formais**, sem multi-seed e sem auditoria de fold duro — e mesmo assim ficou abaixo de 0,50; (b) o pior fold daquele protocolo já era o fold 5 (F1(F)=0,330, mínimo dos 5 folds); (c) sampling redistribui gradientes, mas não altera a geometria que produz 99,44% de margens F negativas no fold 5 — onde o treino já contém 945 beats F de 34 pacientes, ou seja, o gargalo ali é transferência inter-paciente, não contagem bruta de F; (d) para fechar a lacuna no protocolo de auditoria, sampling teria de produzir efeito ~3,6× maior que o teto de representação medido (0,137 → 0,50).
- **Expectativa honesta:** ganho parcial é plausível (a alavanca é real); fechamento da lacuna até 0,50 **não é suportado** pela evidência existente. E07 deve ser executada como teste falsificável com gates formais: se o melhor braço não superar F1(F) ≥ 0,50 no protocolo de auditoria, a conclusão "causa A dominante" torna-se definitiva e a branch deve escalar para decisão de escopo (dados adicionais de F, p. ex. fontes fora dos datasets atuais, ou arquitetura híbrida E12 — cuja condição de entrada do doc mestre, "mean F1(F) inter-paciente < 0,40 após E06–E10 com evidência de gargalo morfológico/representacional", estaria então formalmente atendida), e não para mais tuning incremental.

## 5. Integridade da produção

- **Nenhum artefato em `models/` foi tocado** por esta branch/campanha: `git status --porcelain -- models/` limpo ao fechamento; a última alteração em `models/` é o commit `886f1b3` (2026-07-16), restrito a `models/quantized/` (artefatos de quantização de firmware), anterior à campanha de auditoria.
- **v2.3 segue como produção**: modelos, scalers e thresholds v2.3 permanecem os artefatos oficiais; o incidente de sobrescrita de dados de 2026-07-18 foi revertido byte-exato (SHA-256 verificado) e os bytes v3.1 estão em quarentena (`data/features/quarantine_v31_working_20260718/`); detalhes em `experiments/stage2_v2.4_research/E065_recovery_20260725.json`.
- Artefatos v2.4 existem **somente** sob `experiments/stage2_v2.4_research/`, classificados como pesquisa.
- Declaração herdada do ciclo E00–E09 mantida: nenhum PII exposto; dados fisiológicos anonimizados (PhysioNet); LGPD preservada.

## 6. Fontes

- `experiments/stage2_v2.4_research/E06_5/e065-audit-v3/summary.json` e `metrics.json` das 100 células
- `experiments/stage2_v2.4_research/reports/e065_verify.json`; `manifests/preflight.json`; `manifests/e065_smoke_gate.json`
- `experiments/stage2_v2.4_research/fold_audits/e065-audit-v3/fold5_report.json` (+ `fold5_F_confusion.csv`, `fold5_seed_comparison.csv`)
- `experiments/stage2_v2.4_research/selections/representation_selection.json`
- `experiments/stage2_v2.4_research/E04_qg5_gates/qg5_v2.4_report.json`
- `experiments/stage2_v2.4_research/E07_label_audit/baseline_resampled/baseline_enhanced_metrics.json`
- `experiments/stage2_v2.4_research/E065_recovery_20260725.json`
- `docs/policies/authenticated_research_decision_v1.md` §3.2; `AGENTS.md` (QG5'); `docs/STAGE 2 RESEARCH BRANCH v2.4.md`
- `docs/stage2_v2.4_research_report.md` (histórico E00–E09, não sobrescrito)
