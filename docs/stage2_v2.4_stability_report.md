# Relatório de Estabilidade Multi-Seed — E06.5 (audit) — Stage 2 v2.4, classe F

- **Experimento:** `E06.5` — estabilidade multi-seed e instabilidade de predição (etapa correspondente a E10 do doc mestre `docs/STAGE 2 RESEARCH BRANCH v2.4.md`)
- **Campanha analisada:** `experiments/stage2_v2.4_research/E06_5/e065-audit-v3/` — 100 células `DONE` (4 candidatos × 5 folds × 5 seeds)
- **Candidatos:** `baseline` (16 features, controle CE) e `H6`/`H11`/`H12` (representações com template features; 8/16/24 templates de fusão, conforme `template_states` do fold audit)
- **Protocolo comum (100/100 células):** `minimal_mlp_128`, loss `sparse_categorical_crossentropy`, sampling `natural`, decisão `raw_softmax_argmax`, scaler `outer_train_standard_scaler`, imputação `outer_train_median`, `deterministic=true`, `device=cpu:CPU`, `outer_test_used_for_selection=false`
- **Seeds fixas e versionadas:** `[17, 29, 43, 71, 101]`; `split_random_state=42`
- **Hashes da campanha (idênticos nas 100 células):**
  - dataset manifest: `168224d0198233f4619f356745366dd2775c23525109bba8d6ae029685ebe6f5`
  - split manifest: `d02d284c532da7eed8367c74b89539d79a0a580ddebe11d1730fde88f79f1593`
  - preflight: `4248e7dcdd9be6bbbf54698ea36c88f439474a03563ce73a98b09fa827972aa8`
  - runtime identity: `c936105e95af0c51b1ac60a9c8ead03d1fdfe5c1c96dac7ca238c5ff7f3d39ff`
- **Artefatos de decisão relacionados:** `experiments/stage2_v2.4_research/selections/representation_selection.json`, `experiments/stage2_v2.4_research/fold_audits/e065-audit-v3/fold5_report.json`
- **Geração:** 2026-07-26 (UTC). Análise executada com scripts temporários fora do repositório (`/tmp/e065_metrics_agg.py`, `/tmp/e065_pii.py`), sem modificação de `src/`, `tests/`, `config/` ou `models/`.
- **Nota metodológica:** o **PII_F (Prediction Instability Index)** usado nas seções 6–8 é uma **métrica experimental interna deste projeto** (definida na etapa E10 do doc mestre), não uma métrica clínica nem de publicação.

---

## 1. ETAPA EXECUTADA

```text
E06.5 — Estabilidade multi-seed e instabilidade de predição da research branch v2.4 (Stage 2, classe F).
Leitura e análise das 100 células da campanha e065-audit-v3 (baseline/H6/H11/H12 × folds 1–5 × seeds
17/29/43/71/101), com verificação bit-a-bit das cadeias e065-audit-v1 e e065-audit-v2 contra v3.
Nenhum treinamento foi executado nesta etapa; trata-se de etapa analítica sobre campanha congelada.
```

## 2. HIPÓTESE

Hipótese falsificável composta:

- **H-A:** o ganho médio de F1(F) dos candidatos de representação (H6/H11/H12) sobre o baseline é **maior que a variabilidade entre seeds** do treinamento (ganho robusto, não artefato de seed).
- **H-B:** o ganho incremental **entre** candidatos de representação (H11 vs H6, H12 vs H6) **não** excede a variabilidade entre seeds — ou seja, aumentar o número de templates de fusão além de H6 não produz ganho distinguível do ruído de treinamento.
- **H-C:** a variância total de F1(F) na campanha é dominada pela **composição dos folds** (quais pacientes/grupos F caem no outer test) e não pelo ruído entre seeds.

## 3. ALTERAÇÕES IMPLEMENTADAS

Nenhuma alteração de código, configuração, teste ou modelo. Etapa exclusivamente analítica:

| Arquivo | Propósito |
|---|---|
| `/tmp/e065_metrics_agg.py` (temporário, não versionado) | Agregação dos 100 `metrics.json`; decomposição de variância; deltas pareados; verificação bit-a-bit v1≡v2≡v3 |
| `/tmp/e065_pii.py` (temporário, não versionado) | Cálculo do PII_F e das taxas de discordância sobre os 100 `predictions.parquet` |
| `docs/stage2_v2.4_stability_report.md` (este arquivo, novo) | Relatório da etapa |

Artefatos lidos (não modificados): 100 × `metrics.json`, 100 × `predictions.parquet`, 100 × `run_manifest.json`, 3 × `summary.json` (v1/v2/v3), `selections/representation_selection.json`, `fold_audits/e065-audit-v3/*`. Nenhum dos relatórios existentes (`stage2_v2.4_research_report.md`, `stage2_e065_robustness_report.md`, `stage2_fold5_root_cause.md`) foi alterado.

## 4. VARIÁVEL CONCEITUAL ALTERADA

Nenhuma variável foi alterada nesta etapa. A variável conceitual **sob investigação** é:

```text
training stability (estabilidade do treinamento frente a seeds)
```

sobre a variável já fixada da campanha E06.5: `representation` (baseline vs H6/H11/H12).

## 5. CHECAGEM PÓS-OPERATÓRIA

```text
[PASS] mesmas seeds entre finalistas — [17,29,43,71,101] presentes nas 100/100 células
[PASS] mesmos folds — folds 1–5 completos para os 4 candidatos (20 células por candidato)
[PASS] mesmo manifest — dataset/split/preflight/runtime hashes com exatamente 1 valor distinto nas 100 células
[PASS] mesmas políticas de avaliação — architecture/loss/sampling/decision/deterministic/device idênticos (100/100);
       outer_test_used_for_selection=false (100/100)
[PASS] ausência de runs faltantes — 100/100 células com marcador DONE e status PASS
[PASS] nenhum run descartado por métrica ruim — n=25 runs por candidato em todos os agregados
[PASS] integridade das chaves de predição — (record_id, beat_idx) sem duplicatas; conjunto, ordem e y_true
       idênticos entre as 5 seeds de cada fold×candidato (verificado por assert nos 20 grupos)
[PASS] NaN/Inf — nenhum valor não-finito nos campos métricos lidos das 100 células
[PASS] split integrity / no group overlap — garantido a montante pelo split manifest congelado
       (d02d284c…) e auditado em E03/fold_audits; nada reaberto nesta etapa
[PASS] artifacts isolated — nenhum artefato v2.3, modelo, scaler ou doc existente modificado;
       único arquivo criado: este relatório
[NOT RUN] testes unitários / flake8 / mypy — etapa analítica sem alteração de código-fonte;
       scripts de análise temporários em /tmp, fora da árvore do repositório
```

## 6. RESULTADOS

### 6.1 Estatística global multi-seed (25 runs por candidato; std populacional, idêntica a `summary.json`)

| Candidato | F1(F) mean±std | min / max | F1-macro | F1(S) | F1(V) | precision_F | recall_F | AP_F | runs F1(F)=0 |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.0423 ± 0.0442 | 0.0000 / 0.1206 | 0.4445 ± 0.0625 | 0.4893 ± 0.1498 | 0.8018 ± 0.0344 | 0.1884 ± 0.2539 | 0.0285 ± 0.0289 | 0.1782 ± 0.1838 | **11/25** (fold2: 1, fold4: 5, fold5: 5) |
| H6 | 0.1373 ± 0.0947 | 0.0000 / 0.3034 | 0.4794 ± 0.0760 | 0.5009 ± 0.1321 | 0.7998 ± 0.0384 | 0.4158 ± 0.2647 | 0.0910 ± 0.0640 | 0.3133 ± 0.1922 | **2/25** (fold5: 2) |
| H11 | 0.1470 ± 0.0960 | 0.0000 / 0.3017 | 0.4824 ± 0.0720 | 0.4999 ± 0.1304 | 0.8002 ± 0.0368 | 0.4294 ± 0.2597 | 0.0994 ± 0.0671 | 0.3125 ± 0.1958 | **2/25** (fold5: 2) |
| H12 | 0.1449 ± 0.0944 | 0.0000 / 0.3162 | 0.4823 ± 0.0742 | 0.5016 ± 0.1307 | 0.8005 ± 0.0368 | 0.4291 ± 0.2675 | 0.0971 ± 0.0657 | 0.3121 ± 0.1939 | **2/25** (fold5: 2) |

Leitura honesta (regras de interpretação do doc mestre):

- O F1(F) médio subiu de 0.042 (baseline) para 0.137–0.147 (H6/H11/H12); 4 de 5 folds melhoraram em todos os candidatos H (exceção: fold 2, marginalmente abaixo do baseline — 0.0887–0.0939 vs 0.0943); o pior fold é o 5 para todos os candidatos H (no baseline, folds 4 e 5 empatam em zero — ver seção 7).
- **Resultado negativo não suavizado:** mesmo o melhor candidato fica em F1(F) ≈ 0.15 médio — muito abaixo do target de pesquisa/publicação `mean_F1_F ≥ 0.50` (QG5_PATIENTWISE falharia). O recall médio de F é ≤ 0.10: cerca de 90% dos batimentos F verdadeiros continuam não detectados no regime inter-paciente.
- F1(S) e F1(V) praticamente inalterados vs baseline (Δ F1(S) ≈ +0.011 a +0.012; Δ F1(V) ≈ −0.002 a +0.000): o ganho em F não foi comprado com regressão material das outras classes.
- Os 11 runs com F1(F)=0 do baseline concentram-se nos folds 4 e 5 (5/5 em cada) mais 1 run no fold 2; nos candidatos H restam apenas 2 runs zero, sempre no fold 5.

### 6.2 Deltas pareados por célula (mesmo fold + mesma seed, n=25)

CI95 oficial = bootstrap registrado em `selections/representation_selection.json`; CI95 recomp. = t-Student recomputado nesta análise (n=25, confirmação independente).

| Comparação | Δ médio | CI95 oficial | CI95 recomp. | win / tie / loss |
|---|---|---|---|---|
| H6 − baseline | +0.0950 | [+0.0567, +0.1338] | [+0.0530, +0.1370] | 0.80 / 0.08 / 0.12 |
| H11 − baseline | +0.1047 | [+0.0641, +0.1460] | [+0.0607, +0.1487] | 0.80 / 0.08 / 0.12 |
| H12 − baseline | +0.1026 | [+0.0641, +0.1414] | [+0.0607, +0.1445] | 0.80 / 0.08 / 0.12 |
| H11 − H6 | +0.0097 | **[−2.6e-06, +0.0204]** | [−0.0012, +0.0206] | 0.56 / 0.12 / 0.32 |
| H12 − H6 | +0.0076 | (não registrado) | [−0.0005, +0.0158] | 0.56 / 0.04 / 0.40 |
| H11 − H12 | +0.0021 | (não registrado) | [−0.0062, +0.0104] | 0.52 / 0.08 / 0.40 |

### 6.3 Critério numérico de classificação do ganho (documentado)

Ruído de treinamento de referência: **σ_seed** = desvio-padrão pooled entre seeds dentro de fold (seção 6.4): baseline 0.0242, H6 0.0199, H11 0.0206, H12 0.0213.

Um ganho pareado é classificado **ROBUST_GAIN** somente quando **todas** as condições valem:

1. CI95 do Δ médio pareado **exclui 0** (bootstrap oficial; confirmado pelo t-Student recomputado);
2. |Δ médio| > 2 × σ_seed do candidato (ganho maior que o dobro do ruído de treinamento);
3. win_fraction ≥ 0.60 (maioria consistente das 25 células pareadas).

Caso contrário: **GAIN_WITHIN_TRAINING_VARIANCE**.

| Comparação | Δ/σ_seed | CI95 exclui 0? | win | **Classificação** |
|---|---|---|---|---|
| H6 vs baseline | 4.8× | sim | 0.80 | **ROBUST_GAIN** |
| H11 vs baseline | 5.1× | sim | 0.80 | **ROBUST_GAIN** |
| H12 vs baseline | 4.8× | sim | 0.80 | **ROBUST_GAIN** |
| H11 vs H6 | 0.47× | não (limite inferior −2.6e-06 ≈ 0) | 0.56 | **GAIN_WITHIN_TRAINING_VARIANCE** |
| H12 vs H6 | 0.36× | não | 0.56 | **GAIN_WITHIN_TRAINING_VARIANCE** |
| H11 vs H12 | 0.10× | não | 0.52 | **GAIN_WITHIN_TRAINING_VARIANCE** |

Esta classificação confirma a decisão oficial de `representation_selection.json` (regra: *H11 não substitui H6 sem ganho > variabilidade entre seeds*): **H6 permanece a representação selecionada**. As diferenças médias entre H6/H11/H12 (≤ 0.010) são da mesma ordem — ou menores — que a oscilação multi-seed de uma única célula fold×candidato (σ_seed ≈ 0.02), e portanto não justificam promover H11/H12.

### 6.4 Variância entre seeds vs variância entre folds — o que domina?

| Candidato | std total F1(F) | σ entre seeds (pooled, dentro de fold) | σ entre folds (médias por fold) | η² fold (ANOVA) | razão fold/seed |
|---|---|---|---|---|---|
| baseline | 0.0442 | 0.0242 | 0.0370 | 0.70 | 1.53 |
| H6 | 0.0947 | 0.0199 | 0.0926 | 0.96 | 4.66 |
| H11 | 0.0960 | 0.0206 | 0.0938 | 0.95 | 4.54 |
| H12 | 0.0944 | 0.0213 | 0.0919 | 0.95 | 4.32 |

**A variância entre folds domina decisivamente.** Para os candidatos H, ~95% da variância total de F1(F) é explicada pelo fator fold (composição do outer test: quais pacientes/grupos F caem no teste — ver seção 8), e apenas ~5% pelo ruído agregado de seed+interação. O ruído puro entre seeds dentro de um mesmo fold é pequeno (σ ≈ 0.02; std por fold de H11: 0.0121 / 0.0216 / 0.0156 / 0.0327 / 0.0143 nos folds 1–5), mas **não desprezível para comparações finas**: é exatamente por isso que Δ(H11−H6) = +0.0097 não é distinguível de zero.

Corolário: seeds fixas e versionadas bastam para comparar candidatos frente ao baseline; qualquer seleção entre candidatos H exigiria mais seeds ou mais folds para resolução fina — não sendo esse o caso, a regra GAIN_WITHIN_TRAINING_VARIANCE é a decisão correta.

### 6.5 PII_F — Prediction Instability Index (métrica experimental interna do projeto)

Definição (etapa E10 do doc mestre): para cada batimento presente nas avaliações comparáveis, `PII_F = max(P_F) − min(P_F)` entre as 5 seeds; agregado mean/median/p90/p95/max. Alinhamento por `(record_id, beat_idx)`; cada exemplo aparece exatamente 1 vez no outer test de um fold (55.161 exemplos no total; 1.044 verdadeiros F). Taxa de discordância de classe: fração de exemplos com qualquer divergência de `y_pred` entre as 5 seeds (any-seed) e média pareada dos 10 pares de seeds.

**Global (todos os exemplos):**

| Candidato | PII_F mean | median | p90 | p95 | max | disc. any-seed | disc. pareada média |
|---|---|---|---|---|---|---|---|
| baseline | 0.0144 | 0.0032 | 0.0328 | 0.0670 | 0.6915 | 0.1246 | 0.0600 |
| H6 | 0.0092 | 0.0003 | 0.0188 | 0.0504 | 0.5517 | 0.1255 | 0.0608 |
| H11 | 0.0092 | 0.0003 | 0.0184 | 0.0489 | 0.6045 | 0.1221 | 0.0594 |
| H12 | 0.0091 | 0.0002 | 0.0179 | 0.0500 | 0.5438 | 0.1221 | 0.0594 |

**Somente verdadeiros F (n=1.044):**

| Candidato | PII_F mean | p95 | disc. any-seed |
|---|---|---|---|
| baseline | 0.0677 | 0.2189 | 0.2720 |
| H6 | 0.1044 | 0.2618 | 0.2490 |
| H11 | 0.1026 | 0.2554 | 0.2835 |
| H12 | 0.1013 | 0.2599 | 0.2701 |

Instabilidade da classe F na decisão: exemplos preditos F por alguma seed mas não por todas = 0.32% (baseline) a 0.41–0.42% (H6/H11/H12) de todos os exemplos; exemplos preditos F por ao menos uma seed = 0.44% (baseline) a 0.64–0.67% (H*).

Interpretação: (i) globalmente os candidatos H são **mais estáveis** em P_F que o baseline (PII médio 0.009 vs 0.014) — o baseline concentra sua incerteza em poucos exemplos; (ii) nos verdadeiros F o quadro se inverte: os H* produzem P_F mais altas e mais responsivas, que oscilam mais em magnitude (PII ≈ 0.10 vs 0.068); (iii) a discordância de classe global (~12% any-seed) é dominada pela fronteira S↔V, não por F (apenas ~0.4% dos exemplos oscilam envolvendo F); (iv) a instabilidade de decisão em torno de F é pequena em frequência, mas quando ocorre é severa — o PII_F máximo chega a 0.54–0.69, e é ela que produz os runs F1(F)=0 vs F1(F)=0.02–0.04 do fold 5.

### 6.6 Reprodutibilidade determinística e ambiente

**v1 ≡ v2 ≡ v3 (bit-a-bit):** comparação dos 100 `metrics.json` de `e065-audit-v1` e `e065-audit-v2` contra `e065-audit-v3`, por célula e por campo (F1_F, macro_F1, F1_S, F1_V, precision_F, recall_F, AP_F e os três escopos 208/213/outside): **diferença absoluta máxima = 0.0** em ambas as comparações. Os agregados oficiais são idênticos nas três cadeias — p.ex. mean F1(F) H11 = `0.14703608392635772` em v1, v2 e v3 — apesar de as três cadeias terem preflights e smokes distintos. Condições registradas (100/100 células): `deterministic=true`, `device=cpu:CPU`, seeds de Keras/NumPy/TF/`PYTHONHASHSEED` = seed da célula, `split_random_state=42`, dataset/split manifest hashes fixos (cabeçalho).

**Ambiente (environment.json / run_manifest.json):** CPU-only; Python 3.12.3; TensorFlow 2.21.0; Keras 3.14.1; NumPy 2.4.6; Linux x86_64 (glibc 2.39); `git_head 6957bf6a…` (git_dirty=true na execução); `uv.lock` hash `8c0d70ad…`. Nenhuma GPU envolvida; a reprodutibilidade bit-a-bit é portanto válida **para este ambiente CPU determinístico**, não sendo garantida em GPU ou em outra versão de TF/NumPy.

## 7. ANÁLISE POR FOLD

Suporte F no outer test por fold (idêntico para os 4 candidatos): fold 1 = 379 (372 do registro 208 + 7 fora), fold 2 = 386 (362 do registro 213 + 24 fora), fold 3 = 86, fold 4 = 94, fold 5 = 99 — folds 3–5 contêm **somente** F fora de 208/213.

**F1(F) por fold × seed (valores brutos, sem seleção do melhor):**

| Candidato | Fold | seed 17 | seed 29 | seed 43 | seed 71 | seed 101 | média fold |
|---|---|---|---|---|---|---|---|
| baseline | 1 | 0.0388 | 0.0560 | 0.0242 | 0.0541 | 0.0972 | 0.0541 |
| baseline | 2 | 0.1193 | 0.1129 | 0.0000 | 0.1206 | 0.1185 | 0.0943 |
| baseline | 3 | 0.0690 | 0.0682 | 0.0625 | 0.0442 | 0.0727 | 0.0633 |
| baseline | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| baseline | 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| H6 | 1 | 0.2708 | 0.2797 | 0.3034 | 0.2976 | 0.2910 | 0.2885 |
| H6 | 2 | 0.0876 | 0.0782 | 0.1002 | 0.0688 | 0.1349 | 0.0939 |
| H6 | 3 | 0.0952 | 0.1504 | 0.1301 | 0.0960 | 0.0960 | 0.1135 |
| H6 | 4 | 0.2044 | 0.1752 | 0.1583 | 0.2137 | 0.1440 | 0.1791 |
| H6 | 5 | 0.0000 | 0.0200 | 0.0196 | 0.0000 | 0.0182 | 0.0116 |
| H11 | 1 | 0.2757 | 0.2788 | 0.2688 | 0.3017 | 0.2689 | 0.2788 |
| H11 | 2 | 0.0741 | 0.0732 | 0.0870 | 0.0786 | 0.1308 | 0.0887 |
| H11 | 3 | 0.1094 | 0.1324 | 0.1562 | 0.1212 | 0.1343 | 0.1307 |
| H11 | 4 | 0.2754 | 0.1752 | 0.2098 | 0.2326 | 0.2154 | 0.2217 |
| H11 | 5 | 0.0000 | 0.0381 | 0.0200 | 0.0000 | 0.0183 | 0.0153 |
| H12 | 1 | 0.2590 | 0.2727 | 0.2960 | 0.3162 | 0.2893 | 0.2866 |
| H12 | 2 | 0.0739 | 0.0874 | 0.0876 | 0.0971 | 0.1224 | 0.0937 |
| H12 | 3 | 0.1053 | 0.1679 | 0.1613 | 0.1077 | 0.1280 | 0.1340 |
| H12 | 4 | 0.2361 | 0.1515 | 0.1986 | 0.2031 | 0.1875 | 0.1954 |
| H12 | 5 | 0.0000 | 0.0000 | 0.0200 | 0.0190 | 0.0360 | 0.0150 |

**F1-macro médio por fold** (5 seeds): baseline 0.5527 / 0.4436 / 0.4296 / 0.3649 / 0.4317; H6 0.6243 / 0.4340 / 0.4835 / 0.4311 / 0.4239; H11 0.6179 / 0.4333 / 0.4909 / 0.4447 / 0.4252; H12 0.6218 / 0.4357 / 0.4919 / 0.4369 / 0.4255 (folds 1–5).

**PII_F médio por fold (todos os exemplos / verdadeiros F):**

| Candidato | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |
|---|---|---|---|---|---|
| baseline | 0.0147 / 0.1066 | 0.0201 / 0.0453 | 0.0125 / 0.0584 | 0.0073 / 0.0444 | 0.0169 / 0.0368 |
| H6 | 0.0105 / 0.1305 | 0.0082 / 0.0940 | 0.0070 / 0.0942 | 0.0074 / 0.1091 | 0.0131 / 0.0497 |
| H11 | 0.0106 / 0.1262 | 0.0082 / 0.0939 | 0.0070 / 0.0915 | 0.0074 / 0.1071 | 0.0127 / 0.0512 |
| H12 | 0.0107 / 0.1279 | 0.0073 / 0.0890 | 0.0068 / 0.0898 | 0.0073 / 0.1049 | 0.0132 / 0.0546 |

Observações por fold:

- **Fold 1** (teste contém 208): maior F1(F) absoluto (H* ≈ 0.26–0.32) e alta estabilidade entre seeds (σ = 0.0118 / 0.0121 / 0.0196 para H6/H11/H12 — os dois menores σ de H6 e H11 entre todos os folds, contra σ ≈ 0.027–0.033 no fold 4); é também o fold com maior suporte F (379). Não representa o regime geral.
- **Fold 2** (teste contém 213): H* ≈ 0.07–0.13; baseline parcialmente comparável (0–0.12); seed 101 sistematicamente melhor para todos os H — interação seed×fold visível.
- **Fold 4** (94 F, todos fora de 208/213): baseline colapsa em 5/5 seeds (F1=0), enquanto H* atingem 0.14–0.28 em todas as seeds — **a evidência mais limpa de ganho de representação fora dos grupos dominantes**.
- **Fold 5** (99 F de 11 pacientes, todos fora de 208/213): `OPTIMIZATION_COLLAPSE` (classificação de `fold_audits/e065-audit-v3/fold5_report.json`): max F1(F) > 0 e min = 0 entre seeds em todos os candidatos H; baseline 0/5. Fração de margens F negativas = 0.9944 (1.980 linhas F verdadeiras). É o pior fold para todos e a principal fonte dos runs zero restantes (2/25 por candidato H).
- Médias por seed (margem sobre folds), H11: 0.1469 / 0.1395 / 0.1484 / 0.1468 / 0.1536 — nenhuma seed é globalmente “boa” ou “má”; a hierarquia de dificuldade é fold-driven.

## 8. ANÁLISE DA CLASSE F

Escopos calculados **somente nos folds em que o registro consta no outer test** (208 → fold 1, n=5 seeds; 213 → fold 2, n=5 seeds; fora de 208/213 → todos os folds, n=25). Nota: as médias de escopo presentes em `summary.json` incluem zeros estruturais dos folds sem o registro (suporte 0) e não devem ser usadas para leitura por grupo.

**F1(F) por grupo:**

| Grupo | baseline | H6 | H11 | H12 |
|---|---|---|---|---|
| 208 (n=5 seeds, fold 1) | 0.0577 ± 0.0266 | 0.3066 ± 0.0162 | 0.2956 ± 0.0148 | 0.3035 ± 0.0247 |
| 213 (n=5 seeds, fold 2) | 0.1027 ± 0.0515 | 0.0962 ± 0.0228 | 0.0915 ± 0.0219 | 0.0938 ± 0.0178 |
| remaining F (n=25) | 0.0127 ± 0.0257 | 0.0800 ± 0.0678 | 0.0917 ± 0.0800 | 0.0972 ± 0.0683 |

**PII_F / discordância por grupo (métrica experimental interna):**

| Grupo (n exemplos) | Métrica | baseline | H6 | H11 | H12 |
|---|---|---|---|---|---|
| 208 (1.366) | PII médio / disc. any-seed | 0.0637 / 0.0578 | 0.0586 / 0.0520 | 0.0590 / 0.0512 | 0.0585 / 0.0571 |
| 213 (610) | PII médio / disc. any-seed | 0.0366 / 0.6082 | 0.0853 / 0.4279 | 0.0865 / 0.5377 | 0.0811 / 0.4525 |
| remaining F (310) | PII médio / disc. any-seed | 0.0505 / 0.0419 | 0.0816 / 0.1419 | 0.0807 / 0.1516 | 0.0820 / 0.1677 |

Leitura:

- **208:** ganho robusto e estável (H* ≈ 0.30 vs 0.06; σ de seed baixo). Mesmo com 208 **no teste** do fold 1 (ausente do treino desse fold), os templates generalizam de outros pacientes para a morfologia de 208.
- **213 (resultado negativo, não suavizado):** o baseline é comparável ou ligeiramente **superior** aos candidatos H (0.1027 vs 0.0915–0.0962), e a discordância de classe entre seeds é altíssima para todos (43–61% any-seed): 213 permanece um paciente essencialmente não generalizado e instável, para qualquer representação testada. O ganho agregado dos H* **não** vem de 213.
- **remaining F (fora de 208/213):** ganho de ~6–7× sobre o baseline (0.080–0.097 vs 0.013), consistente com o fold 4, mas em magnitude absoluta ainda fraco — e é exatamente esse grupo que colapsa no fold 5.
- A concentração estrutural documentada em E01 se manifesta aqui: o desempenho médio de F1(F) continua fortemente modulado pela presença de 208/213 no teste; fora deles, F1(F) ≤ 0.10 mesmo com as novas representações.

## 9. REGRESSÕES

- **Introduzidas por esta etapa:** nenhuma — etapa analítica, sem modificação de código/artefatos.
- **Vs baseline (25 runs pareadas):** nenhuma regressão material: Δ F1(S) = +0.011 a +0.012 (a favor dos H*); Δ F1(V) = −0.002 a +0.000 (dentro do ruído); Δ F1-macro = +0.035 a +0.038 (a favor dos H*).
- **Observação negativa registrada (não é regressão de pipeline, é resultado do experimento):** no registro 213, os candidatos H ficam ~0.006–0.011 **abaixo** do baseline em F1(F) médio; e o fold 5 mantém colapso parcial (2/25 runs com F1(F)=0 por candidato H). Ambos os achados ficam preservados como evidência, sem descarte de runs.

## 10. HIPÓTESE

```text
H-A (ganho H* > baseline excede a variabilidade entre seeds): SUPPORTED
H-B (ganho incremental entre H6/H11/H12 não excede a variância de treinamento): SUPPORTED
H-C (variância de F1(F) dominada pela composição dos folds, não por seeds): SUPPORTED

Classificação global da hipótese: SUPPORTED
```

Evidências: seção 6.3 (ROBUST_GAIN ×3 vs baseline; GAIN_WITHIN_TRAINING_VARIANCE ×3 entre H*); seção 6.4 (η² fold 0.95, razão fold/seed ≈ 4.5 para H*); seções 7–8 (fold 4 como evidência positiva fora de 208/213; fold 5 e registro 213 como limites do ganho).

## 11. CHECKPOINT

```text
PASS
```

Justificativa: integridade total da campanha (100/100 DONE, nenhum run descartado, manifests/seeds/folds idênticos, outer test nunca usado para seleção), reprodutibilidade bit-a-bit v1≡v2≡v3 verificada por célula, critério numérico de classificação documentado e aplicado de forma idêntica aos quatro candidatos, resultados negativos preservados (fold 5, registro 213, F1(F) médio ainda ≈ 0.15, recall_F ≤ 0.10). A seleção oficial de H6 (`representation_selection.json`) é confirmada por análise independente.

## 12. PRÓXIMA ETAPA AUTORIZADA

```text
E07 — AUDITORIA DE SAMPLING (doc mestre), executada sobre a representação selecionada H6,
mantendo seeds [17,29,43,71,101], folds 1–5 e manifests congelados; fold 5 e o registro 213
devem ser reportados explicitamente em qualquer comparação de sampling.
```

---

### Anexo A — Rastreabilidade da análise

- Fontes de dados: `experiments/stage2_v2.4_research/E06_5/e065-audit-v{1,2,3}/<cand>/fold_N/seed_M/{metrics.json, predictions.parquet, run_manifest.json, DONE}`; `E06_5/e065-audit-v{1,2,3}/summary.json`; `selections/representation_selection.json`; `fold_audits/e065-audit-v3/fold5_report.json` e `fold5_seed_comparison.csv`.
- Scripts (temporários, não versionados): `/tmp/e065_metrics_agg.py` (agregados, variância, deltas pareados, bitwise check) e `/tmp/e065_pii.py` (PII_F e discordâncias). Resultados brutos: `/tmp/e065_metrics_results.json`, `/tmp/e065_pii_results.json`, `/tmp/e065_scopes_filtered.json`.
- Convenções: std populacional (ddof=0) nas tabelas globais, idêntica a `summary.json`; CI95 oficial = bootstrap de `representation_selection.json`; CI95 de confirmação = t-Student (n=25, t(0.975,24)=2.0639); mapeamento de classes das predições 0=S, 1=V, 2=F (consistente com as matrizes de confusão das células).
- Limitações conhecidas: PII_F calculado sobre probabilidades float32 serializadas; escopo 208 limitado ao fold 1 e escopo 213 ao fold 2 (n=5 seeds cada) — estatísticas desses grupos têm resolução de seed, não de fold; a equivalência bit-a-bit é válida para o ambiente CPU descrito na seção 6.6 e não é garantida em outros ambientes.
