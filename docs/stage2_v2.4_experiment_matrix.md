# Stage 2 v2.4 — Matriz Experimental da Etapa E06.5 (audit) — Classe F (fusion beat AAMI)

- **Etapa:** E06.5 audit — matriz de auditoria multi-seed da representação E06
- **Campanha auditada:** `experiments/stage2_v2.4_research/E06_5/e065-audit-v3/` (100/100 células `DONE`)
- **Candidatos:** `baseline` (16 features), `H6` / `H11` / `H12` (base16 + causal_rr_h3 + class_templates_h5; bancos de 8/16/24 templates de fusão)
- **Grade:** 4 candidatos × 5 folds externos × 5 seeds (`17, 29, 43, 71, 101`)
- **Classe F:** batimento de fusão V+N (AAMI `F`), 1.044 batimentos (~1,9% de 55.161), ~70,3% concentrados nos records 208 (372 F) e 213 (362 F) — `experiments/stage2_v2.4_research/E01_patient_distribution/f_concentration_report.json`
- **Data do relatório:** 2026-07-26
- **Formato:** seções 1–12 obrigatórias conforme `docs/STAGE 2 RESEARCH BRANCH v2.4.md`
- **Documentos relacionados (não sobrescritos por este relatório):** `docs/stage2_v2.4_research_report.md` (linha de pesquisa anterior E00–E09), `docs/stage2_e065_robustness_report.md`, `docs/stage2_fold5_root_cause.md`

> **Convenções numéricas.** Tabelas arredondadas em 3 casas decimais; valores-chave também citados em precisão cheia quando necessário à rastreabilidade. `std` segue a convenção do `summary.json` da campanha (ddof=0); recomputação independente com pandas reproduziu todas as estatísticas do artefato com diferença < 1e-12. Nenhum resultado negativo foi omitido; nenhum fold foi descartado.

---

## 1. ETAPA EXECUTADA

```text
E06.5 — audit da representação E06 (context features v2.4) sob protocolo multi-seed
```

Auditoria integral da campanha `e065-audit-v3`: 100 células (`metrics.json`, `predictions.parquet`, `run_manifest.json`, `DONE`), agregação independente dos resultados, análise por fold e por escopo da classe F (record 208 / record 213 / remaining F groups), verificação de reprodutibilidade contra as cadeias `e065-audit-v1` e `e065-audit-v2`, e consolidação da decisão de seleção de representação registrada em `experiments/stage2_v2.4_research/selections/representation_selection.json`.

Este relatório **não retreina modelos**: é a etapa de consolidação/auditoria dos artefatos já produzidos pela campanha.

## 2. HIPÓTESE

> Sob protocolo auditável (5 folds externos × 5 seeds, CPU determinístico, seleção inner-only), a representação `base16 + causal_rr_h3 + class_templates_h5` aumenta o F1(F) inter-paciente médio em relação ao baseline de 16 features com ganho maior que a variância multi-seed, sem degradar materialmente F1(S), F1(V) ou F1-macro.

## 3. ALTERAÇÕES IMPLEMENTADAS

Nenhuma alteração em código-fonte, testes, configuração ou modelos publicados:

- `src/`, `tests/`, `config/`, `models/`: **inalterados** (verificação read-only).
- Criado somente este documento: `docs/stage2_v2.4_experiment_matrix.md`.
- Scripts temporários de agregação/verificação fora do repositório (apagáveis, não versionados):
  - `/tmp/e065_aggregate.py` — carga das 100 células, resumos, deltas pareados;
  - `/tmp/e065_scope_analysis.py` — análise da classe F por escopo via `predictions.parquet`;
  - `/tmp/e065_repro_check.py` — identidade v1/v2/v3 e consistência de manifests;
  - `/tmp/e065_final_tables.py` — tabelas por fold, wins pareados, runs degeneradas;
  - `/tmp/e065_make_fragments.py` — geração das tabelas markdown embutidas aqui.

Artefatos lidos (nenhum modificado):

- `experiments/stage2_v2.4_research/E06_5/e065-audit-v3/**` (100 células + `summary.json`)
- `experiments/stage2_v2.4_research/E06_5/e065-audit-v1/**`, `.../e065-audit-v2/**` (comparação)
- `experiments/stage2_v2.4_research/selections/representation_selection.json`
- `experiments/stage2_v2.4_research/fold_audits/e065-audit-v3/fold5_report.json`
- `experiments/stage2_v2.4_research/E01_patient_distribution/f_concentration_report.json`
- `experiments/stage2_v2.4_research/manifests/feature_manifests.json`
- `config/stage2_research.yaml`

## 4. VARIÁVEL CONCEITUAL ALTERADA

```text
representation (feature schema)
```

Somente o esquema de features varia entre candidatos; todos os demais fatores estão congelados e foram verificados como idênticos nas 100 células (constantes extraídas dos `run_manifest.json` e de `config/stage2_research.yaml`):

| Fator | Valor congelado |
| --- | --- |
| Arquitetura | `minimal_mlp_128` |
| Loss | `sparse_categorical_crossentropy` (CE pura; sem focal, sem class weight) |
| Sampling | `natural` (sem SMOTE/oversampling) |
| Decisão | argmax do softmax cru (sem thresholds/calibração) |
| Dispositivo | `cpu:CPU`, `deterministic: true`, profile `audit` |
| Seleção/early stopping | inner-only (`outer_test_used_for_selection: false`; early stop em `inner_validation`, patience 5, max 30 épocas, batch 512) |
| Imputação/scaling | mediana do outer train / StandardScaler do outer train |
| Split | StratifiedGroupKFold 5 folds × 4 inner splits, `split_random_state: 42` |
| dataset_manifest_hash | `168224d0198233f4619f356745366dd2775c23525109bba8d6ae029685ebe6f5` (único nas 100 células) |
| split_manifest_hash | `d02d284c532da7eed8367c74b89539d79a0a580ddebe11d1730fde88f79f1593` (único nas 100 células) |

Composição das representações (`manifests/feature_manifests.json`):

| Candidato | n_features | Famílias | Templates de fusão | manifest_hash (collection) |
| --- | --- | --- | --- | --- |
| baseline | 16 | `base16` | 0 | `244fe5cb80dadf02…` |
| H6 | 54 | `base16, causal_rr_h3, class_templates_h5` | 8 | `50762aa120d6ad0f…` |
| H11 | 54 | idem H6 | 16 | `8d899c1b17d18231…` |
| H12 | 54 | idem H6 | 24 | `fe36ea004983aa00…` |

Nota: H6/H11/H12 têm **as mesmas 54 colunas** (16 base + 17 `causal_rr_h3` + 21 `class_templates_h5`); o que muda é o **banco de templates de fusão** (8/16/24 protótipos, shape `[k, 150]`, janela QRS −120 ms/+180 ms a 500 Hz), que altera os valores das 21 features de template. Os templates são ajustados **somente com dados de treino** do fold (`template_fit_scope: inner_train_then_outer_train`; hash versionado por fold em `manifests/features/<candidato>/fold_N/`), com seed fixa 42 e teto de 256 batimentos/grupo — sem contribuição do outer test.

## 5. CHECAGEM PÓS-OPERATÓRIA

Escopo desta etapa: auditoria de artefatos (sem alteração de código). Itens não aplicáveis estão marcados `[NOT RUN]`.

```text
[PASS] integridade da campanha — 100/100 células com metrics.json + DONE (4×5×5, sem runs faltantes)
[PASS] dataset manifest — hash único 168224d0…ebe6f5 nas 100 células
[PASS] feature schema — 4 manifests versionados; templates fit apenas em treino (inner→outer), por fold
[PASS] split integrity — split hash único d02d284c…f1593; suportes F por fold consistentes com o fold5_report
[PASS] no group overlap — outer test dos folds 1/2 contém 208/213; folds 3/4/5 contêm apenas remaining groups; fold 5 auditado (11 pacientes F, 0 sobreposição)
[PASS] seleção inner-only — outer_test_used_for_selection=false nas 100 células
[PASS] determinismo — deterministic=true, device cpu:CPU, runtime_identity_hash único c936105e…
[PASS] NaN/Inf — todas as métricas carregadas finitas (estatísticas agregadas finitas em 100/100 células)
[PASS] reprodutibilidade — 0 diferenças em 100 células × 8 chaves de métrica entre e065-audit-v1/v2/v3
[PASS] cross-check métricas — scopes do metrics.json == recomputação do predictions.parquet (diff < 1e-9); negative margin fold 5 recomputado == 0.9944444444444445 do artefato
[PASS] artifacts isolated — nenhuma escrita em models/, src/, tests/, config/; artefatos v2.3 intocados
[NOT RUN] testes unitários / flake8 / mypy — nenhum arquivo de código modificado nesta etapa
```

Observação de rastreabilidade: os `run_manifest.json` registram `git_head 6957bf6a90735c003f46f494f7877be3a3fd8090` com `git_dirty: true` (working tree continha modificações não commitadas durante a campanha). Registrado para auditoria; não invalida a cadeia porque dataset/split/feature/runtime hashes são estáveis e a reprodução bitwise foi demonstrada (seção 11).

## 6. RESULTADOS

Baseline e candidatos lado a lado, sempre com as 25 runs de cada candidato (5 folds × 5 seeds). **Nenhum número abaixo é "o melhor fold".**

### 6.1 Resumo por candidato (25 runs cada; std ddof=0)

| Candidato | Métrica | mean | std | min | p10 | median | p90 | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | F1(S) | 0.489 | 0.150 | 0.266 | 0.295 | 0.473 | 0.735 | 0.781 |
| baseline | F1(V) | 0.802 | 0.034 | 0.756 | 0.762 | 0.797 | 0.862 | 0.879 |
| baseline | F1(F) | 0.042 | 0.044 | 0.000 | 0.000 | 0.039 | 0.116 | 0.121 |
| baseline | F1-macro | 0.445 | 0.062 | 0.357 | 0.366 | 0.439 | 0.554 | 0.572 |
| baseline | Prec(F) | 0.188 | 0.254 | 0.000 | 0.000 | 0.067 | 0.600 | 0.786 |
| baseline | Rec(F) | 0.028 | 0.029 | 0.000 | 0.000 | 0.021 | 0.067 | 0.070 |
| baseline | AP(F) | 0.178 | 0.184 | 0.031 | 0.047 | 0.069 | 0.522 | 0.574 |
| baseline | runs F1(F)=0 | 11/25 | — | — | — | — | — | — |
| H6 | F1(S) | 0.501 | 0.132 | 0.306 | 0.324 | 0.476 | 0.721 | 0.746 |
| H6 | F1(V) | 0.800 | 0.038 | 0.740 | 0.746 | 0.794 | 0.864 | 0.871 |
| H6 | F1(F) | 0.137 | 0.095 | 0.000 | 0.019 | 0.130 | 0.286 | 0.303 |
| H6 | F1-macro | 0.479 | 0.076 | 0.407 | 0.421 | 0.439 | 0.623 | 0.636 |
| H6 | Prec(F) | 0.416 | 0.265 | 0.000 | 0.115 | 0.333 | 0.685 | 1.000 |
| H6 | Rec(F) | 0.091 | 0.064 | 0.000 | 0.010 | 0.075 | 0.183 | 0.201 |
| H6 | AP(F) | 0.313 | 0.192 | 0.054 | 0.063 | 0.234 | 0.573 | 0.600 |
| H6 | runs F1(F)=0 | 2/25 | — | — | — | — | — | — |
| H11 | F1(S) | 0.500 | 0.130 | 0.305 | 0.321 | 0.486 | 0.714 | 0.749 |
| H11 | F1(V) | 0.800 | 0.037 | 0.743 | 0.748 | 0.796 | 0.861 | 0.867 |
| H11 | F1(F) | 0.147 | 0.096 | 0.000 | 0.019 | 0.132 | 0.276 | 0.302 |
| H11 | F1-macro | 0.482 | 0.072 | 0.409 | 0.427 | 0.448 | 0.618 | 0.632 |
| H11 | Prec(F) | 0.429 | 0.260 | 0.000 | 0.127 | 0.429 | 0.724 | 1.000 |
| H11 | Rec(F) | 0.099 | 0.067 | 0.000 | 0.010 | 0.105 | 0.180 | 0.206 |
| H11 | AP(F) | 0.312 | 0.196 | 0.059 | 0.063 | 0.211 | 0.574 | 0.618 |
| H11 | runs F1(F)=0 | 2/25 | — | — | — | — | — | — |
| H12 | F1(S) | 0.502 | 0.131 | 0.305 | 0.324 | 0.485 | 0.717 | 0.751 |
| H12 | F1(V) | 0.800 | 0.037 | 0.743 | 0.748 | 0.795 | 0.861 | 0.873 |
| H12 | F1(F) | 0.145 | 0.094 | 0.000 | 0.019 | 0.128 | 0.283 | 0.316 |
| H12 | F1-macro | 0.482 | 0.074 | 0.409 | 0.426 | 0.446 | 0.618 | 0.638 |
| H12 | Prec(F) | 0.429 | 0.267 | 0.000 | 0.153 | 0.353 | 0.738 | 1.000 |
| H12 | Rec(F) | 0.097 | 0.066 | 0.000 | 0.010 | 0.093 | 0.183 | 0.219 |
| H12 | AP(F) | 0.312 | 0.194 | 0.058 | 0.061 | 0.224 | 0.577 | 0.601 |
| H12 | runs F1(F)=0 | 2/25 | — | — | — | — | — | — |

Leitura conforme as regras de interpretação do doc mestre: o F1(F) médio aumentou de 0.042 (baseline) para 0.137/0.147/0.145 (H6/H11/H12); 3 de 5 folds melhoraram de forma limpa (1, 3, 4 — ver seção 7); o pior fold mudou de 0.000 (baseline, folds 4 e 5) para 0.012–0.015 (candidatos, fold 5); a variabilidade **aumentou** (std 0.044 → 0.094–0.096; p90−p10 0.116 → 0.257–0.268) porque o ganho é concentrado nos folds 1/3/4. Recall(F) médio sobe de 0.028 para 0.091–0.099 e AP(F) de 0.178 para 0.312–0.313 — discriminação aproximadamente duplicada, porém ainda baixa em termos absolutos. F1(S) médio essencialmente estável (0.489 → 0.500–0.502) e F1(V) inalterado (0.802 → 0.800).

### 6.2 Deltas pareados (fold, seed) vs baseline — valores do artefato de seleção

Fonte: `experiments/stage2_v2.4_research/selections/representation_selection.json` (n=25 pares por comparação). Recomputação independente reproduziu mean/median/std/win idênticos; o CI95 do artefato é levemente assimétrico (estimador não-paramétrico), com semic amplitude compatível com o CI paramétrico (±1,96·SE).

| Comparação | Δ mean | Δ median | Δ std | CI95 | win fraction | Classificação |
| --- | --- | --- | --- | --- | --- | --- |
| H6 − baseline | +0.095 | +0.068 | 0.100 | [+0.057, +0.134] | 0.80 (20/25) | ROBUST_GAIN |
| H11 − baseline | +0.105 | +0.077 | 0.104 | [+0.064, +0.146] | 0.80 (20/25) | ROBUST_GAIN |
| H12 − baseline | +0.103 | +0.088 | 0.099 | [+0.064, +0.141] | 0.80 (20/25) | ROBUST_GAIN |
| H11 − H6 | +0.010 | +0.000 | 0.026 | [−0.000003, +0.020] | 0.56 (14/25) | **GAIN_WITHIN_TRAINING_VARIANCE** |
| H12 − H6 | +0.008 (médias) | — | — | — | — | sem ganho isolado que justifique +16 templates |

Decisão registrada no artefato (`selected_name: H6`, `selected_feature_manifest_hash: 50762aa1…`): a policy da research branch não substitui H6 por H11 sem ganho > variabilidade entre seeds, e H12 não vence por delta isolado. **H6 é a representação selecionada da etapa E06.5.**

Deltas pareados de F1-macro (recomputados; mean / median / min / max / win): H6 +0.035 / +0.054 / −0.031 / +0.092 / 0.68; H11 +0.038 / +0.058 / −0.030 / +0.102 / 0.68; H12 +0.038 / +0.055 / −0.030 / +0.089 / 0.68. Ou seja: o ganho de F1-macro é real em média, mas **8 de 25 runs regridem** em F1-macro para cada candidato (concentrados nos folds 2 e 5 — ver seções 7 e 9).

### 6.3 Confronto com os gates da research branch (`config/stage2_research.yaml`)

| Gate | Threshold | baseline | H6 | H11 | H12 | Desfecho |
| --- | --- | --- | --- | --- | --- | --- |
| `publication_f1_f` | mean F1(F) ≥ 0.50 | 0.042 | 0.137 | 0.147 | 0.145 | **TARGET_NOT_MET** (melhor run isolado: 0.316) |
| `research_candidate_f1_f` | mean F1(F) ≥ 0.25 | — | 0.137 | 0.147 | 0.145 | não atingido por nenhum candidato |
| `research_baseline_f1_f` | mean F1(F) ≥ 0.18 | 0.042 | 0.137 | 0.147 | 0.145 | não atingido por nenhum candidato |
| `minimum_macro_f1` | mean F1-macro ≥ 0.45 | 0.445 | 0.479 | 0.482 | 0.482 | candidatos acima; baseline abaixo |
| `material_gain_outside_208_213` | Δ F1(F) outside ≥ 0.05 | ref. (0.013) | +0.067 (→0.080) | +0.079 (→0.092) | +0.084 (→0.097) | **atingido pelos 3 candidatos** |

Referência contextual (não é gate formal desta etapa): o QG5' do projeto para o Estágio 2 (AGENTS.md) pede F1(F) ≥ 0.15, F1(S) ≥ 0.55, F1(V) ≥ 0.70, F1-macro ≥ 0.45 — os candidatos ficam marginalmente abaixo em F1(F) (0.137–0.147) e em F1(S) (0.500–0.502) no protocolo desta research branch.

### 6.4 Tabela completa — 100 células (candidato × fold × seed)

Runs com F1(F)=0 marcadas como `**0.000**†` (run degenerada; detalhe na seção 9). Valores com 3 casas decimais; fonte: `metrics.json` de cada célula.

| Candidato | Fold | Seed | F1(S) | F1(V) | F1(F) | F1-macro | Prec(F) | Rec(F) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 1 | 17 | 0.758 | 0.866 | 0.039 | 0.554 | 0.242 | 0.021 |
| baseline | 1 | 29 | 0.781 | 0.879 | 0.056 | 0.572 | 0.786 | 0.029 |
| baseline | 1 | 43 | 0.696 | 0.848 | 0.024 | 0.523 | 0.147 | 0.013 |
| baseline | 1 | 71 | 0.737 | 0.866 | 0.054 | 0.552 | 0.393 | 0.029 |
| baseline | 1 | 101 | 0.732 | 0.856 | 0.097 | 0.562 | 0.396 | 0.055 |
| baseline | 2 | 17 | 0.473 | 0.756 | 0.119 | 0.449 | 0.520 | 0.067 |
| baseline | 2 | 29 | 0.460 | 0.758 | 0.113 | 0.444 | 0.615 | 0.062 |
| baseline | 2 | 43 | 0.493 | 0.769 | **0.000**† | 0.421 | 0.000 | 0.000 |
| baseline | 2 | 71 | 0.473 | 0.763 | 0.121 | 0.452 | 0.578 | 0.067 |
| baseline | 2 | 101 | 0.476 | 0.761 | 0.118 | 0.452 | 0.694 | 0.065 |
| baseline | 3 | 17 | 0.394 | 0.779 | 0.069 | 0.414 | 0.085 | 0.058 |
| baseline | 3 | 29 | 0.493 | 0.799 | 0.068 | 0.453 | 0.067 | 0.070 |
| baseline | 3 | 43 | 0.441 | 0.790 | 0.062 | 0.431 | 0.068 | 0.058 |
| baseline | 3 | 71 | 0.466 | 0.796 | 0.044 | 0.435 | 0.042 | 0.047 |
| baseline | 3 | 101 | 0.395 | 0.775 | 0.073 | 0.414 | 0.076 | 0.070 |
| baseline | 4 | 17 | 0.266 | 0.805 | **0.000**† | 0.357 | 0.000 | 0.000 |
| baseline | 4 | 29 | 0.285 | 0.802 | **0.000**† | 0.362 | 0.000 | 0.000 |
| baseline | 4 | 43 | 0.309 | 0.810 | **0.000**† | 0.373 | 0.000 | 0.000 |
| baseline | 4 | 71 | 0.310 | 0.804 | **0.000**† | 0.371 | 0.000 | 0.000 |
| baseline | 4 | 101 | 0.280 | 0.803 | **0.000**† | 0.361 | 0.000 | 0.000 |
| baseline | 5 | 17 | 0.542 | 0.789 | **0.000**† | 0.443 | 0.000 | 0.000 |
| baseline | 5 | 29 | 0.387 | 0.785 | **0.000**† | 0.391 | 0.000 | 0.000 |
| baseline | 5 | 43 | 0.519 | 0.797 | **0.000**† | 0.439 | 0.000 | 0.000 |
| baseline | 5 | 71 | 0.524 | 0.793 | **0.000**† | 0.439 | 0.000 | 0.000 |
| baseline | 5 | 101 | 0.544 | 0.798 | **0.000**† | 0.447 | 0.000 | 0.000 |
| H6 | 1 | 17 | 0.723 | 0.865 | 0.271 | 0.620 | 0.644 | 0.172 |
| H6 | 1 | 29 | 0.733 | 0.866 | 0.280 | 0.626 | 0.670 | 0.177 |
| H6 | 1 | 43 | 0.688 | 0.852 | 0.303 | 0.615 | 0.623 | 0.201 |
| H6 | 1 | 71 | 0.717 | 0.861 | 0.298 | 0.625 | 0.600 | 0.198 |
| H6 | 1 | 101 | 0.746 | 0.871 | 0.291 | 0.636 | 0.651 | 0.187 |
| H6 | 2 | 17 | 0.467 | 0.747 | 0.088 | 0.434 | 0.720 | 0.047 |
| H6 | 2 | 29 | 0.460 | 0.740 | 0.078 | 0.426 | 0.696 | 0.041 |
| H6 | 2 | 43 | 0.467 | 0.748 | 0.100 | 0.438 | 0.636 | 0.054 |
| H6 | 2 | 71 | 0.456 | 0.745 | 0.069 | 0.423 | 0.667 | 0.036 |
| H6 | 2 | 101 | 0.469 | 0.740 | 0.135 | 0.448 | 0.659 | 0.075 |
| H6 | 3 | 17 | 0.544 | 0.802 | 0.095 | 0.480 | 0.150 | 0.070 |
| H6 | 3 | 29 | 0.542 | 0.813 | 0.150 | 0.502 | 0.213 | 0.116 |
| H6 | 3 | 43 | 0.521 | 0.804 | 0.130 | 0.485 | 0.216 | 0.093 |
| H6 | 3 | 71 | 0.519 | 0.811 | 0.096 | 0.475 | 0.154 | 0.070 |
| H6 | 3 | 101 | 0.528 | 0.801 | 0.096 | 0.475 | 0.154 | 0.070 |
| H6 | 4 | 17 | 0.306 | 0.800 | 0.204 | 0.437 | 0.326 | 0.149 |
| H6 | 4 | 29 | 0.323 | 0.790 | 0.175 | 0.429 | 0.279 | 0.128 |
| H6 | 4 | 43 | 0.325 | 0.803 | 0.158 | 0.429 | 0.244 | 0.117 |
| H6 | 4 | 71 | 0.310 | 0.793 | 0.214 | 0.439 | 0.378 | 0.149 |
| H6 | 4 | 101 | 0.326 | 0.794 | 0.144 | 0.421 | 0.290 | 0.096 |
| H6 | 5 | 17 | 0.503 | 0.791 | **0.000**† | 0.431 | 0.000 | 0.000 |
| H6 | 5 | 29 | 0.443 | 0.794 | 0.020 | 0.419 | 1.000 | 0.010 |
| H6 | 5 | 43 | 0.418 | 0.784 | 0.020 | 0.407 | 0.333 | 0.010 |
| H6 | 5 | 71 | 0.476 | 0.786 | **0.000**† | 0.421 | 0.000 | 0.000 |
| H6 | 5 | 101 | 0.515 | 0.790 | 0.018 | 0.441 | 0.091 | 0.010 |
| H11 | 1 | 17 | 0.709 | 0.861 | 0.276 | 0.615 | 0.626 | 0.177 |
| H11 | 1 | 29 | 0.749 | 0.867 | 0.279 | 0.632 | 0.595 | 0.182 |
| H11 | 1 | 43 | 0.670 | 0.848 | 0.269 | 0.596 | 0.589 | 0.174 |
| H11 | 1 | 71 | 0.717 | 0.861 | 0.302 | 0.627 | 0.565 | 0.206 |
| H11 | 1 | 101 | 0.726 | 0.865 | 0.269 | 0.620 | 0.660 | 0.169 |
| H11 | 2 | 17 | 0.461 | 0.748 | 0.074 | 0.428 | 0.789 | 0.039 |
| H11 | 2 | 29 | 0.466 | 0.743 | 0.073 | 0.427 | 0.625 | 0.039 |
| H11 | 2 | 43 | 0.464 | 0.748 | 0.087 | 0.433 | 0.643 | 0.047 |
| H11 | 2 | 71 | 0.459 | 0.749 | 0.079 | 0.429 | 0.762 | 0.041 |
| H11 | 2 | 101 | 0.471 | 0.748 | 0.131 | 0.450 | 0.667 | 0.073 |
| H11 | 3 | 17 | 0.542 | 0.809 | 0.109 | 0.487 | 0.167 | 0.081 |
| H11 | 3 | 29 | 0.544 | 0.814 | 0.132 | 0.497 | 0.180 | 0.105 |
| H11 | 3 | 43 | 0.520 | 0.803 | 0.156 | 0.493 | 0.238 | 0.116 |
| H11 | 3 | 71 | 0.522 | 0.810 | 0.121 | 0.485 | 0.174 | 0.093 |
| H11 | 3 | 101 | 0.540 | 0.804 | 0.134 | 0.493 | 0.188 | 0.105 |
| H11 | 4 | 17 | 0.305 | 0.798 | 0.275 | 0.459 | 0.432 | 0.202 |
| H11 | 4 | 29 | 0.320 | 0.791 | 0.175 | 0.429 | 0.279 | 0.128 |
| H11 | 4 | 43 | 0.328 | 0.806 | 0.210 | 0.448 | 0.306 | 0.160 |
| H11 | 4 | 71 | 0.305 | 0.793 | 0.233 | 0.444 | 0.429 | 0.160 |
| H11 | 4 | 101 | 0.322 | 0.794 | 0.215 | 0.444 | 0.389 | 0.149 |
| H11 | 5 | 17 | 0.510 | 0.791 | **0.000**† | 0.434 | 0.000 | 0.000 |
| H11 | 5 | 29 | 0.449 | 0.796 | 0.038 | 0.428 | 0.333 | 0.020 |
| H11 | 5 | 43 | 0.422 | 0.785 | 0.020 | 0.409 | 1.000 | 0.010 |
| H11 | 5 | 71 | 0.486 | 0.787 | **0.000**† | 0.424 | 0.000 | 0.000 |
| H11 | 5 | 101 | 0.490 | 0.786 | 0.018 | 0.431 | 0.100 | 0.010 |
| H12 | 1 | 17 | 0.718 | 0.863 | 0.259 | 0.613 | 0.663 | 0.161 |
| H12 | 1 | 29 | 0.729 | 0.862 | 0.273 | 0.621 | 0.629 | 0.174 |
| H12 | 1 | 43 | 0.675 | 0.848 | 0.296 | 0.606 | 0.612 | 0.195 |
| H12 | 1 | 71 | 0.716 | 0.860 | 0.316 | 0.631 | 0.568 | 0.219 |
| H12 | 1 | 101 | 0.751 | 0.873 | 0.289 | 0.638 | 0.667 | 0.185 |
| H12 | 2 | 17 | 0.464 | 0.748 | 0.074 | 0.429 | 0.750 | 0.039 |
| H12 | 2 | 29 | 0.465 | 0.743 | 0.087 | 0.432 | 0.692 | 0.047 |
| H12 | 2 | 43 | 0.466 | 0.747 | 0.088 | 0.434 | 0.720 | 0.047 |
| H12 | 2 | 71 | 0.460 | 0.753 | 0.097 | 0.437 | 0.769 | 0.052 |
| H12 | 2 | 101 | 0.472 | 0.748 | 0.122 | 0.448 | 0.667 | 0.067 |
| H12 | 3 | 17 | 0.545 | 0.805 | 0.105 | 0.485 | 0.149 | 0.081 |
| H12 | 3 | 29 | 0.544 | 0.814 | 0.168 | 0.509 | 0.244 | 0.128 |
| H12 | 3 | 43 | 0.528 | 0.808 | 0.161 | 0.499 | 0.263 | 0.116 |
| H12 | 3 | 71 | 0.516 | 0.808 | 0.108 | 0.478 | 0.159 | 0.081 |
| H12 | 3 | 101 | 0.535 | 0.805 | 0.128 | 0.489 | 0.205 | 0.093 |
| H12 | 4 | 17 | 0.305 | 0.798 | 0.236 | 0.446 | 0.340 | 0.181 |
| H12 | 4 | 29 | 0.327 | 0.793 | 0.152 | 0.424 | 0.263 | 0.106 |
| H12 | 4 | 43 | 0.332 | 0.806 | 0.199 | 0.446 | 0.298 | 0.149 |
| H12 | 4 | 71 | 0.305 | 0.793 | 0.203 | 0.434 | 0.382 | 0.138 |
| H12 | 4 | 101 | 0.322 | 0.795 | 0.188 | 0.435 | 0.353 | 0.128 |
| H12 | 5 | 17 | 0.505 | 0.788 | **0.000**† | 0.431 | 0.000 | 0.000 |
| H12 | 5 | 29 | 0.445 | 0.794 | **0.000**† | 0.413 | 0.000 | 0.000 |
| H12 | 5 | 43 | 0.422 | 0.785 | 0.020 | 0.409 | 1.000 | 0.010 |
| H12 | 5 | 71 | 0.485 | 0.787 | 0.019 | 0.430 | 0.167 | 0.010 |
| H12 | 5 | 101 | 0.508 | 0.788 | 0.036 | 0.444 | 0.167 | 0.020 |
## 7. ANÁLISE POR FOLD

Todos os 5 folds abaixo; nenhum fold omitido. `F test (208/213/out)` = suporte de batimentos F no outer test do fold, decomposto em record 208 / record 213 / remaining groups. Médias sobre as 5 seeds (std ddof=0).

| Candidato | Fold | F1(F) mean±std | F1(F) min–max | F1-macro | Prec(F) | Rec(F) | F1(S) | F1(V) | F test (208/213/out) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 1 | 0.054±0.024 | 0.024–0.097 | 0.553 | 0.393 | 0.030 | 0.741 | 0.863 | 379 (372/0/7) |
| baseline | 2 | 0.094±0.047 | 0.000–0.121 | 0.444 | 0.482 | 0.052 | 0.475 | 0.761 | 386 (0/362/24) |
| baseline | 3 | 0.063±0.010 | 0.044–0.073 | 0.430 | 0.067 | 0.060 | 0.438 | 0.788 | 86 (0/0/86) |
| baseline | 4 | 0.000±0.000 | 0.000–0.000 | 0.365 | 0.000 | 0.000 | 0.290 | 0.805 | 94 (0/0/94) |
| baseline | **5** | 0.000±0.000 | 0.000–0.000 | 0.432 | 0.000 | 0.000 | 0.503 | 0.792 | 99 (0/0/99) |
| H6 | 1 | 0.289±0.012 | 0.271–0.303 | 0.624 | 0.638 | 0.187 | 0.721 | 0.863 | 379 (372/0/7) |
| H6 | 2 | 0.094±0.023 | 0.069–0.135 | 0.434 | 0.676 | 0.051 | 0.464 | 0.744 | 386 (0/362/24) |
| H6 | 3 | 0.114±0.023 | 0.095–0.150 | 0.484 | 0.177 | 0.084 | 0.531 | 0.806 | 86 (0/0/86) |
| H6 | 4 | 0.179±0.027 | 0.144–0.214 | 0.431 | 0.304 | 0.128 | 0.318 | 0.796 | 94 (0/0/94) |
| H6 | **5** | 0.012±0.009 | 0.000–0.020 | 0.424 | 0.285 | 0.006 | 0.471 | 0.789 | 99 (0/0/99) |
| H11 | 1 | 0.279±0.012 | 0.269–0.302 | 0.618 | 0.607 | 0.182 | 0.714 | 0.860 | 379 (372/0/7) |
| H11 | 2 | 0.089±0.022 | 0.073–0.131 | 0.433 | 0.697 | 0.048 | 0.464 | 0.747 | 386 (0/362/24) |
| H11 | 3 | 0.131±0.016 | 0.109–0.156 | 0.491 | 0.189 | 0.100 | 0.534 | 0.808 | 86 (0/0/86) |
| H11 | 4 | 0.222±0.033 | 0.175–0.275 | 0.445 | 0.367 | 0.160 | 0.316 | 0.796 | 94 (0/0/94) |
| H11 | **5** | 0.015±0.014 | 0.000–0.038 | 0.425 | 0.287 | 0.008 | 0.471 | 0.789 | 99 (0/0/99) |
| H12 | 1 | 0.287±0.020 | 0.259–0.316 | 0.622 | 0.628 | 0.187 | 0.718 | 0.861 | 379 (372/0/7) |
| H12 | 2 | 0.094±0.016 | 0.074–0.122 | 0.436 | 0.720 | 0.050 | 0.466 | 0.748 | 386 (0/362/24) |
| H12 | 3 | 0.134±0.026 | 0.105–0.168 | 0.492 | 0.204 | 0.100 | 0.534 | 0.808 | 86 (0/0/86) |
| H12 | 4 | 0.195±0.027 | 0.152–0.236 | 0.437 | 0.327 | 0.140 | 0.318 | 0.797 | 94 (0/0/94) |
| H12 | **5** | 0.015±0.014 | 0.000–0.036 | 0.426 | 0.267 | 0.008 | 0.473 | 0.788 | 99 (0/0/99) |
**Estrutura dos folds quanto à classe F** (fonte: scopes dos `metrics.json` e `fold5_report.json`):

- O record 208 só aparece no outer test do **fold 1** (372 F) e o record 213 só no **fold 2** (362 F); eles nunca dividem o mesmo outer test.
- Os folds **3, 4 e 5** testam exclusivamente *remaining F groups* (86/94/99 batimentos F em 11/12/11 records, respectivamente) — são os folds mais informativos para generalização inter-paciente fora dos grupos dominantes.
- Records com F no outside por fold: fold 1 → 4 records/7 batimentos; fold 2 → 5/24; fold 3 → 11/86; fold 4 → 12/94; fold 5 → 11/99.

Leitura por fold:

- **Fold 1 (208 no teste):** ganho grande e limpo — F1(F) 0.054 → 0.279–0.289; 5/5 seeds melhoram para os 3 candidatos (Δ mín. +0.172). F1-macro 0.553 → 0.618–0.624 (5/5). Custo: F1(S) 0.741 → 0.714–0.721.
- **Fold 2 (213 no teste):** **sem ganho** — F1(F) 0.094 (baseline) vs 0.094/0.089/0.094 (H6/H11/H12); apenas 2/5 seeds melhoram; H11 regrid em média (−0.006). F1-macro regrid levemente (0.444 → 0.433–0.436; 1/5 seeds). Ver seção 8: os candidatos melhoram o outside do fold 2 (0.000 → 0.046–0.091), mas perdem um pouco no record 213 (0.103 → 0.092–0.096).
- **Fold 3 (só remaining):** ganho consistente — F1(F) 0.063 → 0.114/0.131/0.134; 5/5 seeds (Δ mín. +0.023). F1-macro 0.430 → 0.484–0.492 (5/5).
- **Fold 4 (só remaining):** baseline colapsa totalmente (F1(F)=0 em 5/5 seeds, 0 TP); candidatos recuperam sinal — F1(F) 0.179/0.222/0.195; 5/5 seeds (Δ mín. +0.144). F1-macro 0.365 → 0.431–0.445 (5/5).
- **Fold 5 (só remaining; OPTIMIZATION_COLLAPSE):** todos os candidatos colapsam praticamente a zero — F1(F) 0.012–0.015 com 2/5 runs degeneradas por candidato (baseline 5/5). Os 3/5 runs não-zero dos candidatos têm recall(F) de 0.01–0.02 (1–2 TPs). Detalhe em 7.1.

### 7.1 Fold 5 — OPTIMIZATION_COLLAPSE (destaque obrigatório)

Fonte: `experiments/stage2_v2.4_research/fold_audits/e065-audit-v3/fold5_report.json` (status PASS, classificação **OPTIMIZATION_COLLAPSE**, `audit_hash af2f561e…`), com recomputação independente a partir dos `predictions.parquet`.

Composição das partições do fold 5 (batimentos F):

| Partição | F total | F em 208 | F em 213 | F outside | Pacientes com F |
| --- | --- | --- | --- | --- | --- |
| outer_train | 945 | 372 | 362 | 211 | 34 |
| inner_train | 559 | 372 | 0 | 187 | 28 |
| inner_validation | 386 | 0 | 362 | 24 | 6 |
| **outer_test** | **99** | **0** | **0** | **99** | **11** |

Evidências quantitativas (pool das 20 runs do fold 5 = 4 candidatos × 5 seeds; 1.980 linhas de F verdadeiro):

- `negative_margin_fraction` = **0.9944** (margem `logit_F − max(logit_S, logit_V)` negativa em 99,44% dos F verdadeiros) — recomputado exatamente: 0.9944444444444445. Por candidato: baseline 1.000, H6 0.994, H11 0.992, H12 0.992.
- Probabilidade atribuída a F nos batimentos F verdadeiros: mean p_F 0.057 (baseline) / 0.068 (H6) / 0.071 (H11) / 0.069 (H12); mediana 0.035–0.043; **máximo < 0.50 em todos os candidatos** — o modelo nunca "acredita" em F nesses pacientes.
- Early stopping na época **1–2 em todas as 20 runs** do fold 5 (mediana 2), contra épocas 26–30 nos folds 1/3/4 dos candidatos: a perda de validação interna para de melhorar quase imediatamente quando o teste interno de early stopping é dominado por 213 (362 dos 386 F da inner_validation) e o treino interno por 208.
- O colapso ocorre **apesar de 945 batimentos F de 34 pacientes no outer train** — incluindo 208 e 213 inteiros. A informação que o modelo extrai de 208/213 e dos 211 F restantes de treino não transfere para os 11 pacientes F do outer test.

Interpretação: o fold 5 é o teste mais severo de generalização (nenhum paciente dominante no teste) e demonstra que o ganho dos candidatos, embora real nos folds 1/3/4, **não resolve** a transferência para *remaining F groups* no cenário mais adverso. Isto é evidência a favor da causa A (concentração por paciente) como limitante dominante residual, com a representação H6/H11/H12 mitigando mas não eliminando o problema.

## 8. ANÁLISE DA CLASSE F

Separação obrigatória: **record 208**, **record 213**, **remaining F groups** (todos os demais records com F). Tabela micro poolada sobre as 25 runs de cada candidato (cada batimento do outer test aparece 5×, uma por seed), calculada do zero a partir de `predictions.parquet` + `record_id` (label encoding verificado: `y=2 ↔ F`, confirmado por `F_support`; cross-check contra os scopes do `metrics.json` com diff < 1e-9).

| Candidato | Escopo | F verdadeiros | TP | FP | FN | Prec(F) | Rec(F) | F1(F) micro | mean p_F (F verd.) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | record_208 | 1860 | 56 | 11 | 1804 | 0.836 | 0.030 | 0.058 | 0.185 |
| baseline | record_213 | 1810 | 101 | 24 | 1709 | 0.808 | 0.056 | 0.104 | 0.069 |
| baseline | outside_208_213 | 1550 | 26 | 563 | 1524 | 0.044 | 0.017 | 0.024 | 0.087 |
| H6 | record_208 | 1860 | 351 | 77 | 1509 | 0.820 | 0.189 | 0.307 | 0.268 |
| H6 | record_213 | 1810 | 94 | 41 | 1716 | 0.696 | 0.052 | 0.097 | 0.121 |
| H6 | outside_208_213 | 1550 | 106 | 455 | 1444 | 0.189 | 0.068 | 0.100 | 0.160 |
| H11 | record_208 | 1860 | 340 | 98 | 1520 | 0.776 | 0.183 | 0.296 | 0.271 |
| H11 | record_213 | 1810 | 89 | 36 | 1721 | 0.712 | 0.049 | 0.092 | 0.121 |
| H11 | outside_208_213 | 1550 | 129 | 470 | 1421 | 0.215 | 0.083 | 0.120 | 0.163 |
| H12 | record_208 | 1860 | 350 | 92 | 1510 | 0.792 | 0.188 | 0.304 | 0.270 |
| H12 | record_213 | 1810 | 91 | 34 | 1719 | 0.728 | 0.050 | 0.094 | 0.122 |
| H12 | outside_208_213 | 1550 | 123 | 452 | 1427 | 0.214 | 0.079 | 0.116 | 0.160 |
Decomposição por fold × escopo (F1 micro poolado por seed-set; `—` = escopo ausente do teste do fold, F1=0 por convenção sem suporte):

| Candidato | Fold | F1(F) 208 | F1(F) 213 | F1(F) outside | Rec(F) outside |
| --- | --- | --- | --- | --- | --- |
| baseline | 1 | 0.058 | 0.000 | 0.000 | 0.000 |
| baseline | 2 | 0.000 | 0.104 | 0.000 | 0.000 |
| baseline | 3 | 0.000 | 0.000 | 0.063 | 0.060 |
| baseline | 4 | 0.000 | 0.000 | 0.000 | 0.000 |
| baseline | **5** | 0.000 | 0.000 | 0.000 | 0.000 |
| H6 | 1 | 0.307 | 0.000 | 0.037 | 0.086 |
| H6 | 2 | 0.000 | 0.097 | 0.061 | 0.033 |
| H6 | 3 | 0.000 | 0.000 | 0.114 | 0.084 |
| H6 | 4 | 0.000 | 0.000 | 0.179 | 0.128 |
| H6 | **5** | 0.000 | 0.000 | 0.012 | 0.006 |
| H11 | 1 | 0.296 | 0.000 | 0.048 | 0.114 |
| H11 | 2 | 0.000 | 0.092 | 0.047 | 0.025 |
| H11 | 3 | 0.000 | 0.000 | 0.131 | 0.100 |
| H11 | 4 | 0.000 | 0.000 | 0.222 | 0.160 |
| H11 | **5** | 0.000 | 0.000 | 0.015 | 0.008 |
| H12 | 1 | 0.304 | 0.000 | 0.049 | 0.114 |
| H12 | 2 | 0.000 | 0.094 | 0.092 | 0.050 |
| H12 | 3 | 0.000 | 0.000 | 0.134 | 0.100 |
| H12 | 4 | 0.000 | 0.000 | 0.196 | 0.140 |
| H12 | **5** | 0.000 | 0.000 | 0.016 | 0.008 |
Leitura honesta por escopo:

- **Record 208 (372 F; teste do fold 1):** ganho expressivo — F1 micro 0.058 → 0.296–0.307; recall 0.030 → 0.183–0.189 com precision praticamente preservada (0.836 → 0.776–0.820). É o maior contribuinte individual para o ganho agregado.
- **Record 213 (362 F; teste do fold 2):** **leve regressão** vs baseline — F1 micro 0.104 → 0.092–0.097; recall 0.056 → 0.049–0.052 e precision 0.808 → 0.696–0.728. O sinal novo das features não transfere para 213 quando 213 sai do treino; o baseline obtém seus poucos TPs com precision alta.
- **Remaining F groups (43 records distintos poolados; 1.550 linhas F):** ganho real e material — F1 micro 0.024 → 0.100/0.120/0.116; recall 0.017 → 0.068–0.083; precision 0.044 → 0.189–0.215. Em nível de run (média por run), 0.013 → 0.080/0.092/0.097 — acima do gate `material_gain_outside_208_213` (0.05) para os 3 candidatos. **Mas** o valor absoluto permanece baixo e o fold 5 (o teste remaining mais adverso) colapsa (seção 7.1). O ganho fora de 208/213 é, portanto, **real porém frágil e insuficiente**.
- Não é o caso de `PATIENT_SPECIFIC_GAIN_WARNING` puro: o ganho não existe *somente* em 208/213 — folds 3 e 4 (só remaining) melhoram 5/5 seeds. Também não é `REPRESENTATION_HYPOTHESIS_SUPPORTED` irrestrito: o fold 2 é plano e o fold 5 colapsa.

## 9. REGRESSÕES E RUNS DEGENERADAS

### 9.1 Runs degeneradas (F1(F)=0) — 17 de 100, preservadas sem descarte

| Candidato | Fold | Seeds com F1(F)=0 | n |
| --- | --- | --- | --- |
| baseline | 2 | 43 | 1/5 |
| baseline | 4 | 17, 29, 43, 71, 101 | 5/5 |
| baseline | 5 | 17, 29, 43, 71, 101 | 5/5 |
| H6 | 5 | 17, 71 | 2/5 |
| H11 | 5 | 17, 71 | 2/5 |
| H12 | 5 | 17, 29 | 2/5 |

Dois modos de falha distintos, ambos preservados como evidência: (i) fold 4 do baseline — o modelo treina 11–27 épocas e simplesmente nunca prediz F (0 TP em 5/5 seeds); (ii) fold 5 de todos os candidatos — parada na época 1–2 (colapso de otimização, seção 7.1). Nos runs não degeneradas do fold 5, os candidatos obtêm F1(F) ≤ 0.038 com recall(F) 0.01–0.02; em 3 células a precision(F)=1.000 corresponde a exatamente 1 TP sem FP (não é sinal útil).

### 9.2 Regressões detectadas (candidatos vs baseline)

1. **Record 213 (escopo F):** F1 micro 0.104 (baseline) → 0.097/0.092/0.094 (H6/H11/H12); fold 2 mean F1(F) do H11 (0.089) fica abaixo do baseline (0.094); apenas 2/5 seeds melhoram no fold 2 para qualquer candidato (Δ mín. −0.052).
2. **F1-macro nos folds 2 e 5:** fold 2 → 0.444 (baseline) vs 0.433–0.436 (candidatos); fold 5 → 0.432 vs 0.424–0.426. Em nível de run, 8/25 runs regridem F1-macro (pior Δ −0.031).
3. **F1(S):** fold 1 → 0.741 (baseline) vs 0.714–0.721 (candidatos, Δ pareado mín. −0.101); fold 2 → 0.475 vs 0.464–0.466; fold 5 → 0.503 vs 0.470–0.473. Média pareada global de F1(S) é ligeiramente positiva (+0.011 a +0.012) por causa dos folds 3/4, mas a regressão localizada nos folds 1/2/5 é sistemática (win 0.44–0.48).
4. **F1(V):** sem regressão material (Δ médio −0.001 a −0.002; dentro do ruído).
5. **Variância:** o std multi-seed de F1(F) sobe de 0.044 para 0.094–0.096 — consequência da concentração do ganho nos folds 1/3/4, não de instabilidade aleatória (as três cadeias reproduzem bitwise; a variância é estrutural, entre folds/pacientes).

## 10. HIPÓTESE

```text
SUPPORTED — com escopo limitado e ressalvas obrigatórias
```

- A representação `base16 + causal_rr_h3 + class_templates_h5` **aumenta** o F1(F) inter-paciente médio vs baseline de 16 features: +0.095 a +0.105 (CI95 do artefato exclui 0; win 0.80; reproduzido bitwise em 3 cadeias independentes) → **ROBUST_GAIN** frente à variância multi-seed. Classificação da campanha: `REPRESENTATION_SIGNAL_CONFIRMED`.
- O ganho entre candidatos H6/H11/H12 é **irrelevante**: H11−H6 = +0.0097 (CI95 [−0.000003, +0.020], win 0.56) → **GAIN_WITHIN_TRAINING_VARIANCE**; H12 sem ganho isolado. A policy selecionou **H6** (menor complexidade).
- O ganho é **concentrado mas não exclusivo** dos grupos dominantes: grande em 208 (fold 1), nulo/levemente negativo em 213 (fold 2), material mas pequeno nos remaining groups (folds 3/4 melhoram 5/5; fold 5 colapsa com `OPTIMIZATION_COLLAPSE`).
- **TARGET_NOT_MET**: nenhum candidato atinge `research_baseline_f1_f` (0.18), `research_candidate_f1_f` (0.25) ou `publication_f1_f` (0.50) em média; o melhor run isolado chega a 0.316 (H12, fold 1, seed 71) e **não é** base para publicação.
- Efeitos colaterais registrados: regressão leve de F1(S) nos folds 1/2/5, de F1-macro nos folds 2/5 (8/25 runs) e do escopo 213.

## 11. CHECKPOINT

```text
PASS
```

Justificativa: hipótese testada corretamente e suportada com escopo limitado; nenhuma falha de integridade. Verificações desta etapa:

- **Integridade da campanha:** 100/100 células `DONE`; nenhum run descartado; todas as seeds pré-registradas em `config/stage2_research.yaml`.
- **Reprodutibilidade (fecha o `ROBUSTNESS_VALIDATION_REQUIRED` de `docs/stage2_e065_robustness_report.md`):** três cadeias independentes (`e065-audit-v1/v2/v3`) produzem métricas **idênticas até o último dígito** — 0 diferenças em 100 células × 8 chaves (`F1_S, F1_V, F1_F, macro_F1, precision_F, recall_F, AP_F, best_epoch`); ex.: F1(F) médio H11 = `0.14703608392635772` nas três. Os `aggregate_hash` diferem entre cadeias por cobrirem identificadores de diretório/ids próprios de cada cadeia — os agregados numéricos são idênticos.
- **Hashes únicos nas 100 células:** dataset `168224d0198233f4619f356745366dd2775c23525109bba8d6ae029685ebe6f5`; split `d02d284c532da7eed8367c74b89539d79a0a580ddebe11d1730fde88f79f1593`; runtime `c936105e95af0c51b1ac60a9c8ead03d1fdfe5c1c96dac7ca238c5ff7f3d39ff`; preflight `4248e7dcdd9be6bbbf54698ea36c88f439474a03563ce73a98b09fa827972aa8`.
- **Sem leakage de seleção:** `outer_test_used_for_selection=false` nas 100 células; templates fit somente em treino (inner→outer), versionados por fold.
- **Cross-checks:** scopes do `metrics.json` == recomputação de `predictions.parquet` (diff < 1e-9); `negative_margin_fraction` do fold 5 recomputado == 0.9944444444444445; estatísticas do `summary.json` == recomputação pandas (diff < 1e-12); deltas pareados do `representation_selection.json` == recomputação (mean/median/std/win idênticos).
- **Nada publicado:** `models/` intocado; artefatos v2.3 preservados; nenhum "melhor fold" promovido.

Classificações finais da etapa: `REPRESENTATION_SIGNAL_CONFIRMED` + `TARGET_NOT_MET` + `GAIN_WITHIN_TRAINING_VARIANCE` (H11/H12 vs H6) + `OPTIMIZATION_COLLAPSE` (fold 5, já classificado no artefato de auditoria).

## 12. PRÓXIMA ETAPA AUTORIZADA

```text
E07 — AUDITORIA DE SAMPLING, com representação H6 congelada
```

Protocolo conforme `config/stage2_research.yaml` (`e07`): samplers `natural` (controle), `random_oversampling`, `patient_uniform`, `patient_sqrt`, `smote` (k=5, somente no treino do fold); seeds de triagem `[17, 29, 43]` e finais `[17, 29, 43, 71, 101]`; invariantes de E06.5 mantidos (MLP-128, CE, argmax, CPU determinístico, seleção inner-only). A variável conceitual de E07 é **sampling policy** — nenhuma outra mudança autorizada. A pergunta a responder: patient-aware sampling melhora o recall(F) nos *remaining F groups* (em especial o colapso do fold 5) sem destruir precision(F)/F1-macro?

---

## Apêndice A — Artefatos-fonte deste relatório

| Item | Path |
| --- | --- |
| Células da campanha | `experiments/stage2_v2.4_research/E06_5/e065-audit-v3/<candidato>/fold_N/seed_M/{metrics.json, predictions.parquet, run_manifest.json, DONE}` |
| Agregados da campanha | `experiments/stage2_v2.4_research/E06_5/e065-audit-v3/summary.json` |
| Cadeias de reprodução | `experiments/stage2_v2.4_research/E06_5/e065-audit-v1/`, `.../e065-audit-v2/` |
| Seleção de representação | `experiments/stage2_v2.4_research/selections/representation_selection.json` |
| Auditoria do fold 5 | `experiments/stage2_v2.4_research/fold_audits/e065-audit-v3/fold5_report.json` |
| Concentração da classe F | `experiments/stage2_v2.4_research/E01_patient_distribution/f_concentration_report.json` |
| Manifests de features | `experiments/stage2_v2.4_research/manifests/feature_manifests.json` (+ `manifests/features/<candidato>/fold_N/`) |
| Configuração congelada | `config/stage2_research.yaml` |

## Apêndice B — Discrepâncias e notas de auditoria encontradas

1. `aggregate_hash` difere entre `e065-audit-v1/v2/v3` (cobre identificadores da cadeia), mas **todas** as métricas agregadas e de célula são idênticas até o último dígito — sem impacto científico.
2. O CI95 de `representation_selection.json` é levemente assimétrico (estimador não-paramétrico); a recomputação paramétrica (±1,96·SE) produz limites compatíveis (ex.: H11−baseline [+0.063, +0.147] vs [+0.064, +0.146] do artefato). Decisão inalterada.
3. `git_dirty: true` nos `run_manifest.json` da campanha (ver seção 5).
4. `docs/stage2_v2.4_research_report.md` (linha de pesquisa **anterior**, E00–E09 com definições distintas de E07/E08) descreve a classe F como "fibrilação atrial/flutter"; na ontologia v3.0.0 (AGENTS.md, regra 17) **F = batimento de fusão (FUSION V+N)** — este relatório usa a ontologia corrente. O documento anterior não foi alterado.
5. Escopo `record_213` dos `metrics.json` reporta F1=0 com suporte 0 nos folds sem 213 no teste (convenção de escopo vazio); as tabelas da seção 8 marcam esses casos explicitamente.
