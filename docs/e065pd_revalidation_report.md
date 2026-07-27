# E06.5-PD — relatório de revalidação patient-disjoint (FASE 4)

**Data:** 2026-07-26 | **Matriz:** 4 candidatos × 5 folds × 5 seeds = **100/100 células, 0 falhas** | **Status:** `COMPLETE — NO_VALID_CANDIDATE`

## 1. Configuração congelada

Splits `v4.0-patient-disjoint`; MLP-128; CE natural; argmax cru; imputação/scaling train-only; templates fitados somente em grupos de treino; early stopping inner-only; CPU determinística serial. Nenhuma representação, gate ou threshold alterado; `models/` intocado.

## 2. Resultados agregados OOF (média de 5 seeds por amostra)

| Candidato | F1(F) | F1-macro |
| --- | ---: | ---: |
| baseline | **0,6325** | **0,7067** |
| H6 (pré-comprometido) | 0,4724 | 0,6700 |
| H11 (descritivo) | 0,4942 | 0,6801 |
| H12 (descritivo) | 0,5389 | 0,6831 |

## 3. F1(F) por fold (média das seeds)

| Candidato | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 0,666 | 0,790 | 0,148 | 0,066 | 0,041 |
| H6 | 0,328 | 0,750 | 0,213 | 0,197 | 0,073 |
| H11 | 0,432 | 0,704 | 0,187 | 0,250 | 0,008 |
| H12 | 0,484 | 0,742 | 0,195 | 0,262 | 0,008 |

Zero-F1 runs: concentrados nos folds 4–5 (H11/H12 com 3/5 e 2/5 seeds em zero no fold 5; baseline/H6 sem zeros absolutos).

## 4. Inferência pré-registrada

- Bootstrap pareado por paciente sobre OOF (10.000 repetições, seeds aninhadas): **H6 − baseline ΔF1(F) = −0,1601**, IC95 **[−0,3983; +0,1534]** — inclui zero e ponto estimado negativo.
- Razão de decisão: `H6_GAIN_OVER_BASELINE_NOT_ROBUST`.
- H11/H12 são descritivos e **não podem substituir H6** sob o pré-registro (mesmos outer tests).

## 5. Leitura científica

Sob avaliação patient-disjoint autenticada, o ranking da era record-disjoint **não se reproduz**: o baseline base16 domina todas as representações com templates em F1(F). A dificuldade é estrutural por fold (massa F concentrada em 2–3 pacientes nos folds 1–2 vs dispersa em 3–5). A seleção congelada registra:

```text
status = NO_VALID_CANDIDATE
selection_hash = bc67148a2f2a1ff3fb216d0617d1318ec6875a6de847c3034d2a5b4f7e079b5e
```

## 6. Consequência

Sem candidato H*-PD válido → E07-PD não autorizado (FASE 5 registrada como bloqueio formal, 0/150 células). Evidência: `experiments/stage2_v2.4_research/E06_5_PD/e06-5-pd-v4-0/` (100 células write-once com manifests, predições com `patient_id`, métricas e hashes por célula).
