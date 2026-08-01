# Reconciliação de pré-treinos — ML Protocol v2 (T9.3)

**Data:** 2026-08-01 · **Branch:** `develop` · **Task:** T9.3 · **Avaliador:** `v2.0`
(`src/evaluation/canonical_evaluator.py`, commit `9d26ca7`)

> Os 3 runs de pré-treino Chapman foram reavaliados com o avaliador canônico, sem modificar
> nenhum artefato original. Artefatos novos em `experiments/<run>/evaluation_v2/` (7 JSONs +
> `predictions/`). Todos os números abaixo foram lidos desses artefatos ou computados sobre
> `predictions.npz` (read-only). Nada aqui é estimativa não marcada.

---

## 1. Resumo executivo

1. **O avaliador v2.0 está validado contra o legado**: no A2-full, TODOS os deltas de métricas
   por classe (AUROC, PR-AUC, F1@0.5) e macro-AUROC são **exatamente 0.0**; ECE e T divergem por
   ≤ 1e-8 (ruído de ponto flutuante). No A0 novo, Δ macro-AUROC = 1,3e-9. A regeneração de
   predições a partir do checkpoint reproduz o split e as métricas legadas bit a bit.
2. **BCE pós-temperatura exato, pela primeira vez**: A2-full **0,3417** · A0 novo **0,3869** ·
   A0 histórico **0,3905**. Nenhum se aproxima de 0,15 — munição central para a RFC T9.5: o
   braço BCE do QG4 é inatingível nesta tarefa, mesmo com calibração quase perfeita (ECE 0,015).
3. **A0 hist × A0 novo são COMPARABLE** (mesmo split seed 42, mesmo avaliador): ΔAUC 0,0031 com
   ICs sobrepostos → a diferença pré/pós-strict é **ruído**, confirmando a leitura do benchmark.
4. **A0 × A2-full são NON_COMPARABLE por `split_id`** (seed 42 ≠ 13) — os deltas A0→A2 citados
   historicamente (+2,3 p.p.) misturam efeito real com variação de split. Resolver exige as
   células de controle da T10.3 sob split único (H7/H8 da auditoria).
5. **ECE estratificado por NORM** revela que a temperatura global calibra a marginal, não os
   estratos: NORM tem ECE pós-T 0,061 no subconjunto NORM=1 vs **0,217** em NORM=0; STTC 0,051
   vs **0,169**. Suporte quantificado para calibração por classe/estrato (C2/C3, hipótese H5).

---

## 2. Tabela 1 — Reconciliação macro

| Run | AUC legada | AUC v2 | Δ | PR-AUC v2 | BCE v2 | BCE pós-T | ECE pré | ECE pós-T | T |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A0 histórico `20260728_033533` | 0,8336 † | 0,8334 | 2,3e-4 † | 0,6734 | 0,3907 | **0,3905** | 0,0191 | 0,0185 | 0,9722 (fit, RETRO) |
| A0 novo `20260729_042301` | 0,8365 | 0,8364643 | 1,3e-9 | 0,6785 | 0,3880 | **0,3869** | 0,0251 | 0,0206 | 0,9130 (congelada) |
| **A2-full `20260728_053011`** | 0,8639394700 | 0,8639394700 | **0.0** | **0,7065** | 0,4317 | **0,3417** | 0,1508 | 0,0152 | 0,3741 (congelada) |

† AUC "legada" do A0 histórico é o log Keras (batch-averaged); o Δ de 2,3e-4 mede a diferença
de caminho de métrica, não de modelo. A0 novo/A2-full reconciliam contra a avaliação offline
legada (`metrics_per_class.json` pinado).

IC95 bootstrap (n=200, macro):

| Run | macro_auroc IC95 | macro_pr_auc IC95 |
|---|---|---|
| A0 histórico | [0,8310; 0,8354] | [0,6696; 0,6779] |
| A0 novo | [0,8341; 0,8386] | [0,6745; 0,6831] |
| A2-full | [0,8623; 0,8660] | [0,7031; 0,7099] |

Leituras: (i) ICs de A0 hist × A0 novo **sobrepostos** → diferença strict é ruído;
(ii) A2-full está ~3 p.p. acima dos ICs dos A0 → ganho real de ranking, embora sob split
diferente (NON_COMPARABLE, Tabela 3); (iii) macro-F1@0.5→tuned: A2-full 0,6089→**0,6847**,
A0 novo 0,5623→0,6534, A0 hist 0,5322→0,6499 (RETROSPECTIVE — refit prospectivo na T10.3).

## 3. Tabela 2 — A2-full por classe (n=45.040; IC95 PR-AUC bootstrap n=200, nível relatório)

| Classe | Suporte | PR-AUC | IC95 PR-AUC | F1@0.5 | F1@tuned ‡ | ECE pré | ECE pós-T | Brier pré |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| NORM | 33.750 | 0,9887 | [0,9875; 0,9899] | 0,9557 | 0,9562 (t=0,557) | 0,1293 | 0,0098 | 0,0696 |
| CD | 7.350 | 0,5561 | [0,5450; 0,5670] | 0,3881 | 0,5178 (t=0,239) | 0,1693 | 0,0184 | 0,1319 |
| MI | 13.060 | 0,6252 | [0,6175; 0,6339] | 0,4931 | 0,5905 (t=0,322) | 0,1406 | 0,0100 | 0,1841 |
| HYP | 9.930 | 0,5078 | [0,4957; 0,5176] | 0,4208 | 0,5612 (t=0,301) | 0,1470 | 0,0181 | 0,1638 |
| STTC | 12.380 | 0,8548 | [0,8474; 0,8621] | 0,7867 | 0,7979 (t=0,415) | 0,1681 | 0,0198 | 0,1140 |

‡ Thresholds `max_f1_per_class` ajustados no próprio split de avaliação (protocolo RETROSPECTIVE;
valores otimistas — o refit prospectivo com split de calibração ocorre na T10.3).
Leitura: CD e HYP confirmam-se como gargalo (PR-AUC 0,556/0,508 com ICs estreitos — o déficit é
sistemático, não ruído); o threshold tuning recupera 12–14 p.p. de F1 nessas classes.

## 4. Tabela 3 — Comparabilidade (contrato `ml_protocol_v2.md` §7)

| Par | Mesmo split? | Mesmo evaluator? | Status | Razão |
|---|---|---|---|---|
| A0 hist × A0 novo (v2×v2) | ✅ seed 42 | ✅ v2.0 | **COMPARABLE** | — |
| A0 novo × A2-full (v2×v2) | ❌ seed 42 ≠ 13 | ✅ v2.0 | **NON_COMPARABLE** | `split_id` |
| A0 hist × A2-full (v2×v2) | ❌ seed 42 ≠ 13 | ✅ v2.0 | **NON_COMPARABLE** | `split_id` |
| Legado × v2 (qualquer run) | — | ❌ | **NON_COMPARABLE** | `evaluator_version` |

Nota: NON_COMPARABLE legado×v2 é de **contrato**; numericamente a reconciliação provou
equivalência (Δ=0.0) nas métricas offline compartilhadas (seção 1.1). O que muda é que o v2 é a
única fonte de verdade daqui em diante.

## 5. Tabela 4 — Co-ocorrência NORM × calibração (A2-full, pós-T, n_bins=15)

| Classe | Suporte | P(NORM=1 \| classe=1) | ECE (subconj. NORM=1) | ECE (subconj. NORM=0) |
|---|---:|---:|---:|---:|
| NORM | 33.750 | 1,000 | 0,0611 | **0,2167** |
| CD | 7.350 | 0,586 | 0,0251 | 0,0230 |
| MI | 13.060 | 0,540 | 0,0149 | 0,0373 |
| HYP | 9.930 | 0,578 | 0,0253 | 0,0596 |
| STTC | 12.380 | 0,173 | 0,0505 | **0,1685** |

Leituras: (i) P(NORM|classe) no val confirma a auditoria T10.1 (catálogo: 0,54–0,59) — NORM não é
"ausência de doença"; (ii) NORM e STTC ficam mal calibrados **no estrato NORM=0** — exatamente o
subconjunto clinicamente relevante (registros sem ritmo normal); a T global otimiza a marginal e
deixa o estrato difícil descalibrado ⇒ entrada para a ablação C2 (Platt por classe) e para a
discussão H10 (semântica de NORM).

## 6. Gaps e follow-ups registrados

| # | Gap | Estado |
|---|---|---|
| G1 | Bound inferior do fit de T (0,05) é arbitrário | Documentado; T observadas 0,37–0,97 — longe do limite; alertar se T atingir o bound |
| G2 | Regeneração read-only | **Verificado**: escritas restritas a `evaluation_v2/`; sha256 do checkpoint conferido contra provenance nos 3 runs |
| G3 | Smoke só com sintéticos | **Resolvido nesta task**: reconciliação real Δ=0.0 (A2-full) |
| G4 | IC bootstrap | Macro em `confidence_intervals.json` (artefato); **por classe no nível relatório** (Tabela 2); extensão do avaliador para per-class em artefato = follow-up |
| G5 | Co-ocorrência NORM | **Resolvido nesta task** (Tabela 4) |
| G6 (novo) | `reconcile_with_legacy` só lê chaves top-level (`temperature`, `ece_before/after`); o `calibration.json` legado do A0 novo aninha em `temperature_scaling.*` → campos `null` no reconciliation (workaround: adapter `evaluation_v2/temperature_source.json`) | Follow-up do avaliador: suportar schema aninhado legado |

## 7. Insumos entregues às próximas tasks

- **T9.5 (RFC QG4)**: BCE pós-T exato dos 3 runs (0,3905 / 0,3869 / **0,3417**); evidência de que
  nem ECE 0,015 torna 0,15 atingível; divergência época-17 × checkpoint-28 (auditoria §6.3).
- **T10.2 (matriz de hipóteses)**: ICs por classe (CD/HYP = gargalo sistemático); ECE×NORM
  (Tabela 4); F1@tuned RETROSPECTIVE como teto otimista; NON_COMPARABLE A0×A2 por split.
- **T10.3 (pilotos)**: baseline canônico congelado (este relatório); obrigatoriedade de split
  único + células `A1+BCE`/`A0+focal` para comparabilidade real.

## Fontes

- `experiments/20260728_033533_pretrain_chapman/evaluation_v2/` (fit T; fallback sem provenance —
  suportes validados vs A0 novo: divergência máx. 0,24%)
- `experiments/20260729_042301_pretrain_chapman/evaluation_v2/` (T congelada via adapter)
- `experiments/20260728_053011_pretrain_chapman/evaluation_v2/` (T congelada; reconciliação Δ=0.0)
- IC95 por classe e Tabela 4: computados nesta sessão sobre `predictions.npz` (read-only),
  `n_bootstrap=200`, seed 13, `n_bins=15`.
