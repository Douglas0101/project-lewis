# Fase 0 — Congelamento do estado atual (PRD+SDD CPU-First §19)

**Data:** 2026-08-02 · **Branch:** `develop` · **Ref:** `docs/PRD + SDD — Otimização CPU-First
do Project Lewis.md` (Fase 0 do plano de migração)

> Registro de congelamento dos 4 pilotos da Fase 1 (controle H7), executados manualmente pelo
> owner via `make pilot-c{0..3}` em 2026-08-02, **antes** da adequação da esteira (Fase 1/P0).
> Números verificados contra `pilot_status.json` e `evaluation_v2/metrics.json` de cada run.

## 1. Pilotos congelados (hashes SHA-256 dos checkpoints, prefixo 16)

| Célula | Arch × Loss | Run dir | checkpoint sha256 (16) | macro PR-AUC | macro AUROC |
|---|---|---|---|---:|---:|
| C0 | A0 + BCE | `experiments/20260802_035117_pretrain_chapman` | `581d8e042ecbcb91` | 0,6458 | 0,8241 |
| C1 | A1 + BCE | `experiments/20260802_043748_pretrain_chapman` | `74f71b6c524bfc27` | **0,6749** | 0,8502 |
| C2 | A1 + focal γ=2 | `experiments/20260802_051138_pretrain_chapman` | `fed052189682c6c7` | 0,6748 | **0,8521** |
| C3 | A0 + focal γ=2 | `experiments/20260802_054320_pretrain_chapman` | `c069ffac9ca4ab5d` | 0,6507 | 0,8280 |

Calibração (fit na partição calibration, aplicada à partição de comparação):

| Célula | ECE pós-T | ECE NORM=0 | T | BCE pós-T |
|---|---:|---:|---:|---:|
| C0 | 0,0159 | 0,1618 | 0,9394 | 0,3932 |
| C1 | 0,0187 | **0,1122** | 0,9150 | 0,3540 |
| C2 | 0,0217 | 0,1165 | **0,3399** | **0,3523** |
| C3 | **0,0149** | 0,1579 | 0,3439 | 0,3892 |

## 2. Leitura H7 (atribuição do ganho)

- **Arquitetura:** C1 − C0 = **+2,91 p.p.** macro PR-AUC (A1 > A0 sob BCE, mesmo split/seed).
- **Loss (focal):** C2 − C1 = **−0,01 p.p.** (neutro em PR-AUC; C2 ganha +0,19 p.p. AUROC mas
  piora ECE e exige T=0,34); C3 − C0 = +0,49 p.p. (focal no A0, marginal).
- **Conclusão:** o ganho histórico do A2-full vem da **arquitetura A1**; o focal é neutro em
  ranking e caro em calibração (underconfidence T≈0,34, mesma assinatura do A2-full original).
  A decisão do PRD — **A1 + BCE como baseline qualificado** — é coerente com os dados.
- Observação: C2 tem o menor BCE pós-T (0,3523) — o focal continua ganhando em NLL calibrado,
  não em ranking. Entrada para a RFC T9.5.

## 3. Classificação da partição `test` (declaração de governança)

Os 4 pilotos compararam células usando a partição `test` do split pareado v2 (com T/thresholds
fit na `calibration` — fluxo prospectivo, mas com o teste **consultado no fluxo de comparação**,
o que o PRD proíbe: RF-DATA-005/RF-SEL-001). Portanto:

```text
A partição `test` de chapman-record-disjoint-paired-v2 está classificada como
DEVELOPMENT-CONSULTED a partir de 2026-08-02.

Consequências:
1. Os números C0–C3 acima são LEITURAS DE DESENVOLVIMENTO — não evidência final.
2. A comparação entre células passa a ocorrer na partição `validation`
   (adequação Fase 1, P0-03 — implementada nesta sessão).
3. A qualificação oficial (Fase 3, P1-02) exigirá um NOVO teste bloqueado
   (novo split ou partição re-selada com freeze prévio — decisão de governança).
4. O manifesto do split pareado v2 permanece write-once e imutável; esta
   classificação vive neste relatório e em GOVERNANCE_NOTE.md ao lado do manifesto.
```

## 4. Estado da esteira no congelamento

- Problemas P0 confirmados e endereçados na adequação (Fase 1, mesma sessão):
  esgotamento época 6 (falta de `.repeat()` no builder pareado — corrigido); teste consultado
  na comparação (corrigido — comparação movida para `validation`); QG4 sem código de saída
  (gates de célula com exit 3); oneDNN inconsistente (perfis `configs/runtime/` + política
  CPU-only); IDs de registro/segmento ausentes nas predições (adicionados); CUDA init
  (eliminado via `CUDA_VISIBLE_DEVICES=-1`).
- Pendências (Fases 2–5 do PRD): inventário/exclusões, cache canônico/shards, multiseed
  [13,29,47,71,101], bootstrap agrupado por registro, thresholds/temperatura cross-fitted,
  benchmarks de threads/batch/steps_per_execution, LiteRT/OpenVINO, release imutável.

## Fontes

`experiments/20260802_0{35117,43748,51138,54320}_pretrain_chapman/pilot_status.json` ·
`evaluation_v2/metrics.json` · sha256 calculado nesta sessão sobre
`backbone_pretrained.keras` de cada run.
