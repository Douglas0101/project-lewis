# Runbook — Execução manual dos pilotos T10.3 (Fase 1: C0–C3)

**Versão:** v1.0.0 · **Data:** 2026-08-02 · **Branch:** `develop`
**Escopo:** execução MANUAL via Makefile da Fase 1 do plano de pilotos
(`docs/t10_3_pilot_execution_plan.md`). **Nenhum piloto é promovido; QG4-BCE permanece FAIL.**

---

## 1. Pré-requisitos (uma única vez)

```bash
make lint && make e07r-check     # devem estar verdes (9/9)
make pilot-split                 # gera data/splits/chapman_paired_v2/manifest.json
                                 # (idempotente; FORCE=1 regenera — exige governança)
make pilot-smoke                 # validação end-to-end (~3 min; métricas sem sentido
                                 # clínico — é prova de engenharia, não resultado)
```

Verifique ao final do smoke: `protocol=PROSPECTIVE` e `status: PILOT` no log.

## 2. Execução da Fase 1 (ordem obrigatória)

| Ordem | Comando | Duração ~ (CPU) | Gate para prosseguir (T10.3-P §2) |
|---:|---|---|---|
| 1 | `make pilot-c0` | ~35 min | — (baseline) |
| 2 | `make pilot-c1` | ~40 min | C1 macro PR-AUC ≥ C0 |
| 3 | `make pilot-c2` | ~40 min | C2 macro PR-AUC ≥ C1 e ≈ 0,70 (sanidade A2-full) |
| 4 | `make pilot-c3` | ~35 min | — (leitura arch×loss) |

**Regra de parada mestra:** se C1 não superar C0 (Δ macro PR-AUC < 1 p.p. com IC95 sobreposto),
H7 está comprometida — **pare a rodada** e registre em `reports/t10_3_pilot_failures.md`;
a matriz deve ser revisada antes de qualquer Fase 3.

Cada célula produz, em `experiments/<ts>_pretrain_chapman/`:

```text
backbone_pretrained.keras   # melhor checkpoint (ES por val_auc_pr — protocolo v2)
provenance.json             # inclui split_id=chapman-record-disjoint-paired-v2
evaluation_v2/              # 7 artefatos schema 2.0 + predictions/{validation,calibration,test}.npz
pilot_status.json           # {"status": "PILOT", "cell": "cN", ...}
```

## 3. O que conferir ao final de cada célula

No log final do `run_pilot_cell.py` (e em `evaluation_v2/metrics.json`):

| Campo | Referência (A2-full, T9.3) | Observação |
|---|---|---|
| `macro_pr_auc` | 0,7065 | primária — gates da Fase 1 |
| `per_class › CD › auc_pr` | 0,5561 | classe-sentinela (H3/H8) |
| `ece_post_calibration` | 0,0152 | global, n_bins=15 |
| `ece_post_calibration_norm0` | 0,2167 | estrato patológico (H2) |
| `protocol_status` | — | **deve ser PROSPECTIVE** (T/thresholds fit na calibration) |

Leitura arch×loss (H7): C1≈C2 com IC sobreposto ⇒ ganho do A2-full é **arquitetura**;
C1≈C0 e C3≈C2 ⇒ ganho é **loss (focal)**; intermediário ⇒ efeito misto.

## 4. Rollback / falhas

Se qualquer célula falhar ou divergir:

```text
1. NÃO prosseguir a fila.
2. Marcar a run como REJECTED em experiments/<run>/pilot_status.json.
3. Registrar em reports/t10_3_pilot_failures.md (célula, sintoma, hipótese).
4. Não modificar models/, splits ou artefatos congelados.
```

## 5. Fora de escopo nesta esteira

- **D0 (self-distillation)** e trilha D: dependem do trainer KD — task de implementação
  separada (`configs/ml_protocol/v2/distillation_kd.yaml` já normatiza o protocolo).
- Fases 3 (F/K/T/D1–D2) e trilhas R/B/P: só após relatório da Fase 1 + D0.
- Promoção de qualquer piloto: bloqueada (revisão humana + RFC T9.5).

## 6. Lembretes de governança

- Todo piloto nasce `PILOT` — nunca `CANDIDATE` sem os 10+12+5 critérios (T10.3-P §5).
- Avaliação canônica já roda PROSPECTIVE ao final de cada célula (T/thresholds fit na
  partição `calibration`, aplicados ao `test` congelado).
- `make pilot-split FORCE=1` regenera o split e **invalida a comparabilidade** de todas as
  células anteriores — usar somente com decisão de governança registrada.
