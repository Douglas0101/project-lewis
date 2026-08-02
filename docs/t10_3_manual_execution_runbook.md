# Runbook — Execução manual dos pilotos T10.3 (Fase 1: C0–C3)

**Versão:** v2.0.0 (pós-adequação PRD CPU-First) · **Data:** 2026-08-02 · **Branch:** `develop`
**Escopo:** execução MANUAL via Makefile da Fase 1 do plano de pilotos
(`docs/t10_3_pilot_execution_plan.md`). **Nenhum piloto é promovido; QG4-BCE permanece FAIL.**

> **Mudanças da adequação (PRD Fase 1/P0):** comparação de células agora ocorre na partição
> **`validation`** (teste **bloqueado** até `model_freeze.json`); esgotamento de dataset é
> falha; CUDA desabilitado; perfil numérico uniforme (`configs/runtime/fast.yaml`); predições
> com `record_id`/`segment_id`; gates com **exit code 3**. Os 4 pilotos de 2026-08-02 foram
> congelados como leituras de desenvolvimento — ver `reports/fase0_freeze/fase0_freeze_report.md`.

---

## 1. Pré-requisitos (uma única vez)

```bash
make lint && make e07r-check     # devem estar verdes (9/9)
make pilot-split                 # split pareado v2 (idempotente; FORCE=1 só com governança)
make pilot-smoke                 # validação end-to-end (~3 min; prova de engenharia)
```

Verifique ao final do smoke: `protocol=PROSPECTIVE`, `evaluation_split: validation` e
`test_status: locked_until_model_freeze` no log/`pilot_status.json`.

## 2. Execução da Fase 1 (ordem obrigatória)

| Ordem | Comando | Duração ~ (CPU) | Gate (exit 3 se reprovar) |
|---:|---|---|---|
| 1 | `make pilot-c0` | ~35 min | — (baseline) |
| 2 | `make pilot-c1` | ~40 min | C1 macro PR-AUC ≥ C0 (banda 0,5 p.p.) |
| 3 | `make pilot-c2` | ~40 min | C2 ≥ C1 e ≥ 0,60 (sanidade) |
| 4 | `make pilot-c3` | ~35 min | — (leitura arch×loss) |

**Regra de parada mestra:** gate de C1 reprovado (exit 3) ⇒ **pare a rodada**, registre em
`reports/t10_3_pilot_failures.md` — a matriz deve ser revisada antes da Fase 3.

**Leituras de desenvolvimento já congeladas** (pilotos 20260802_*, pré-adequação): C1 A1+BCE
0,6749 · C2 A1+focal 0,6748 · C0 0,6458 · C3 0,6507 — a leitura H7 (ganho = arquitetura;
focal neutro) já está registrada; a re-execução sob a esteira adequada serve para confirmar
reprodutibilidade do fluxo, não para reabrir a conclusão.

Cada célula produz, em `experiments/<ts>_pretrain_chapman/`:

```text
backbone_pretrained.keras   # melhor checkpoint (ES por val_auc_pr)
provenance.json             # inclui split_id + early_stopping_metric
evaluation_v2/              # 7 artefatos schema 2.0 + predictions/{validation,calibration}.npz
                            #   (com record_ids/segment_ids/patient_ids)
pilot_status.json           # {"status": "PILOT", "evaluation_split": "validation",
                            #   "test_status": "locked_until_model_freeze", "gate": {...}}
```

## 3. O que conferir ao final de cada célula

| Campo | Referência dev (C1 congelado) | Observação |
|---|---|---|
| `macro_pr_auc` | 0,6749 | primária — gates |
| `per_class › CD › auc_pr` | ~0,55 | classe-sentinela |
| `ece_post_calibration` | 0,0187 | global, n_bins=15 |
| `ece_post_calibration_norm0` | 0,1122 | estrato patológico |
| `protocol_status` | — | **PROSPECTIVE** (fit na calibration, apply na validation) |
| exit code | — | 0 ok · 1 execução · 2 config · **3 gate reprovado** · 4 leakage |

## 4. Isolamento do teste (RF-DATA-005)

- `scripts/export_pilot_predictions.py --partition test` **recusa** sem `model_freeze.json`
  (exit 4).
- Freeze de seleção (quando governança autorizar): `model_freeze.json` via
  `src/governance/freeze_manager.py` — aí, e só aí, o teste pode ser exportado/avaliado,
  **uma única vez**, para o relatório final.
- A partição `test` atual é DEVELOPMENT-CONSULTED (ver `GOVERNANCE_NOTE.md` no split) — a
  qualificação oficial usará **novo teste bloqueado** (Fase 3, P1-02).

## 5. Rollback / falhas

```text
1. NÃO prosseguir a fila.
2. Marcar a run como REJECTED em experiments/<run>/pilot_status.json.
3. Registrar em reports/t10_3_pilot_failures.md (célula, sintoma, hipótese).
4. Não modificar models/, splits ou artefatos congelados.
```

## 6. Fora de escopo nesta esteira

- **D0 (self-distillation)** e trilha D: dependem do trainer KD — task separada.
- Fases 3 (F/K/T/D1–D2) e trilhas R/B/P: só após relatório da Fase 1 + D0.
- Promoção de qualquer piloto: bloqueada (revisão humana + RFC T9.5).
- Multiseed [13,29,47,71,101], bootstrap agrupado, shards: Fases 2–3 do PRD (pendentes).

## 7. Lembretes de governança

- Todo piloto nasce `PILOT` — nunca `CANDIDATE` sem os 10+12+5 critérios (T10.3-P §5).
- `make pilot-split FORCE=1` regenera o split e **invalida a comparabilidade** de todas as
  células anteriores — usar somente com decisão de governança registrada.
