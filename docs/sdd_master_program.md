# SDD Mestre — Programa de Correção Estrutural Project-Lewis

Data de abertura: 2026-07-29 | Branch de trabalho: `feat/pretrain-architecture-v2`
Manifesto de governança: `experiments/governance/sdd_master_manifest.json`

## Escopo executado (mapeamento FASE → evidência)

| FASE | Tema | Documento de detalhe |
|---|---|---|
| 0 | Governança e snapshot | este arquivo + `experiments/governance/sdd_master_manifest.json` |
| 1 | Integridade e pins | `tests/test_integrity.py` (novo) + chmod 0444 nos pins |
| 2 | Mapeamento paciente e splits | `docs/e07R_split_regeneration_report.md` |
| 3 | Correção do pré-treino | `docs/pretrain_engineering_fix.md` (FASES 0–4 do programa anterior) |
| 4 | Exit code e wrapper | este arquivo §FASE 4 + `scripts/pretrain_wrapper.py` |
| 5 | Determinismo e proveniência | `src/models/pretrain_provenance.py` + env estrito no wrapper |
| 6 | Baseline A0 e arquitetura | `docs/pretrain_architecture_baseline_A0.md`, `docs/pretrain_architecture_v2.md` |
| 7 | Avaliação avançada | `src/models/pretrain_evaluation.py` + relatórios por run |
| 8 | Makefile UX | `docs/makefile_refactor_plan.md` |
| 9 | Testes e validação | suíte completa (`make test`) |
| 10 | Relatórios e checkpoint | `experiments/governance/sdd_master_checkpoint_20260729.json` |

## Estado herdado (2026-07-28/29)

- E07R-PD concluído (2026-07-26): splits `v4.0-patient-disjoint` congelados, leakage
  report PASS, 101 pins com 0 mismatch de hash.
- Pipeline de pré-treino corrigido (2026-07-28): sem warning `ran out of data`,
  wrapper com validação de artefatos, proveniência por run, A0 congelada,
  variantes A1/A2 implementadas e orçadas, experimentos E0–E3 documentados.
- QG4: FAIL honesto (braço AUC passou no run A2-full; braço BCE não). Thresholds
  inalterados.

## FASE 4 — Política de exit code (normativa deste SDD)

| Comando | execution_success + qg4 PASS | execution_success + qg4 FAIL | falha real | erro de config/uso |
|---|---|---|---|---|
| `python -m src.models.pretrain_chapman` | 0 | 0 | 1 | 2 |
| `scripts/pretrain_wrapper.py` | 0 | 0 | 1 | 2 |
| `scripts/pretrain_wrapper.py --enforce-qg4` | 0 | **10** | 1 | 2 |
| `make pretrain` | 0 | 0 | 1 | 2 |
| `make pretrain-qg` | 0 | **10** | 1 | 2 |

- `run_status.json` e `qg4_result.json` são gravados por run (contratos 10.4/10.6).
- QG4 fail é **resultado científico**, não falha de processo (PRN-008, DEF-008/DEF-010).

## Regras ativas

- Publicação: **HOLD** | Promoção para `models/`: bloqueada (opt-in `--promote`, não usado)
- Gates: inalterados | Artefatos congelados: preservados | Leakage: zero tolerância
