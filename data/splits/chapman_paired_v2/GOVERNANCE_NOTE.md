# Nota de governança — split pareado v2 (2026-08-02)

A partição **`test`** de `chapman-record-disjoint-paired-v2` está classificada como
**DEVELOPMENT-CONSULTED**: ela foi consultada no fluxo de comparação dos pilotos C0–C3
(2026-08-02), antes da adequação Fase 1 do PRD+SDD CPU-First.

- Comparações entre células passam a ocorrer na partição **`validation`** (implementado em
  `scripts/run_pilot_cell.py` — T/thresholds fit na `calibration`, aplicados à validation).
- A partição `test` permanece **bloqueada** para exportação até a existência de
  `model_freeze.json` na run (`src/governance/freeze_manager.py`, RF-DATA-005).
- A qualificação oficial exigirá um **novo teste bloqueado** (Fase 3 / P1-02 do PRD).

Este manifesto permanece write-once e imutável. Detalhes:
`reports/fase0_freeze/fase0_freeze_report.md`.
