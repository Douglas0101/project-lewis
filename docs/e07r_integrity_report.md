# E07R — relatório de integridade (FASE 1)

**Data:** 2026-07-26 | **Status:** `PASS` | **Autoridade:** `AUTONOMOUS_GOVERNANCE_PREAUTH`

## 1. Freeze manifest

- **Artefato:** `experiments/stage2_v2.4_research/integrity/e07r_freeze_manifest.json`
- **Hash do manifest:** `ba1c4aa1b109cb9a11c3facde215624a5674853caf8d19faeb67cf75c0f28171`
- **Pins:** 101 arquivos content-addressed (CUSTODY, IDENTITY, SPLIT, GOVERNANCE, QUARANTINE, SOURCE, LEGACY_SENTINEL).
- Nota: uma primeira publicação foi substituída após correção de contrato de dados no loader full-template (`src/stage2_research/data.py`, layout `(n,500,1)` do r4); a re-publicação é o estado válido e o preflight `e065pd-audit-v1` preserva a evidência do estado anterior.

## 2. Hashes herdados validados

| Artefato | SHA-256 | Estado |
| --- | --- | --- |
| `data/features/stage2_multiclass.npz` | `68fb0a8e…b04a` | íntegro |
| `data/features/stage2_multiclass.parquet` | `870b386e…3802` | íntegro |
| `data/features/finetuning_mitbih_family.npz` | `6b38eaf8…a54a` | íntegro (ver incidente §5) |
| `data/features/finetuning_mitbih_family.parquet` | `9efe6681…0c26` | íntegro |
| `experiments/stage2_v2.4_research/E07_blocked_20260726.json` | `d8a684c0…49f63` | íntegro |

## 3. Proteção implementada

- **Permissões:** pins E07R novos em `0444` (GOVERNANCE/CUSTODY/IDENTITY/SPLIT/QUARANTINE); pins LEGACY_SENTINEL (`models/`, `backup_v2.3/`, quarentena v3.1, legados) protegidos por hash sem chmod (bytes históricos intocados).
- **Proteção lógica:** `guard_e07r_write` rejeita escrita em paths congelados com evento JSON de violação; `assert_authorized_split_path` bloqueia uso de splits legados; preflight fail-closed antes de cada workflow e antes de `DONE`.
- **chattr +i:** não disponível sem privilégio adicional — registrado como `NOT_AVAILABLE`; a proteção lógica cobre o gap.

## 4. Preflight fail-closed

`run_e07r_preflight(FREEZE_VALIDATION)` — **9/9 PASS** (inicial e pós-missão, report `dbaf5f73…`): FREEZE_INVENTORY_LINKS, FREEZE_PINS, PREAUTHORIZATION, CUSTODY_BINDING, PATIENT_MAPPING, SPLIT_BUNDLE, LEGACY_QUARANTINE, PD_PROTOCOL, EVIDENCE_COMPLETION.

## 5. Incidente de drift (resolvido)

Em 2026-07-26, `data/features/finetuning_mitbih_family.npz` apresentou CRC-32 quebrado e hash divergente do pin (`cbbd8550…` vs `6b38eaf8…`), com inode/mtime de 25/jul — corrupção in-place fora da cadeia de ferramentas. O preflight bloqueou (fail-closed correto). Restauração byte-a-byte a partir de `data/features/backup_v2.3/` (hash idêntico) e varredura CRC de todos os npz críticos: íntegros. Causa externa não determinada; recomendado monitoramento de disco.

## 6. Testes de integridade

- `tests/test_e07r_integrity_v4.py`, `test_stage2_custody_v4.py`, `test_stage2_patient_disjoint_v4.py`, `test_stage2_pd_workflows_v4.py`, `test_stage2_full_template_loader_v4.py`, `test_e07r_status_cli.py`, `test_e07r_watch_state.py`, `test_e07r_watch_cli.py` — cobrem freeze, custódia, mapping, splits, quarentena, write-violation, loaders e CLIs de status/watch. Estado: verdes.
