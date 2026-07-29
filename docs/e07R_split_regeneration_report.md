# E07R — Regeneração de splits patient-disjoint (FASE 2 do SDD Mestre)

Data de verificação: 2026-07-29 | Status: **VERIFICADO — PASS**
Execução original: E07R-PD, 2026-07-26 (concluído antes deste SDD; esta fase
**verificou e registrou**, sem regerar artefatos congelados — NAU-004).

## 1. Mapeamento record_id → patient_id (RF-SPLIT-001)

`data/metadata/physionet_mitdb_patient_mapping.json` (pinada no freeze E07R):

- Política: `official_evidence_required` — record_id = patient_id, salvo evidência oficial.
- Grupo OFFICIAL: `mitdb:subject:201_202` → record_ids `["201", "202"]`.
- Fonte: PhysioNet MIT-BIH intro (`docs/physionet_mitdb_patient_statement.md` cacheado),
  `access_date: 2026-07-26`, `mapping_hash` registrado.

## 2. Splits v4.0 (RF-SPLIT-002) — verificação independente executada agora

`data/splits/stage2_multiclass_patient_disjoint_v4.0/` (congelados):

| Checagem | Resultado |
|---|---|
| Outer folds (5) — 201/202 sempre no **mesmo lado** | ✅ (fold 4: ambos teste; demais: ambos treino) |
| Outer — `patient_overlap` / `record_overlap` | ✅ vazios em todos os folds |
| Inner folds (20) — overlaps paciente/registro (train↔val, val↔outer-test, train↔outer-test) | ✅ zero violações |
| `leakage_checks.json` / `e07r_split_leakage_report.json` | ✅ `status: PASS`, `patient_disjoint: true`, `known_group_201_202_respected: true`, sem `low_support`/`structural_zero` |
| Hashes (freeze manifest, 101 pins) | ✅ 0 mismatches (`tests/test_integrity.py`) |

Nenhum paciente/classe removido para melhorar métricas (NAU-006): o relatório de
leakage registra `low_support_folds: []` e `structural_zero_folds: []`.

## 3. Quarentena dos splits antigos (RF-SPLIT-003)

- `data/splits/groupkfold_5_stratified/QUARANTINED.json` criado (marcador aditivo,
  conteúdo preservado para auditoria; uso proibido em novos treinamentos).
- Diretórios de quarentena ativos: `experiments/stage2_v2.4_research/quarantine`,
  `data/features/quarantine_v31_working_20260718`; backup: `data/features/backup_v2.3`.

## 4. Consequência experimental (herdada do E07R)

- E06.5-PD: 100/100 células — H6 **não** supera baseline (ΔF1(F)=−0,1601) → `NO_VALID_CANDIDATE`.
- E07-PD: **não executado (0/150)** por pré-registro.
- Publicação: **HOLD**. Models/: congelados por hash (35 pins), intactos.

## 5. Conclusão

DEF-001 (leakage por paciente) está **remediado e verificado**: splits atuais são
patient-disjoint com evidência oficial para 201/202, leakage report PASS e
integridade congelada por hash. Nenhuma ação adicional requerida nesta fase além
do registro — nenhum artefato congelado foi regenerado ou sobrescrito.
