# E07R — relatório de regeneração dos splits patient-disjoint (FASE 3)

**Data:** 2026-07-26 | **Versão:** `v4.0-patient-disjoint` | **Status:** `PASS`

## 1. Entradas congeladas

- Stage 2 autorizado: `data/features/v3.1.0-r5-stage2-pd/` (33.001 beats S/V/F; custódia NPZ↔Parquet ordenada; parent r4 `d8ce5061…` byte-verificado).
- Mapping: `data/metadata/stage2_patient_identity_v4.0.json` — 119 records confirmatórios → 76 pacientes (MIT-BIH 48→47, com 201/202 unificados; INCART 74→29 grupos autenticados). SVDB: `IDENTITY_UNVERIFIED`, fora do fitting confirmatório.

## 2. Geração

- Outer: `StratifiedGroupKFold` 5 folds, `shuffle=true`, seed 42, grupo=`patient_id`.
- Inner: por outer train, candidatos seeds 43–47 com a política `select_inner_split` legada; primeiro split determinístico publicado + 4 candidatos para auditoria.
- Disjointidade priorizada sobre balanceamento; zeros estruturais reportados, não mascarados.

## 3. Estatísticas por fold (test)

| Fold | Pacientes | Records | Amostras | F | S | V | Pacientes F |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 16 | 26 | 6.557 | 374 | 749 | 5.434 | 3 |
| 2 | 12 | 18 | 6.963 | 373 | 1.156 | 5.434 | 2 |
| 3 | 15 | 25 | 6.281 | 86 | 714 | 5.481 | 9 |
| 4 | 15 | 22 | 6.275 | 93 | 721 | 5.461 | 9 |
| 5 | 18 | 28 | 6.925 | 95 | 1.399 | 5.431 | 7 |

Observação estrutural: a massa de F concentra-se em poucos pacientes nos folds 1–2 (escopo 208/213) e se espalha em 3–5 — causa dominante da dificuldade crescente por fold. Nenhum zero estrutural; nenhum SVDB em teste.

## 4. Verificações de leakage (outer + inner, 5+20)

- interseções train/val/test de patient IDs: **vazias em todos os folds**;
- interseções de record IDs: vazias;
- cada patient ID em uma única partição; todos os records de um paciente juntos;
- **201/202 sempre juntos** (`known_group_201_202_respected=true`);
- cobertura/cardinalidade = dataset congelado; índices únicos e em range.

Evidência: `experiments/stage2_v2.4_research/integrity/e07r_split_leakage_report.json` — `patient_disjoint=true`, `record_disjoint=true`, `status=PASS`.

## 5. Quarentena dos splits legados

`experiments/stage2_v2.4_research/quarantine/splits_record_disjoint_leakage_era_v2.3/quarantine_manifest.json`: paths, tamanhos e hashes preservados; `status=QUARANTINED_NOT_DELETED`, `reason=PATIENT_LEAKAGE_RECORD_DISJOINT_ONLY`, `active_for_e07r=false`. Preflight falha se qualquer workflow apontar para eles.
