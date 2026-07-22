# Especificação de Reconstrução do Project-Lewis — Índice e Estado Final

**Status:** PROPOSTO — A01 autoriza implementação/preflight; ratificação formal pendente
**Data:** 2026-07-18
**Autor:** Kimi Code (sob supervisão do arquiteto)
**Referências:** `reports/forensic_data_quality_report_v1.0.md`,
`artifacts/stage1_recall_investigation/` (R00–R04), `docs/policies/authenticated_research_decision_v1.md`,
`AGENTS.md` (QG0–QG19)

---

## 1. Estado inicial (confirmado)

```text
LEGACY_ARTIFACTS_INVALID_FOR_NEW_TRAINING
REVIEW_REQUIRED
```

Confirmados como fatos verificados: DQ-01 (índices de R-peak em fs nativa aplicados ao sinal
reamostrado — janelas desalinhadas dos rótulos, drift até ~22 min), DQ-02 (features RR em unidade
incompatível, fator fs_nativo/500), DQ-03 (AFDB com zero episódios AFIB em produção), DQ-04
(7 pares de janelas idênticas com labels conflitantes), DQ-05 (política de Q divergente entre
treino e avaliação), DQ-06 (modelo/scaler/threshold de bundles diferentes; modelo promovido com
`passes_qg5=false`), DQ-11 (associação paciente–fold histórica irrecuperável), DQ-14 (modelo
aprendeu identidade do dataset, AUC≈0,55). **Nenhum modelo, scaler, calibrador ou threshold
anterior pode ser reutilizado após a regeneração corrigida dos dados.**

## 2. Escopo desta especificação

Somente especificação, contratos, parâmetros candidatos e critérios de aceite. **Nada aqui
autoriza** correção, regeneração, treinamento, calibração, assinatura, promoção ou implantação.
Cada decisão D1–D7 permanece `PENDING_RATIFICATION` até aprovação humana nomeada e
autenticada. O evento A01 em `12_human_decision_register.md` autoriza implementação, preflight e
pesquisa controlada com dados existentes, mas não promoção nem fabricação de evidência.

Os splits em `data/splits/groupkfold_5_stratified/v3/` agrupam registros, não pacientes, e são
inválidos para treino confirmatório. A política patient-aware está em
`14_patient_identity_and_preflight.md` e usa novo `split_version=3.1.0` sem overwrite.

## 3. Mapa dos entregáveis

| # | Entregável | Arquivo |
| --- | --- | --- |
| 1 | `clinical_ontology_decision` | `01_clinical_ontology_decision.md` |
| 2 | `dataset_disease_mapping` | `01_clinical_ontology_decision.md` (§5) |
| 3 | `temporal_alignment_specification` | `02_temporal_alignment_specification.md` |
| 4 | `mathematical_feature_contract` | `03_mathematical_feature_contract.md` |
| 5 | `multitask_model_specification` | `04_multitask_model_specification.md` |
| 6 | `advanced_training_protocol` | `05_advanced_training_protocol.md` |
| 7 | `domain_shortcut_audit` | `06_domain_shortcut_audit.md` |
| 8 | `calibration_strategy` | `07_calibration_strategy.md` (§1–§4) |
| 9 | `hierarchical_calibration_specification` | `07_calibration_strategy.md` (§5) |
| 10 | `autonomous_calibration_state_machine` | `07_calibration_strategy.md` (§6) |
| 11 | `artifact_bundle_schema` | `08_artifact_bundle_schema.md` |
| 12 | `authenticated_attestation_schema` | `09_authenticated_attestation_schema.md` |
| 13 | `post_training_monitoring_plan` | `10_post_training_monitoring_plan.md` |
| 14 | `promotion_gate_policy` | `11_promotion_gate_policy.md` |
| 15 | `human_decision_register` | `12_human_decision_register.md` |
| 16 | `risk_register` | `13_risk_register.md` |
| 17 | `patient_identity_and_preflight` | `14_patient_identity_and_preflight.md` |

## 4. Ordem de leitura e de dependência das decisões

```text
01 (ontologia: D2–D4)  →  02 (relógio: D1)  →  regeneração v3 (fora de escopo aqui)
   → 03/04/05/06 (features, modelo, treino, anti-shortcut: D5)
   → 07 (calibração: D6, D7)  → 08/09 (bundle + attestation)
   → 10/11 (monitoramento + gates)  → 12/13 (decisões e riscos)
```

## 5. Decisão final desta etapa

```text
REVIEW_REQUIRED
```

Ratificadas as decisões ontológicas (D2–D4) e de relógio (D1), o próximo estado acionável é
`TEMPORAL_PIPELINE_RECONSTRUCTION_REQUIRED`. Nenhum outro estado de ativação
(`DATA_REGENERATION_AUTHORIZED_FOR_PLANNING`, `CONTROLLED_TRAINING_PROTOCOL_READY`,
`CALIBRATION_PROTOCOL_READY`, `AUTHENTICATED_BUNDLE_PROTOCOL_READY`) é declarado nesta etapa:
todos dependem de ratificação humana registrada em `12_human_decision_register.md`.
