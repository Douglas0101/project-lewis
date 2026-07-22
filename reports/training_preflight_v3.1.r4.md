# `training_preflight_v3.1`

**Gerado em:** 2026-07-22T03:48:49.343777+00:00  
**Geração:** `advanced-training-v3.1.0-r4`  
**Estado:** `REVIEW_REQUIRED`  
**Treinamento permitido:** `false`

## Checks

| Código | Status | Categoria | Evidência | Denominador | Limitação |
| --- | --- | --- | --- | --- | --- |
| SESSION_AUTHORIZATION_RECORDED | WARN | OBSERVED | project owner authorized continued research with existing project data | 1 interactive authorization statement | approver name and authenticated identity were not supplied; promotion remains blocked |
| CONFIRMATORY_PATIENT_IDENTITY_VALIDATED | PASS | DERIVED_MATHEMATICALLY | INCART header groups and MITDB documented subject groups are complete | 123 records / 79 patient groups | SVDB and AFDB are excluded from the confirmatory patient-wise core |
| UNVERIFIED_DATASETS_QUARANTINED | WARN | DERIVED_MATHEMATICALLY | unresolved record-to-patient mappings are assigned non-confirmatory roles | 101 records | record-clustered patient metrics are not supported for quarantined datasets |
| LEGACY_SPLIT_PATIENT_LEAKAGE_DETECTED | WARN | DERIVED_MATHEMATICALLY | 29 known patients cross legacy outer folds | 79 verified patient groups checked | legacy v3 splits and matrix remain invalid and are never repaired in place |
| PATIENT_SPLIT_V3_1_VALIDATED | PASS | DERIVED_MATHEMATICALLY | each confirmatory patient and record is assigned to exactly one outer test fold | 79 patients across 5 folds | inner/calibration/threshold partitions are not generated in this task |
| FAMILY_SOURCE_CUSTODY_COMPLETE | PASS | OBSERVED | parent family artifact exposes complete source-bound row custody | 15 parquet columns and 4 NPZ members | values and source reconstruction are checked separately |
| FAMILY_ORDERED_ROW_BINDING_VALIDATED | PASS | OBSERVED | ordered sample IDs, labels, and recomputed waveform digests match | 469723 rows | validation is bound to canonical float32 waveform bytes |
| FAMILY_SOURCE_RECONSTRUCTION_VALIDATED | PASS | OBSERVED | every parent beat row and waveform reconstructs from bound source evidence | 469723 rows | processed-signal generation remains a separately hashed preprocessing claim |
| SAMPLE_LINEAGE_COMPLETE | PASS | OBSERVED | all required sample custody columns are present | 18 required columns | schema presence alone does not prove row values |
| SAMPLE_LINEAGE_VALUES_VALIDATED | PASS | OBSERVED | all sample identity, clock, ontology, quality, and split values satisfy the contract | 461600 rows | validation does not establish external clinical validity |
| ORDERED_ROW_BINDING_VALIDATED | PASS | OBSERVED | ordered sample IDs, labels, and recomputed waveform digests match | 461600 rows | validation is bound to canonical float32 waveform bytes |
| STAGE1_PARENT_BINDING_VALIDATED | PASS | OBSERVED | Stage 1 exactly matches the deterministic parent population transform | 461600 rows | the parent source reconstruction is a separate required check |
| AFDB_LINEAGE_COMPLETE | PASS | OBSERVED | all required AFDB episode custody columns are present | 23 required columns | schema presence alone does not prove row values |
| AFDB_LINEAGE_VALUES_VALIDATED | PASS | OBSERVED | all AFDB episode source, clock, ontology, and quarantine values satisfy the contract | 83750 rows | AFDB patient identity remains a separate evidence requirement |
| AFDB_ORDERED_ROW_BINDING_VALIDATED | PASS | OBSERVED | ordered sample IDs, labels, and recomputed waveform digests match | 83750 rows | validation is bound to canonical float32 waveform bytes |
| AFDB_SOURCE_RECONSTRUCTION_VALIDATED | PASS | OBSERVED | every eligible AFDB episode and waveform reconstructs from bound source evidence | 83750 episodes | AFDB patient identity remains a separate requirement |
| INPUT_SNAPSHOT_STABLE | PASS | DERIVED_MATHEMATICALLY | all consumed input bytes remained stable during preflight | 8 input artifacts | filesystem metadata detects concurrent change but not malicious timestamp forgery |
| AFDB_PATIENT_IDENTITY_UNVERIFIED | BLOCK | NOT_SUPPORTED | AFDB rhythm records lack authenticated patient mappings | 23 AFDB records | family D rhythm output remains exploratory until identity evidence is resolved |
| EXACT_INPUT_BYTES_BOUND | PASS | DERIVED_MATHEMATICALLY | raw, annotation, processed, ontology, preprocessing, config, source, and environment bytes are content-addressed | 10 required generation identities | hashing does not repair missing row-level custody |
| EXTERNAL_VALIDATION_REQUIRED | WARN | NOT_SUPPORTED | no genuinely untouched external validation source is authenticated | 0 external sources | internal transport analysis cannot be renamed external validation |

## Identidade da geração

```text
raw_data_hash=696e2c7878c40c1a1c230a228680baba46a430c1bef8c83cb6c7e6d5803e2c51
annotation_hash=5e4d624d5d82b61d564d0c4e7acf499714d833733f94a10dae26ebaed3788d8d
processed_data_hash=bdb9ea7c356f73b31514c1e6d5e92c5c56780e0c52a9b1ee84c4cd38bfdcdeda
ontology_hash=051f4d8b476e4dde2e32b125b302b18b091e4d582d97fc217a50b1d5ab0f78a1
preprocessing_hash=e263cca99214d6579317fdad4aef0ebc5b70414be4e8b1a49c6cfce7dd748644
feature_schema_hash=52495044777f92c274f5c1e2f5cebe16fff38d1a2f1af7ca988ea955ea6492d5
patient_split_hash=3a15bf1e591f32e29010796d3c3a282795f4ba384da3d29ccce5c5630e9156e3
training_config_hash=3438a386179fbd54f06adcc10c3ba852a04bd33e686fde161bc08bb827950be8
source_revision=1f003a3d7e678451669d70a9782ef9775ffade4bad0833b380047a682a6a9ccd
environment_hash=b9ca82a645075b08eb4456801730aac1b604cb3b647a1cc854f6be4b3f2eb529
```

## Decisão

```text
TRAINING_BLOCKED_BY_DATA_PROVENANCE
REVIEW_REQUIRED
```

Nenhuma célula de treinamento pode iniciar.
