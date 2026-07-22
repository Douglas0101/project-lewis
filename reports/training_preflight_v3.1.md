# `training_preflight_v3.1`

**Gerado em:** 2026-07-21T14:27:00.798190+00:00  
**Geração:** `advanced-training-v3.1.0-r1`  
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
| FAMILY_SOURCE_CUSTODY_INCOMPLETE | BLOCK | OBSERVED | parent family artifact lacks source-bound row custody | 15 parquet columns and 4 NPZ members | derived row hashes without parent custody are insufficient |
| FAMILY_ORDERED_ROW_BINDING_INCOMPLETE | BLOCK | OBSERVED | NPZ lacks ordered sample identity or waveform digests | 4 required NPZ members | same-class waveform permutations cannot be detected |
| FAMILY_SOURCE_RECONSTRUCTION_NOT_EVALUATED | BLOCK | NOT_SUPPORTED | parent source reconstruction prerequisites did not pass | 1 parent NPZ/parquet pair | source population and waveform custody remain unproven |
| SAMPLE_LINEAGE_INCOMPLETE | BLOCK | OBSERVED | 16 required custody columns are absent | 18 required columns | record-level lineage cannot replace sample-level dual-clock custody |
| ORDERED_ROW_BINDING_INCOMPLETE | BLOCK | OBSERVED | NPZ lacks ordered sample identity or waveform digests | 4 required NPZ members | same-class waveform permutations cannot be detected |
| STAGE1_PARENT_BINDING_NOT_EVALUATED | BLOCK | NOT_SUPPORTED | Stage 1 parent-binding prerequisites did not pass | 1 parent/child artifact pair | Stage 1 population completeness remains unproven |
| AFDB_LINEAGE_INCOMPLETE | BLOCK | OBSERVED | 21 required AFDB custody columns are absent | 23 required columns | episode hashes alone do not prove source-record or clock custody |
| AFDB_ORDERED_ROW_BINDING_INCOMPLETE | BLOCK | OBSERVED | NPZ lacks ordered sample identity or waveform digests | 4 required NPZ members | same-class waveform permutations cannot be detected |
| AFDB_SOURCE_RECONSTRUCTION_NOT_EVALUATED | BLOCK | NOT_SUPPORTED | AFDB source-reconstruction prerequisites did not pass | 1 AFDB NPZ/parquet pair | episode population and source waveform custody remain unproven |
| INPUT_SNAPSHOT_STABLE | PASS | DERIVED_MATHEMATICALLY | all consumed input bytes remained stable during preflight | 8 input artifacts | filesystem metadata detects concurrent change but not malicious timestamp forgery |
| AFDB_PATIENT_IDENTITY_UNVERIFIED | BLOCK | NOT_SUPPORTED | AFDB rhythm records lack authenticated patient mappings | 23 AFDB records | family D rhythm output remains exploratory until identity evidence is resolved |
| EXACT_INPUT_BYTES_BOUND | PASS | DERIVED_MATHEMATICALLY | raw, annotation, processed, ontology, preprocessing, config, source, and environment bytes are content-addressed | 10 required generation identities | hashing does not repair missing row-level custody |
| EXTERNAL_VALIDATION_REQUIRED | WARN | NOT_SUPPORTED | no genuinely untouched external validation source is authenticated | 0 external sources | internal transport analysis cannot be renamed external validation |

## Identidade da geração

```text
raw_data_hash=696e2c7878c40c1a1c230a228680baba46a430c1bef8c83cb6c7e6d5803e2c51
annotation_hash=5e4d624d5d82b61d564d0c4e7acf499714d833733f94a10dae26ebaed3788d8d
processed_data_hash=ef40e2222aaf42a5729e2005dcefe0e7b5c50cb2e308eb5303847d2a2723fff4
ontology_hash=051f4d8b476e4dde2e32b125b302b18b091e4d582d97fc217a50b1d5ab0f78a1
preprocessing_hash=2160ceda923bd7971639d232ef7b45d5eb49a771fbab15e7a44079b0411711fd
feature_schema_hash=52495044777f92c274f5c1e2f5cebe16fff38d1a2f1af7ca988ea955ea6492d5
patient_split_hash=5a357bfc9efb8fcb0078a2117975cf43003c61160c9894b39d96a1a06dcf968c
training_config_hash=b2941946f192608f488aa6ba010fd085de5966722c0c98b7b71ce9e5abafde19
source_revision=5851aa6697288b891bbb6a56125629da598666a482c6a91aba96b2d1db42cf28
environment_hash=b9ca82a645075b08eb4456801730aac1b604cb3b647a1cc854f6be4b3f2eb529
```

## Decisão

```text
TRAINING_BLOCKED_BY_DATA_PROVENANCE
REVIEW_REQUIRED
```

Nenhuma célula de treinamento pode iniciar.
