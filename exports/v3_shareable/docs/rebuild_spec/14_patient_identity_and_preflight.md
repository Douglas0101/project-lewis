# 14 — Identidade de Paciente e Preflight v3.1

**Status:** `IMPLEMENTED_FOR_PREFLIGHT / TRAINING_BLOCKED_BY_DATA_PROVENANCE`  
**Geração:** `advanced-training-v3.1.0-r1`  
**Escopo:** pesquisa offline; promoção clínica proibida

## 1. Regra fail-closed

`record_id` não é sinônimo de `patient_id`. Uma fonte só pode participar do núcleo
confirmatório quando existe evidência versionada para a relação registro–paciente. Aprovação
humana não converte metadado ausente em evidência observada.

Estados usados:

```text
IDENTITY_VERIFIED
IDENTITY_UNVERIFIED
IDENTITY_UNVERIFIED_QUARANTINED
TRAINING_BLOCKED_BY_DATA_PROVENANCE
```

## 2. Política por dataset

| Dataset | Evidência disponível | Registros | Grupos verificáveis | Papel v3.1 |
| --- | --- | ---: | ---: | --- |
| INCART | `# patient N` nos headers; N em 1..32 | 75 | 32 | `CONFIRMATORY_CORE` |
| MITDB | 48 registros/47 sujeitos; 201 e 202 na mesma fita | 48 | 47 | `CONFIRMATORY_CORE` |
| SVDB | headers e overview sem mapping paciente | 78 | não demonstrado | `DOMAIN_SENSITIVITY` |
| AFDB | manifest atual sem mapping autenticado | 23 completos | não demonstrado | `RHYTHM_EXPLORATORY` |

IDs são prefixados pelo dataset para impedir colisões. O manifest não persiste idade, sexo,
diagnóstico, medicamentos ou comentários clínicos dos headers.

## 3. Defeito comprovado nos splits legados

O builder legado agrupava por `record_id` e publicava esses valores como `val_patient_ids`.
Nos 75 registros INCART, **29 dos 32 pacientes reais** aparecem em mais de um outer fold v3.
Logo, a matriz antiga não é patient-wise e permanece inválida.

Os builders legados, o runner da matriz e a célula de sanidade foram retidos somente como
dados não executáveis em `scripts/legacy/`; seus entry points ativos agora retornam exit code 10
antes de carregar TensorFlow ou escrever dados. O snapshot da matriz inclui o último guard de
segurança anterior ao arquivamento e, portanto, não é descrito como cópia original `verbatim`.

Durante o teste TDD que demonstrou a ausência inicial desse bloqueio, o builder legado foi
executado uma vez e regravou os cinco manifests beat v3 em
`2026-07-21T03:49:51.776670+00:00`–`2026-07-21T03:49:51.821142+00:00`. O diretório `data/` é
ignorado pelo Git, logo não existe baseline versionado capaz de provar quais bytes precediam o
incidente. Há, contudo, incompatibilidade positiva: o resultado de sanidade de 2026-07-18
registra para o fold 1 o hash
`157842016633827dc3f1a67f5cb0b06af3b12f1e671a549a56ca5a3ebd711813`, enquanto o manifest
regravado declara
`a53d29b232c78ba5e473ec3a3eca9e1cf6cbf6fa2d01a500d4ff3fa82a6a8a75`. Nenhuma métrica é
reinterpretada; `v3/` permanece histórico e não confirmatório. O novo publicador usa lock com
ownership, recusa overwrite e nunca repara `v3/` em lugar.

## 4. Split candidato v3.1.0

O split v3.1 usa `StratifiedGroupKFold(n_splits=5, random_state=42)` somente no núcleo com
identidade verificável:

```text
MITDB + INCART = 123 registros / 79 grupos de paciente
SVDB = quarentena para sensibilidade de domínio
AFDB = tarefa de ritmo exploratória
```

Cada paciente e registro do núcleo aparece em exatamente um conjunto de outer test. O bundle só
poderia ser publicado depois de `FAMILY_SOURCE_CUSTODY_COMPLETE`, binding ordenado e reconstrução
integral da família fonte. Esses gates passaram para a geração r4 e o bundle foi publicado uma
única vez em:

```text
data/splits/groupkfold_5_stratified/v3.1.0/
```

Cada JSON possui SHA-256 detached e o marker `SPLIT_BUNDLE_COMPLETE.json` foi escrito por último.
Uma segunda publicação, mesmo idêntica, é rejeitada. O split vincula a família r4 e o manifest de
identidade construído com os 23 registros AFDB quarentenados; nenhuma linha do split v3 legado foi
reparada ou reutilizada.

## 5. Cadeia de custódia regenerada e blocker residual

Os artefatos legados continuam inválidos e inalterados. A regeneração write-once produziu duas
tentativas rejeitadas antes da geração aceita para preflight:

- `v3.1.0-r2`: rejeitada porque a família tinha rows `(500,)`, não `(500, 1)`, e o AFDB serializou
  IDs como arrays object incompatíveis com `allow_pickle=False`;
- `v3.1.0-r3`: rejeitada porque drops tolerados de anotações de borda renumeravam `beat_idx` e
  `sample_id` em sete registros; os artefatos e `GENERATION_REJECTED.json` foram preservados;
- `v3.1.0-r4`: preserva o índice original da anotação, usa shape canônico, arrays NPZ sem pickle e
  passou reconstrução integral da fonte.

A geração r4 contém:

```text
data/features/v3.1.0-r4/finetuning_mitbih_family.{npz,parquet}
  201 registros / 469.723 beats / MITDB+SVDB+INCART
data/features/v3.1.0-r4/stage1_binary.{npz,parquet}
  461.600 beats / 406.440 N / 55.160 S+V+FUSION
data/features/v3.1.0-r4/afdb_rhythm_episodes.{npz,parquet}
  83.750 episódios / 23 registros / tarefa de ritmo quarentenada
```

O parquet Stage 1 contém o contrato completo por amostra:

```text
dataset_id
patient_id
record_id
beat_index
segment_id
sample_id
waveform_sha256
source_sampling_rate
target_sampling_rate
annotation_index_native
annotation_time_seconds
annotation_index_target
class_original
class_canonical
y
quality_label
split
fold
```

Os NPZ r4 contêm `sample_id` e `waveform_sha256` ordenados. Os produtores usam paths específicos
da geração, lock com ownership, staging no mesmo filesystem, hard-link sem replace e targets
read-only. O Stage 1 recebe explicitamente o manifest de identidade e o split completo; não
inventa campos ausentes. O preflight recalcula SHA-256 sobre os bytes canônicos `float32`, exige
a transformação pai→Stage 1 completa e reconstrói toda a população AFDB elegível.

A execução final r4 deve manter todos os gates de custódia em `PASS`, incluindo:

```text
FAMILY_SOURCE_CUSTODY_COMPLETE
FAMILY_ORDERED_ROW_BINDING_VALIDATED
FAMILY_SOURCE_RECONSTRUCTION_VALIDATED
SAMPLE_LINEAGE_VALUES_VALIDATED
ORDERED_ROW_BINDING_VALIDATED
STAGE1_PARENT_BINDING_VALIDATED
AFDB_LINEAGE_VALUES_VALIDATED
AFDB_ORDERED_ROW_BINDING_VALIDATED
AFDB_SOURCE_RECONSTRUCTION_VALIDATED
```

O blocker material residual é `AFDB_PATIENT_IDENTITY_UNVERIFIED`. Portanto o estado continua
`REVIEW_REQUIRED`, nenhum `PRETRAINING_GATE_PASS` pode ser criado e nenhum treinamento é iniciado.

## 6. Identidade da geração

A CLI canônica calcula e vincula:

```text
raw_data_hash
annotation_hash
processed_data_hash
ontology_hash
preprocessing_hash
feature_schema_hash
patient_split_hash
training_config_hash
source_revision
environment_hash
```

`source_revision` é um snapshot content-addressed dos arquivos semânticos mais a identidade Git;
não é substituído por `HEAD` quando a árvore está dirty. `payload_hash` e SHA-256 dos bytes finais
são contratos distintos. Antes de qualquer publicação, todos os arquivos do bundle são
re-hasheados; a publicação final r4 só é considerada completa após o marker
`reports/training_preflight_v3.1.r4.COMPLETE`, escrito por último e vinculado aos hashes do JSON,
do Markdown e do bundle. Publicação e leitura recomputam o preflight canônico, portanto checks
`PASS` fornecidos pelo chamador não são confiados. Um gate exige o conjunto canônico de checks,
todos os componentes obrigatórios, os vínculos fonte→identidade→split→geração e
`requires_revalidation_at_consumption=true`; o marker isolado nunca autoriza um consumidor.

## 7. CLI

```bash
uv run --locked python -m src.cli.advanced_training_v3 preflight \
  --config config/advanced_training_v3.1.r4.yaml

uv run --locked python -m src.cli.advanced_training_v3 status \
  --config config/advanced_training_v3.1.r4.yaml

uv run --locked python -m src.cli.advanced_training_v3 verify-generation \
  --config config/advanced_training_v3.1.r4.yaml
```

Exit code 10 significa bloqueio científico válido, não falha de software.
