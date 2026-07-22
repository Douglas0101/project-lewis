# Handoff para o parceiro — Advanced Training v3.1

**Data:** 2026-07-22  
**Branch:** `develop`  
**Classificação:** `NEW_EVIDENCE_GENERATION`  
**Estado científico:** `REVIEW_REQUIRED`  
**Treinamento permitido:** `false`

## 1. Leia antes de executar

1. `AGENTS.md`
2. `docs/rebuild_spec/README.md`
3. `docs/rebuild_spec/14_patient_identity_and_preflight.md`
4. Este handoff
5. `reports/training_preflight_v3.1.r4.md`

Não reutilize splits v3, modelos/scalers/thresholds v2.x nem as 100 células históricas como
evidência confirmatória. Não invente identidade de paciente e não transforme record ID em patient
ID sem fonte autenticada.

## 2. O que está concluído

A geração local r4 passou todos os contratos de custódia, binding e reconstrução para:

- família MITDB+SVDB+INCART: 469.723 beats / 201 registros;
- Stage 1: 461.600 beats, sendo 406.440 N e 55.160 S+V+FUSION;
- AFDB rhythm: 83.750 episódios / 23 registros;
- split patient-wise: 123 registros confirmatórios / 79 pacientes;
- leakage legado documentado: 29 pacientes INCART cruzavam folds v3.

O único blocker do preflight r4 foi:

```text
AFDB_PATIENT_IDENTITY_UNVERIFIED
```

Não existe `PRETRAINING_GATE_PASS` e nenhum treinamento novo foi executado.

## 3. Evidência e hashes

### Artefatos grandes — locais, não enviados pelo Git

| Artefato | Bytes | SHA-256 |
| --- | ---: | --- |
| `data/features/v3.1.0-r4/finetuning_mitbih_family.npz` | 668083744 | `d8ce5061634a22aafc01cc7489552b2b4b1112338bba3c870e5ce22486168f57` |
| `data/features/v3.1.0-r4/finetuning_mitbih_family.parquet` | 51465183 | `92e0018a59bf9bad945ac833e038377d256414b2ea63486ce0efc614386b22e3` |
| `data/features/v3.1.0-r4/stage1_binary.npz` | 1111534408 | `a73e53df80ce0b178b93ba35f01f438862c88e9308eae3ce1c76b2c885b48e80` |
| `data/features/v3.1.0-r4/stage1_binary.parquet` | 80298979 | `09fa8c250353d45fcfed2ce27811d71359389166ec900668fc7c9adaadd7eb08` |
| `data/features/v3.1.0-r4/afdb_rhythm_episodes.npz` | 1715871002 | `2b4d16bbf9fe9e9894703a24b88cf5e48c6309d570c846a93b354b87efa5cf57` |
| `data/features/v3.1.0-r4/afdb_rhythm_episodes.parquet` | 8464753 | `c7676252b519aad487a95cc84829b1c8a9c00f2c809d3756944e025d8da8e791` |

O Stage 2 incidental em `v3.1.0-r4` está marcado por `STAGE2_NOT_BOUND.json` e é
**NOT_AUTHORIZED_FOR_TRAINING**.

### Relatório histórico r4

| Arquivo | SHA-256 |
| --- | --- |
| `reports/training_preflight_v3.1.r4.json` | `c7640913246537fafd587cbc5ffacae0260f4e498010a8b2634e2dba2eb9ed7f` |
| `reports/training_preflight_v3.1.r4.md` | `8d7020dd1d9bd7a223c967f41c28049f244537ba8c1ce77e02061a4a055b681e` |
| `reports/training_preflight_v3.1.r4.COMPLETE` | `9c534dfe5933af6de0c7aa2fe0c94c4c2b73a36ca0d336f71600aa351b4986f6` |

`patient_split_hash` r4: `3a15bf1e591f32e29010796d3c3a282795f4ba384da3d29ccce5c5630e9156e3`.

Os manifests pequenos do split, os arquivos `.sha256`, Markdown e markers estão no Git. O JSON
canônico completo do preflight permanece fora do Git porque seu payload de ambiente contém path
local específico da máquina; distribua-o somente por canal privado quando necessário e valide o
hash acima. Não sanitize nem regrave o artefato imutável.

> O relatório r4 foi produzido antes do commit de handoff e mantém essa identidade histórica.
> Como o contrato inclui Git HEAD/tree, após o commit use a configuração r5 para criar uma nova
> atestação local vinculada ao commit efetivamente checkoutado. Não altere o relatório r4.

## 4. Limitação de transporte de dados

Os manifests pequenos de `data/splits/.../v3.1.0/` foram incluídos explicitamente no Git, mas os
artefatos numéricos grandes em `data/features/` continuam ignorados. O remote DVC configurado
atualmente é local e específico da máquina
do owner:

```text
~/.cache/project-lewis-dvc
```

Logo, `git pull` sozinho não entrega os seis artefatos grandes. Antes do preflight, escolha uma das
opções autorizadas:

1. receber uma cópia segura dos artefatos/cache e conferir todos os hashes acima; ou
2. configurar um remote DVC compartilhado por meio de configuração **local** e obter os dados
   fonte; ou
3. regenerar em uma nova geração write-once, nunca fingindo que bytes novos são a r4.

Exemplo de configuração local, sem alterar `.dvc/config` versionado:

```bash
uv run --locked dvc remote modify --local storage /CAMINHO/DO/REMOTE-COMPARTILHADO
uv run --locked dvc pull
```

Se os bytes r4 exatos forem transferidos, valide:

```bash
sha256sum data/features/v3.1.0-r4/*.{npz,parquet}
```

## 5. Bootstrap na outra máquina

```bash
git checkout develop
git pull --ff-only origin develop
uv sync --locked --python 3.12
make lint
uv run --locked pytest -q tests -k 'not test_two_stage_qg5_end_to_end'
make test-e2e
```

Resultado de referência no owner:

```text
963 passed, 1 deselected
17/17 testes E2E passed
```

O teste excluído é o QG5 Stage 1 conhecido; não o marque como xfail e não reduza o threshold.

Após disponibilizar os seis artefatos exatos, gere a atestação pós-commit:

```bash
uv run --locked python -m src.cli.advanced_training_v3 preflight \
  --config config/advanced_training_v3.1.r5.yaml

uv run --locked python -m src.cli.advanced_training_v3 status \
  --config config/advanced_training_v3.1.r5.yaml

uv run --locked python -m src.cli.advanced_training_v3 verify-generation \
  --config config/advanced_training_v3.1.r5.yaml
```

Enquanto a identidade AFDB não for resolvida, o resultado esperado é exit code `10`,
`REVIEW_REQUIRED`, `training_allowed=false` e blocker único
`AFDB_PATIENT_IDENTITY_UNVERIFIED`. Qualquer outro resultado deve interromper o trabalho.

Não use `--publish-splits`: `data/splits/groupkfold_5_stratified/v3.1.0/` já é write-once.

## 6. Tarefas pendentes para o parceiro

### TODO #7 — Resolver o gate de identidade AFDB — P0

Opções aceitáveis:

1. localizar evidência oficial/autenticada de record→patient para todos os 23 registros usados;
2. registrar método, fonte, checksum e cobertura integral na policy e em testes; ou
3. obter decisão humana formal, nomeada e autenticada para retirar AFDB do pretraining gate,
   mantendo-o `RHYTHM_EXPLORATORY` e fora de métricas patient-wise.

Proibido:

- usar record ID como patient ID;
- assumir um paciente por registro;
- inferir identidade por semelhança de ECG;
- autoratificar a mudança de escopo.

A decisão deve atualizar `docs/rebuild_spec/12_human_decision_register.md` e produzir uma nova
geração; nunca altere r4/r5 em lugar.

### TODO #9 — Vincular provenance do Stage 2 — P0 antes de Stage 2

Implementar e testar:

- `sample_id` e `waveform_sha256` ordenados no NPZ;
- lineage patient/split/fold e clocks nativo/500 Hz;
- transformação exata família→Stage 2, incluindo exclusões N/Q e deduplicação DQ-04;
- binding NPZ/parquet e reconstrução da fonte;
- paths e markers write-once em uma nova geração.

Não use `data/features/v3.1.0-r4/stage2_multiclass.*`.

### TODO #3 — Executar matriz controlada — bloqueada por #7

Somente iniciar após `PRETRAINING_GATE_PASS` válido e revalidado no consumo. Preservar:

- outer/inner/calibration/threshold patient-wise;
- GroupKFold por paciente;
- seeds `[17, 29, 43, 71, 101]`;
- ausência de padding zero e SMOTE somente no feature space;
- resultados imutáveis por célula.

### TODO #4 — Robustez confirmatória — bloqueada por #3

Executar bootstrap por paciente, análise de transporte/domínio, controles negativos, atalhos de
dataset e auditoria de calibração sem chamar dados internos de validação externa.

### TODO #5 — Bundle de decisão — bloqueada por #4

Produzir o bundle auditável final, mantendo promoção clínica bloqueada até revisão humana e
validação externa genuína.

## 7. Checklist de entrega do parceiro

- [ ] Branch própria baseada no `develop` atualizado.
- [ ] Nenhum arquivo r2/r3/r4/r5 sobrescrito.
- [ ] Evidência AFDB autenticada ou decisão formal anexada.
- [ ] Testes TDD para qualquer mudança de contrato.
- [ ] `make lint` aprovado.
- [ ] Suíte ampla e E2E aprovadas, com o QG5 conhecido explicitamente reportado.
- [ ] Novo preflight, `status` e `verify-generation` reproduzíveis.
- [ ] Nenhum `PRETRAINING_GATE_PASS` criado enquanto existir blocker.
- [ ] Revisão humana obrigatória antes de merge para mudanças críticas de dados/firmware/LGPD.
