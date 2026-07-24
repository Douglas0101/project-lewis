# Stage 1 positive-class contract

## Status

**PROVED for the v2.0 training contract:** output index `0 = N/Normal` and output
index `1 = Anormal`.

This conclusion does not rely only on the inference pipeline selecting column 1.
It follows the original integer target construction through sparse categorical
training and is corroborated by contemporaneous reports.

## Evidence chain

### 1. Source AAMI mapping to the binary target

`scripts/prepare_two_stage_datasets.py:46-108` constructs and persists:

```python
y_bin = np.where(y == 0, 0, 1).astype(np.int64)
```

The script documents the source AAMI convention as `N=0`, followed by `S/V/F/Q`.
Therefore the persisted sparse Stage 1 labels are:

| Integer target | Meaning | Source classes |
| ---: | --- | --- |
| 0 | N / Normal | N |
| 1 | Anormal | S, V, F, Q by the default builder |

### 2. No one-hot reordering

`config/stage1_binary.yaml` binds training to
`data/features/stage1_binary.npz`, declares `num_classes: 2`,
`activation: softmax`, and `loss: sparse_categorical_crossentropy`.
Sparse categorical cross-entropy supervises output coordinate `k` with integer
target `k`; there is no `LabelEncoder` or one-hot transformation between the
persisted integer target and model training.

### 3. Ordered names used by training and evaluation

`scripts/run_stage1_training.py:64-194` loads that dataset and passes:

```python
class_names = ["N", "Anormal"]
```

together with the unchanged integer `y` into GroupKFold training. The same
ordered names are written to lineage. The original two-stage implementation
commit `27ad38b` contains the same dataset mapping, sparse loss, and ordered
class names, tying the contract to the v2.0 producer rather than a later
inference assumption.

### 4. Serialized artifact structure

The immutable ZIP inspection at
`artifacts/stage1_recall_investigation/R02/artifact_inspection.json` confirms:

- final layer: `Dense(units=2, activation="softmax")`;
- historical loss: `sparse_categorical_crossentropy`;
- output shape: `(None, 2)`;
- no label mapping is serialized inside `config.json`.

The ZIP proves the two-output sparse categorical structure. The training chain
above supplies the semantic mapping that the ZIP itself omits.

### 5. Independent historical corroboration

`reports/two_stage_evaluation_v2.0.json` orders per-class results as `N` then
`Anormal` and stores its 2×2 confusion matrix in that order.
`reports/final_architecture_v2.0.md` also describes output `(Normal, Anormal)`.
These are corroborating evidence, not the primary proof.

## Contract

```text
output[:, 0] = P(Normal)
output[:, 1] = P(Anormal)
positive class for Stage 1 metrics = 1
positive output index = 1
```

The probability notation is justified structurally by the final two-unit
softmax. R07 must still verify runtime sums, output ranges, and absence of an
extra activation in the complete inference path.

## Limitations

- The label mapping is not embedded in the `.keras` archive.
- No immutable manifest cryptographically links the current model, scaler, and
  threshold bytes; that family-level gap remains for R05/R10.
- Historical documentation contains inconsistent parameter-count and loss text,
  but no contradictory output polarity was found. The serialized artifact and
  executable training configuration take precedence.
