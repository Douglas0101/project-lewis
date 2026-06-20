# Project-Lewis — Camada 4: Modelagem (Backbone, Pré-treino, Fine-tuning)
## Responsável: Ciência de Dados / ML Engineering

**Versão:** 1.1 | **Data:** 2026-06-09 | **Arquiteto:** Douglas Souza

---

## 4.1 Objetivo
Construir, pré-treinar e fine-tunar um modelo 1D-CNN enxuto para classificação de arritmias AAMI (5 classes: N, V, S, F, Q) em dispositivos de borda (STM32F4, 192KB SRAM, arena TFLM ~64KB), garantindo reprodutibilidade, rastreabilidade de experimentos e conformidade com métricas AAMI EC57.

---

## 4.2 Decisão Arquitetural: Input Shape Alinhado

O input shape do modelo deve ser consistente com a segmentação da Camada 2:

| Modo | Janela | Amostras @ 500Hz | Uso |
| :--- | :--- | :--- | :--- |
| Padrão | 1000ms | **500** | Backbone + fine-tuning (batimentos normais) |
| Fallback | 600ms | **300** | Fine-tuning (taquicardia, RR < 600ms) |

> **Correção v1.0:** O documento original fixava input em 300 amostras. Isso descarta 40% do contexto temporal em batimentos normais. O backbone deve ser treinado com 500 amostras; o fallback de 300 é aplicado apenas em batimentos curtos, com modelo re-treinado ou via padding interno controlado.

> **Decisão arquitetural pendente — 500 vs 501 amostras:**
> O segmentador da Camada 2 gera janelas de **501 amostras** (ímpar, R-peak no
> centro), enquanto o modelo espera **500 amostras**. A solução recomendada é
> ajustar o segmentador para produzir 500 amostras (`window_len = 2 * half_len`,
> sem `+1`) e manter o modelo em `(500, 1)`. Essa mudança centraliza a
> responsabilidade no pré-processamento e evita camadas de adaptação no modelo.

---

## 4.3 Arquitetura Backbone 1D-CNN

**Input shape:** `(500, 1)` — 1000ms @ 500Hz, 1 canal (MLII-equivalente)  
**Classes fine-tuning:** 5 (N, V, S, F, Q) — softmax  
**Classes pré-treino (Chapman):** 5 superclasses SCP-ECG (NORM, CD, MI, HYP, STTC) — sigmoid multi-label

### src/models/backbone_1d.py
```python
def build_backbone_1d(input_len: int = 500, num_classes: int = 5, embedding_dim: int = 64):
    """1D-CNN enxuto para STM32F4 (arena ~64KB, FlatBuffer < 64KB).

    Arquitetura baseada em literatura TinyML para ECG (≈18K–20K params):
    Input(500, 1)
    → Conv1D(16, kernel=7, activation="relu", padding="same") → MaxPool1D(2)   # 250
    → Conv1D(32, kernel=5, activation="relu", padding="same") → MaxPool1D(2)   # 125
    → Conv1D(64, kernel=3, activation="relu", padding="same") → MaxPool1D(2)   # 62
    → GlobalAveragePooling1D()                                                # 64
    → Dense(embedding_dim, activation="relu")                                   # 64
    → Dropout(0.3)                                                            # 64
    → Dense(num_classes, activation="softmax", name="output")                 # 5

    Para pré-treino (Chapman, multi-label SCP-ECG):
    → Dense(num_scp_classes, activation="sigmoid")                          # 5

    RESTRIÇÕES TFLM:
    - NÃO usar LSTM/GRU/RNN (suporte limitado em TFLM; alto consumo de SRAM)
    - NÃO usar BatchNormalization (pode ser folded em Conv/Dense, mas em PTQ
      full-integer requer cuidado com zero-point; preferir omitir para simplicidade)
    - NÃO usar SeparableConv1D (TFLM decompõe em DepthwiseConv + PointwiseConv;
      nem sempre otimizado via CMSIS-NN; preferir Conv1D padrão)
    - NÃO usar attention mechanisms (overhead de memória em TFLM)
    - NÃO usar GroupNorm/LayerNorm (suporte parcial em TFLM; evitar)
    """
```

**Estimativa de tamanho (500 amostras):**
- Conv1D(16,7): 16*7*1 + 16 = 128 params
- Conv1D(32,5): 32*5*16 + 32 = 2592 params
- Conv1D(64,3): 64*3*32 + 64 = 6208 params
- Dense(64): 64*62 + 64 = 4032 params (GAP → 64 features)
- Dense(5): 5*64 + 5 = 325 params
- **Total: ~13.3K params** (~53KB float32, ~13KB INT8)
- **FlatBuffer TFLM:** ~20–25KB (com metadata e buffers de ativação)
- **Margem confortável para 64KB arena + FlatBuffer em STM32F4**

> **Referência:** Estudo TinyML em Arduino UNO (32KB Flash, 2KB SRAM) usou 1D-CNN com ~18.5K params, 21.8KB Flash, 1.7KB SRAM, inferência 200ms/beat.citeweb_search:15#0 O STM32F4 (192KB SRAM, 1MB Flash) tem margem 10× superior.

---

## 4.4 Pré-treino (Chapman-Shaoxing)

### Correção: SCP-ECG Superclasses

O documento v1.0 afirmava "~20-50 superclasses SCP-ECG". **Isso está incorreto.** O padrão SCP-ECG (como usado no PTB-XL e Chapman-Shaoxing) define **5 superclasses diagnósticas**: NORM (Normal), CD (Conduction Disturbance), MI (Myocardial Infarction), HYP (Hypertrophy), STTC (ST/T Change).citeweb_search:16#0citeweb_search:16#1

| Superclasse SCP-ECG | Descrição | Exemplos de Subclasses |
| :--- | :--- | :--- |
| NORM | ECG Normal | — |
| CD | Distúrbio de Condução | LBBB, RBBB, AV block, WPW |
| MI | Infarto do Miocárdio | Anterior, Inferior, Lateral, Posterior |
| HYP | Hipertrofia | LVH, RVH, LAO, RAO |
| STTC | Alteração ST/T | Isquemia, ST-elevation, T-wave abnormal |

### src/models/pretrain_chapman.py
```python
def pretrain_chapman(backbone, data_generator, epochs: int = 30, batch_size: int = 64):
    """Pré-treino multi-label em Chapman-Shaoxing.

    1. Usar tf.data.Dataset.from_generator (8 GB não cabe em RAM do IdeaPad)
    2. Labels: 5 superclasses SCP-ECG (one-hot, sigmoid output)
       - NORM: [1,0,0,0,0]
       - CD:   [0,1,0,0,0]
       - MI:   [0,0,1,0,0]
       - HYP:  [0,0,0,1,0]
       - STTC: [0,0,0,0,1]
       - Multi-label permitido (ex: CD + MI)
    3. Loss: binary_crossentropy
    4. Métricas: AUC-ROC por classe (macro), AUC-PR por classe
    5. Callbacks:
       - EarlyStopping(patience=5, restore_best_weights=True, monitor="val_loss")
       - ReduceLROnPlateau(patience=3, factor=0.5, monitor="val_loss")
       - ModelCheckpoint("models/backbone_pretrained_v{version}.keras", save_best_only=True)
       - TensorBoard(log_dir="logs/pretrain_chapman/")
    6. Seed fixa: tf.random.set_seed(42), np.random.seed(42), python_random.seed(42)
    7. Salvar pesos: models/backbone_pretrained_v1.0.keras + config/pretrain_v1.0.yaml
    """
```

---

## 4.5 Fine-tuning (MIT-BIH+)

### src/models/finetune_mitbih.py
```python
def finetune_mitbih(backbone, X_train, y_train, X_val, y_val, epochs: int = 100):
    """Fine-tuning com backbone congelado (transfer learning).

    1. Congelar TODAS as camadas convolucionais do backbone
       (Conv1D_1, Conv1D_2, Conv1D_3, GlobalAveragePooling1D)
    2. Retreinar APENAS o classifier (Dense + Dropout + Dense softmax)
    3. Class weights para desbalanceamento AAMI (calculados automaticamente):
       weight_c = total_samples / (n_classes * n_samples_c)
       - N: ~1.0 (majoritária, ~75%)
       - V: ~3.5–5.0 (~15%)
       - S: ~5.0–8.0 (~5%)
       - F: ~15.0–20.0 (~2%)
       - Q: ~5.0–10.0 (~3%)
       (Valores exatos computados no fit via compute_class_weight)
    4. SMOTE opcional no espaço de features (Camada 3) se necessário
    5. Loss: sparse_categorical_crossentropy
    6. Métricas: AAMI EC57 — Sens, PPV, FPR, F1 por classe; Acc global
    7. Callbacks:
       - EarlyStopping(patience=10, restore_best_weights=True, monitor="val_auc")
       - ReduceLROnPlateau(patience=5, factor=0.5, monitor="val_loss")
       - ModelCheckpoint("models/finetuned_float32_v{version}.keras", save_best_only=True)
    8. Salvar: models/finetuned_float32_v1.0.keras
    """
```

> **Nota sobre class weights:** A distribuição AAMI no MIT-BIH é severamente desbalanceada: N ≈ 75%, V ≈ 15%, S ≈ 5%, F ≈ 2%, Q ≈ 3%.citeweb_search:17#0 O uso de `sklearn.utils.class_weight.compute_class_weight` com `balanced` é preferível a valores hardcoded, pois adapta-se automaticamente à composição do fold.

---

## 4.6 Normalização de Input

**Aviso:** O documento v1.0 sugeria normalização min-max por segmento `(X - X.min) / (X.max - X.min)`. **Isso é perigoso:** batimentos com amplitude muito baixa (ruído ou flatline) geram divisão por zero ou amplificação de ruído. A normalização correta é **Z-score global** (usada na Camada 2) ou **min-max global** sobre o dataset de treino.

```python
# CORRETO: Normalização global (fit no treino, transform em val/teste)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train.reshape(-1, 1)).reshape(X_train.shape)
X_val_norm = scaler.transform(X_val.reshape(-1, 1)).reshape(X_val.shape)

# Persistir scaler para firmware
import joblib
joblib.dump(scaler, "models/input_scaler_v1.0.pkl")

# No firmware: aplicar mesma média/std (valores fixos compilados em C)
# mean = scaler.mean_[0], std = scaler.scale_[0]
```

> **Regra:** NUNCA fazer fit do scaler no conjunto de teste. NUNCA normalizar por segmento individualmente — isso quebra a correspondência amplitude→classe (ex: PVC tende a ter amplitude maior que N).

---

## 4.7 Treinamento com GroupKFold (Patient-Wise)

### Fundamentação

O GroupKFold por paciente é obrigatório em sinais fisiológicos. Misturar batimentos do mesmo paciente entre treino e teste causa **data leakage** (overfitting ao padrão individual do paciente) e infla artificialmente as métricas. A literatura distingue explicitamente entre **intra-patient** (batimentos misturados, métricas infladas ~99%) e **inter-patient** (pacientes separados, métricas realistas ~88–96%).citeweb_search:17#0citeweb_search:17#2

### src/models/train.py
```python
def train_group_kfold(X, y, groups, n_splits: int = 5, random_state: int = 42):
    """GroupKFold por paciente — NUNCA misturar batimentos do mesmo paciente.

    1. groups = array com ID do paciente (ex: "100", "101", ..., "I75")
    2. gkf = GroupKFold(n_splits=5)
    3. Para cada fold:
       a. Separar X_train, X_test, y_train, y_test por grupos
       b. Normalização global: fit scaler em X_train, transform em X_test
       c. Carregar backbone pré-treinado (models/backbone_pretrained_v1.0.keras)
       d. Congelar camadas convolucionais (trainable=False)
       e. Treinar classifier em X_train, y_train com class_weight="balanced"
       f. Avaliar em X_test, y_test (pacientes NUNCA vistos)
       g. Registrar métricas AAMI EC57 por fold
    4. Retornar: métricas médias + desvio padrão entre folds
    5. Persistir: melhor fold (maior F1-macro) como modelo final
    """
```

**Regra absoluta:** NUNCA shuffle aleatório. Data leakage em sinais fisiológicos invalida métricas e impede generalização para novos pacientes.

---

## 4.8 Avaliação AAMI EC57

O padrão ANSI/AAMI EC57:1998 define as métricas obrigatórias para avaliação de algoritmos de detecção/classificação de arritmia.citeweb_search:17#0citeweb_search:17#2

### src/models/evaluate.py
```python
def evaluate_aami(y_true, y_pred, class_names=["N","S","V","F","Q"]):
    """Avaliação completa conforme AAMI EC57.

    Métricas por classe:
    - Sensibilidade (Se) = TP / (TP + FN)  — capacidade de detectar a classe
    - PPV (+P) = TP / (TP + FP)           — confiabilidade da detecção
    - FPR = FP / (FP + TN)                — alarmes falsos
    - F1 = 2 * (PPV * Se) / (PPV + Se)    — equilíbrio
    - Especificidade (Spe) = TN / (TN + FP) — capacidade de excluir outras classes

    Métricas globais:
    - Acc = (TP_total) / (TP_total + FN_total + FP_total)
    - F1-macro = mean(F1 por classe)
    - MCC (Matthews Correlation Coefficient) — robusto a desbalanceamento

    Retornar: dict com todas as métricas + confusion_matrix
    """
```

| Métrica | Fórmula | Interpretação Clínica |
| :--- | :--- | :--- |
| **Sensibilidade (Se)** | TP / (TP + FN) | "De 100 batimentos V, quantos o modelo detectou?" |
| **PPV (+P)** | TP / (TP + FP) | "De 100 vezes que o modelo disse V, quantos eram realmente V?" |
| **FPR** | FP / (FP + TN) | "Fração de alarmes falsos por classe" |
| **Especificidade (Spe)** | TN / (TN + FP) | "De 100 batimentos não-V, quantos o modelo corretamente excluiu?" |
| **F1** | 2·Se·PPV / (Se + PPV) | Equilíbrio entre detecção e confiabilidade |
| **MCC** | (TP·TN − FP·FN) / √(...) | Métrica robusta para classes desbalanceadas |

> **Caveat:** Acurácia global (Acc) é enganosa em datasets desbalanceados. Um modelo que classifica tudo como N já atinge ~75% de Acc. Por isso, **F1-macro e MCC são as métricas primárias** para comparação entre modelos.citeweb_search:17#0

---

## 4.9 Exportação para TFLM (TensorFlow Lite Micro)

### src/models/export_tflm.py
```python
def export_tflm(model_path: Path, scaler_path: Path, output_dir: Path):
    """Exportar modelo para TFLM com PTQ INT8.

    1. Carregar modelo Keras (.keras)
    2. Criar representative_dataset (100–200 amostras do treino)
    3. Converter:
       converter = tf.lite.TFLiteConverter.from_keras_model(model)
       converter.optimizations = [tf.lite.Optimize.DEFAULT]
       converter.representative_dataset = representative_dataset
       converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
       converter.inference_input_type = tf.int8   # input em int8
       converter.inference_output_type = tf.int8  # output em int8
       tflite_model = converter.convert()
    4. Salvar: output_dir / "model_int8.tflite"
    5. Converter para C array: xxd -i model_int8.tflite > model_int8.h
    6. Extrair parâmetros de quantização (input_scale, input_zero_point,
       output_scale, output_zero_point) para uso no firmware C
    7. Salvar metadata: output_dir / "quantization_params.json"
    """
```

**Requisitos TFLM:**
- FlatBuffer size: < 64KB (arena TFLM)
- SRAM runtime: < 64KB (buffers de ativação + tensor arena)
- Flash: < 1MB (STM32F4 tem 1MB Flash)
- Ops suportados: Conv2D (usado como Conv1D via reshape), MaxPool, ReLU, FullyConnected, Softmax
- **NÃO usar:** BatchNorm (folded ou não), LSTM, SeparableConv, GroupNorm

> **Referência:** Quantização INT8 reduz modelo em ~75% com perda < 1% de acurácia.citeweb_search:15#0 O esquema de quantização TFLite usa `real_value = (int8_value - zero_point) * scale`, com zero-point representando 0.f exatamente.citeweb_search:16#5

---

## 4.10 Rastreabilidade e Reprodutibilidade

### Versionamento de Experimentos

Cada treinamento gera um diretório versionado em `experiments/`:

```
experiments/
├── exp_20260609_221500_pretrain_chapman/
│   ├── config.yaml          # Hiperparâmetros, seeds, arquitetura
│   ├── model_summary.txt    # summary() do Keras
│   ├── training.log         # CSV de epochs (loss, val_loss, AUC)
│   ├── metrics.json         # Métricas finais (AUC-ROC, AUC-PR)
│   ├── backbone_pretrained.keras
│   └── lineage.json         # SHA256 do dataset, config, seeds
│
├── exp_20260610_103000_finetune_mitbih_fold3/
│   ├── config.yaml
│   ├── model_summary.txt
│   ├── training.log
│   ├── metrics_aami.json    # Se, PPV, FPR, F1 por classe
│   ├── confusion_matrix.png
│   ├── finetuned_float32.keras
│   ├── model_int8.tflite
│   ├── model_int8.h
│   ├── quantization_params.json
│   └── lineage.json
```

### Determinismo

```python
import os, random, numpy as np, tensorflow as tf

def set_seeds(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    # Para determinismo completo em GPU (pode degradar performance):
    # os.environ["TF_DETERMINISTIC_OPS"] = "1"
    # os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
```

---

## 4.11 Dead Letter Queue (DLQ) para Treinamento

Falhas de treinamento (OOM, NaN loss, divergência) são logadas em `data/.dlq/training_failures.jsonl`:

```json
{
  "experiment_id": "exp_20260609_221500",
  "stage": "pretrain_chapman",
  "error": "Loss became NaN at epoch 12",
  "traceback": "...",
  "config": "config/pretrain_v1.0.yaml",
  "seed": 42,
  "timestamp": "2026-06-09T22:15:00Z"
}
```

**Regras:**
- NaN loss → learning rate muito alta ou gradiente exploding; reduzir LR em 10×.
- OOM → reduzir batch_size ou embedding_dim.
- Divergência (val_loss ↑ enquanto train_loss ↓) → overfitting; aumentar dropout ou reduzir embedding_dim.

---

## 4.12 Quality Gate QG4 / QG5

| Gate | Critério | Dataset | Valor Mínimo | Como Validar |
| :--- | :--- | :--- | :--- | :--- |
| **QG4** | Pré-treino convergido | Chapman | AUC-ROC macro > 0.85 | `pytest tests/test_pretrain.py` |
| **QG4** | Pré-treino loss | Chapman | loss < 0.15 | `pytest tests/test_pretrain.py` |
| **QG5** | Fine-tune Acc global | MIT-BIH+ (inter-patient) | > 93% | `pytest tests/test_finetune.py` |
| **QG5** | Fine-tune F1-macro | MIT-BIH+ (inter-patient) | > 85% | `pytest tests/test_finetune.py` |
| **QG5** | Fine-tune MCC | MIT-BIH+ (inter-patient) | > 0.80 | `pytest tests/test_finetune.py` |
| **QG5** | Sens classe N | MIT-BIH+ | > 96% | `pytest tests/test_finetune.py` |
| **QG5** | Sens classe V | MIT-BIH+ | > 90% | `pytest tests/test_finetune.py` |
| **QG5** | Sens classe S | MIT-BIH+ | > 75% | `pytest tests/test_finetune.py` |
| **QG5** | Sens classe F | MIT-BIH+ | > 60% | `pytest tests/test_finetune.py` |
| **QG5** | Sens classe Q | MIT-BIH+ | > 70% | `pytest tests/test_finetune.py` |
| **QG5** | FPR global | MIT-BIH+ | < 5% | `pytest tests/test_finetune.py` |
| **QG5** | GroupKFold std F1-macro | MIT-BIH+ | < 3% entre folds | `pytest tests/test_finetune.py` |
| **QG5** | TFLM export | — | FlatBuffer < 64KB | `ls -la model_int8.tflite` |
| **QG5** | Quantização | — | ΔAcc < 1% vs float32 | `pytest tests/test_quantization.py` |
| **QG5** | DLQ vazia | — | 0 falhas de treinamento | `data/.dlq/training_failures.jsonl` vazio |

> **Nota sobre métricas inter-patient:** Em intra-patient (shuffle aleatório), Acc ~99% é trivial. Em inter-patient (GroupKFold), Acc ~93% e F1-macro ~85% já são resultados de state-of-the-art para 1D-CNN leve.citeweb_search:17#0 Não aceitar métricas intra-patient como validação final.

---

## 4.13 Referências Verificadas

- PTB-XL / SCP-ECG 5 Superclasses (NORM, CD, MI, HYP, STTC): Wagner et al., Sci Data 7, 154 (2020). https://pmc.ncbi.nlm.nih.gov/articles/PMC7248071/ — 5 superclasses diagnósticas, 23 subclasses.citeweb_search:16#0
- SCP-ECG Superclasses em ECG-QA: https://arxiv.org/pdf/2306.15681 — "5 superclasses for diagnostic labels (CD, HYP, MI, NORM, STTC)".citeweb_search:16#3
- TinyML 1D-CNN em Arduino UNO (8-bit, 32KB Flash, 2KB SRAM): MDPI 2026. https://www.mdpi.com/2306-5354/13/5/532 — ~18.5K params, 21.8KB Flash, 1.7KB SRAM, 200ms/beat, 97.6% Acc.citeweb_search:15#0
- AAMI EC57 Métricas (Se, PPV, FPR, Spe, F1, MCC): Systematic Review 2025. https://arxiv.org/html/2503.07276v1 — "Sensitivity, Positive Predictive Value, and False Positive Rate are prioritized as the most relevant metrics."citeweb_search:17#0
- AAMI EC57 Métricas Detalhadas (PMC): https://pmc.ncbi.nlm.nih.gov/articles/PMC9760867/ — Acc, Sen, Spe, Ppv, F1, MCC.citeweb_search:17#1
- Patient-Wise Split (Inter-patient vs Intra-patient): https://hal.science/hal-03682454v5/document — "intra-patient paradigm: Acc 99.48%; inter-patient paradigm: Acc 88.34%".citeweb_search:17#0
- TensorFlow Lite INT8 Quantization (scale/zero-point): https://medium.com/tensorflow/tensorflow-model-optimization-toolkit-post-training-integer-quantization-b4964a1ea9ba — "real_value = (int8_value — zero_point) * scale".citeweb_search:16#5
- STM32F4 Specifications (192KB SRAM, 1MB Flash, 168MHz): https://baike.baidu.com/item/STM32F4/1368593 — 192Kb SRAM, 1MB Flash, 210 DMIPS.citeweb_search:16#12
- TFLM Hybrid Models Error: https://github.com/tensorflow/tensorflow/issues/43386 — "Hybrid models are not supported on TFLite Micro."citeweb_search:15#4
