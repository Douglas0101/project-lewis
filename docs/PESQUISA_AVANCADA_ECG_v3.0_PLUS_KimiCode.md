# Pesquisa Avançada & Aprimoramento de Precisão — Modelos ECG v2.0 → v3.0+
## Compilação de Estado da Arte 2025–2026 para Execução no Kimi Code

> **Projeto**: Pipeline ML para classificação de arritmias ECG (N, S, V, F)
> **Ferramenta**: Kimi Code (K2.6)
> **Autor**: Douglas Souza
> **Data**: 2026-06-28
> **Fontes**: PubMed/Nature/MDPI/Frontiers — papers 2025–2026
> **Status**: Expansão do PRD/SDD com técnicas validadas em literatura

---

## 1. Síntese das Descobertas da Literatura

### 1.1. Arquiteturas Híbridas — Estado da Arte

A literatura 2025–2026 consolida uma hierarquia arquitetural para ECG:

| Arquitetura | F1-macro | Destaque | Limitação | Fonte |
|---|---|---|---|---|
| **CNN-BERT (ECGBert)** | 95.90% | Captura local + global semântico; F1(F) = 97.83% | F1(V) = 84.47% (falha em V) | PMC12739122 |
| **HCTG-Net (CNN-Transformer + Gated Fusion)** | — | Fusão adaptativa CNN/Transformer; residual blocks; positional encoding sinusoidal | Requer tuning do gate | MDPI 2025-11 |
| **MAK-Net (Multi-Scale KAN + BiGRU + Attention)** | **98.88%** | KAN layers com spline learnable; 4-branch multiscale; Focal+SMOTE | Complexidade computacional | PMC12252100 |
| **CNN+Bi-LSTM (baseline)** | 87.40% | Simples, interpretável | Overlap severo N/S/V | PMC12739122 |
| **ECGTransform (pure Transformer)** | 89.22% | Melhor em V (96.69%) | Fraco em F (86.78%) | PMC12739122 |

**Veredito arquitetural**: Nenhuma arquitetura pura resolve todas as classes. A solução é **ensemble arquitetural** ou **fusão gated** (HCTG-Net). O MAK-Net atinge o melhor F1-macro conhecido mas é computacionalmente pesado. Para edge deployment (<150K params), recomenda-se **HCTG-Net leve** ou **CNN-BiGRU-Attention** com KAN na camada de saída.

---

### 1.2. Focal Loss — Superioridade Comprovada

Estudo em dataset real do Seoul Asan Medical Center (6 classes, 12-lead ECG):

| Método | Precision | Recall | F1-score | Accuracy |
|---|---|---|---|---|
| Imbalance (baseline) | 0.90 | 0.86 | 0.88 | 0.90 |
| **Focal Loss** | **0.95** | **0.95** | **0.95** | **0.95** |
| Class Weight | 0.93 | 0.91 | 0.93 | 0.93 |
| Balance (undersampling) | 0.86 | 0.86 | 0.86 | 0.86 |

**Conclusão**: Focal Loss supera class weights e balanceamento por ~2–9% de F1. Undersampling é **prejudicial** (destrói informação da classe majoritária).

**Recomendação para v3.0**: Manter Focal Loss (γ=2.0, α=0.25) como loss primária. Adicionar **class weight dinâmico** como regularizador secundário apenas se F1(F) < 0.30 após 20 epochs.

---

### 1.3. Augmentation Fisiológica — Hierarquia de Eficácia

A Frontiers Digital Health (2025) estabelece uma hierarquia de augmentation para ECG:

**Eficazes (preservam morfologia P-QRS-T)**:
1. **Controlled time-shifting** — deslocamento temporal controlado (< 50ms)
2. **Amplitude scaling** — multiplicação por fator 0.9–1.1
3. **Additive Gaussian noise** — SNR > 20dB
4. **Mild temporal warping** — stretching/compression < 10% do comprimento

**Prejudiciais (degradam performance)**:
- Heavy frequency modulation — altera espectro de frequência do QRS
- SMOTE linear em espaço bruto — interpolação entre batimentos distintos cria artefatos não-fisiológicos
- Oversampling agressivo (>5x) — overfitting na classe minoritária

**Recomendação para v3.0**: Implementar **augmentation fisiológica** (time-shift + amplitude + noise) em pipeline separado do SMOTE. SMOTE deve ser aplicado **após** augmentation, apenas se a classe ainda estiver < 20% do dataset.

---

### 1.4. KAN (Kolmogorov-Arnold Network) — Nova Fronteira

O MAK-Net introduz **KAN layers** com spline activations learnable na camada de classificação:

- **Vantagem**: KAN substitui MLPs tradicionais por funções de base spline, capturando não-linearidades complexas com menos parâmetros.
- **Resultado**: MAK-Net atinge 0.9980 accuracy, 0.9888 F1, 0.9871 recall no MIT-BIH.
- **Custo**: ~2x tempo de treino vs MLP tradicional.

**Recomendação para v3.0**: Substituir a camada Dense(128) -> Dense(64) -> Dense(output) por **KAN layer** na saída do BiGRU. Isso adiciona ~5K parâmetros mas melhora separabilidade de classes minoritárias (S, V, F).

---

### 1.5. Quantização Híbrida para Edge — QAT + KD + Mixed Precision

A Nature Scientific Reports (2025) valida uma estratégia de compressão híbrida:

| Técnica | Tamanho | F1 Score | Aplicabilidade |
|---|---|---|---|
| Baseline FP32 | 100% | 84.36 | Referência |
| PTQ (static) | 25% | 73.84 | Degradação severa |
| **QAT + KD** | **11%** | **83.89** | Melhor custo-benefício |
| LLM.int8() (FP16 outliers) | 25% | 81.20 | Para transformers |
| Mamba puro (sem QAT) | 8% | 72.79 | Apenas se RAM < 32KB |

**Conclusão**: QAT combinado com Knowledge Distillation (KD) é superior a PTQ puro. O teacher é o modelo FP32 completo; o student é uma versão enxuta (CNN-BiGRU leve) que aprende os logits do teacher antes da quantização.

**Recomendação para v3.0**:
1. Treinar teacher FP32 (CNN-BiGRU-KAN, ~120K params).
2. Distilar para student CNN-BiGRU leve (~60K params).
3. Aplicar QAT no student.
4. Converter para INT8 com mixed-precision (output layer em FP16 se ΔF1 > 0.02).

---

### 1.6. Ensemble Arquitetural — Correção de Falhas de Classe

O ECGBert falha em V (84.47% F1) mas é excelente em F (97.83%). O ECGTransform é o oposto (V: 96.69%, F: 86.78%).

**Padrão identificado**: Arquiteturas CNN-lean falham em classes com alta variabilidade morfológica (V). Arquiteturas Transformer-lean falham em classes com dependência local fina (F).

**Recomendação para v3.0**: Implementar **ensemble arquitetural** com 2 especialistas:
- **Especialista A** (CNN-BiGRU): Otimizado para F1(F) e F1(S) — foco em morfologia local.
- **Especialista B** (CNN-Transformer leve): Otimizado para F1(V) — foco em dependências globais.
- **Fusão**: Weighted average por classe, onde os pesos são aprendidos via meta-learning (ou grid search) no conjunto de validação.

---

## 2. Expansão do Plano de Engenharia — v3.0+

### 2.1. Nova Arquitetura Recomendada: HCTG-KAN

Fusão das melhores técnicas identificadas:

```
Input: (None, 500, 1)

BRANCH CNN (Local Morphology)
  Conv1D(64, k=7) -> BN -> ReLU -> MaxPool(2)
  ResBlock(64, k=3) -> BN -> ReLU
  ResBlock(128, k=3, stride=2) -> BN -> ReLU
  ResBlock(256, k=3, stride=2) -> BN -> ReLU
  AdaptiveAveragePooling1D -> c in R^256

BRANCH TRANSFORMER (Global Dependencies)
  LinearEmbedding(d_model=128)
  + Sinusoidal Positional Encoding
  TransformerEncoder(n_layers=4, n_heads=4, d_k=64)
  -> t in R^128

GATED FUSION
  gate = sigma(W_g * [c; t])  # learned gate 0-1
  fused = gate * c + (1-gate) * t

TEMPORAL MODELING
  BiGRU(64, return_sequences=True)
  BiGRU(32, return_sequences=False)

KAN CLASSIFIER
  KANLayer(in_features=32, out_features=64, grid_size=5)
  Dropout(0.3)
  KANLayer(in_features=64, out_features=output_classes)
  Softmax
```

**Parâmetros estimados**: ~110K–130K (dentro do limite edge de 150K).

**Hiperparâmetros críticos** (validados na literatura):
- d_model=128 (Transformer) — balance entre expressividade e eficiência.
- n_layers=4 (Transformer) — mais camadas não melhoram F1 em ECG curto (500 pts).
- grid_size=5 (KAN) — spline com 5 intervals é suficiente para separação de 4 classes.
- gate inicializado com bias = 0.5 (equal weight CNN/Transformer no início).

---

### 2.2. Pipeline de Augmentation Fisiológica

Pseudocódigo para Kimi Code:

```python
import numpy as np

def augment_ecg_physiological(x, prob=0.5):
    """
    x: array shape (500, 1), já normalizado Z-score
    Retorna: x_aug com mesma shape
    """
    if np.random.rand() > prob:
        return x
    
    # 1. Time shifting (±25 amostras = ±50ms a 500Hz)
    shift = np.random.randint(-25, 26)
    x_aug = np.roll(x, shift, axis=0)
    if shift > 0:
        x_aug[:shift] = 0  # zero-padding, não wrap-around
    elif shift < 0:
        x_aug[shift:] = 0
    
    # 2. Amplitude scaling (0.9–1.1)
    scale = np.random.uniform(0.9, 1.1)
    x_aug *= scale
    
    # 3. Additive Gaussian noise (SNR = 20dB)
    signal_power = np.mean(x_aug**2)
    noise_power = signal_power / (10**(20/10))
    noise = np.random.normal(0, np.sqrt(noise_power), x_aug.shape)
    x_aug += noise
    
    # 4. Mild temporal warping (stretch/compress ±5%)
    if np.random.rand() < 0.3:
        factor = np.random.uniform(0.95, 1.05)
        new_len = int(500 * factor)
        x_interp = np.interp(
            np.linspace(0, new_len-1, 500),
            np.arange(new_len),
            np.resize(x_aug, (new_len, 1))[:,0]
        )
        x_aug = x_interp.reshape(-1, 1)
    
    return x_aug
```

**Regra de aplicação**:
- Aplicar **antes** do split (no dataset de treino completo).
- Nunca aplicar no teste/validação.
- Para classe F: aplicar com prob=0.8 (oversampling implícito).
- Para classe N: aplicar com prob=0.2 (evitar overfitting na classe dominante).

---

### 2.3. Knowledge Distillation + QAT Pipeline

Pseudocódigo para Kimi Code — 3 estágios:

```python
# STAGE 1: Teacher Training (FP32)
teacher = build_hctg_kan(config='teacher')  # ~120K params, full KAN
teacher.compile(optimizer='adam', loss=weighted_focal_loss(...))
teacher.fit(X_train, y_train, validation_data=(X_val, y_val))

# STAGE 2: Student Training com KD
student = build_hctg_kan(config='student')  # ~60K params, KAN simplificado
student.compile(
    optimizer='adam',
    loss=lambda y_true, y_pred: (
        0.7 * weighted_focal_loss(y_true, y_pred) +  # hard labels
        0.3 * tf.keras.losses.KLDivergence()(
            tf.nn.softmax(teacher(X_train) / T=4),  # soft labels
            tf.nn.softmax(y_pred / T=4)
        )
    )
)
student.fit(X_train, y_train, validation_data=(X_val, y_val))

# STAGE 3: QAT no Student
import tensorflow_model_optimization as tfmot
q_aware_student = tfmot.quantization.keras.quantize_model(student)
q_aware_student.compile(optimizer=Adam(1e-4), loss=weighted_focal_loss(...))
q_aware_student.fit(X_train, y_train, epochs=10, batch_size=32)

# CONVERSÃO TFLITE
converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_student)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.int8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
converter.representative_dataset = representative_dataset_generator

tflite_model = converter.convert()
# Se F1 degrada > 0.02: fallback para mixed-precision (output FP16)
```

**Temperatura T=4**: Soft labels do teacher são suavizadas, permitindo ao student aprender a incerteza do teacher nas classes minoritárias (V, F).

---

### 2.4. Threshold Calibration por Classe — Método Otsu Adaptativo

Além do PR-AUC, a literatura médica recomenda **Otsu multi-level thresholding** para separar distribuições de probabilidade sobrepostas:

```python
from sklearn.metrics import precision_recall_curve
from skimage.filters import threshold_multiotsu

def calibrate_thresholds_otsu(y_true, y_pred_proba, min_recall=0.30):
    """
    y_true: one-hot (n_samples, n_classes)
    y_pred_proba: softmax output (n_samples, n_classes)
    """
    thresholds = {}
    for i, class_name in enumerate(['N', 'S', 'V', 'F']):
        # Precision-Recall curve
        precision, recall, thr = precision_recall_curve(y_true[:, i], y_pred_proba[:, i])
        
        # Otsu na distribuição de probabilidades da classe
        probs = y_pred_proba[:, i]
        otsu_thr = threshold_multiotsu(probs, classes=2)[0]
        
        # Selecionar threshold que maximize F1 com recall >= min_recall
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        valid_idx = recall >= min_recall
        if valid_idx.any():
            best_idx = np.argmax(f1_scores[valid_idx])
            thresholds[class_name] = thr[valid_idx][best_idx]
        else:
            # Fallback: usar Otsu se PR curve não atinge recall mínimo
            thresholds[class_name] = otsu_thr
    
    return thresholds
```

---

### 2.5. Nested Cross-Validation com Bootstrap CI

```python
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils import resample

def nested_cv_bootstrap_ci(X, y, groups, n_splits=5, n_bootstrap=1000):
    """
    Outer loop: GroupKFold estratificado por paciente
    Inner loop: GridSearchCV para hyperparameters
    Bootstrap: 1000 resamples para CI 95% de F1 por classe
    """
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    results = {fold: {} for fold in range(n_splits)}
    
    for fold, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Inner loop: otimização de gamma (focal loss), dropout, learning_rate
        best_model = grid_search_inner(X_train, y_train)
        
        # Avaliação no teste
        y_pred = best_model.predict(X_test)
        
        # Bootstrap CI
        f1_bootstraps = []
        for _ in range(n_bootstrap):
            idx_boot = resample(range(len(y_test)), random_state=_)
            f1_bootstraps.append(f1_score(y_test[idx_boot], y_pred[idx_boot], average='macro'))
        
        ci_lower = np.percentile(f1_bootstraps, 2.5)
        ci_upper = np.percentile(f1_bootstraps, 97.5)
        results[fold] = {
            'f1_macro': np.mean(f1_bootstraps),
            'ci_95': [ci_lower, ci_upper],
            'f1_per_class': f1_score(y_test, y_pred, average=None)
        }
    
    return results
```

**Regra de aceitação**: Modelo é aprovado apenas se **todos os folds** tiverem CI 95% inferior > 0.50 para F1-macro e CI inferior > 0.30 para F1(F).

---

## 3. Prompts Orquestrados Atualizados para Subagentes

### Subagente: Arquiteto de Modelos (Novo)
```
Você é o Arquiteto de Modelos do projeto ECG v3.0+.
Implemente a arquitetura HCTG-KAN conforme especificação da Seção 2.1.

Entregáveis:
1. src/models/hctg_kan.py com:
   - Branch CNN com ResBlocks (64->128->256, kernel 7->3->3)
   - Branch Transformer com Positional Encoding sinusoidal (d_model=128, n_layers=4)
   - Gated Fusion module (gate = sigmoid(W*[c;t]))
   - BiGRU(64,32) temporal
   - KANLayer (grid_size=5) na saída
   - ~120K params (teacher), ~60K params (student)

2. src/models/kan_layer.py com:
   - Implementação eficiente de Kolmogorov-Arnold Network layer
   - BaseFunction: spline B-spline de ordem 3
   - grid_size=5, k=3 (cubic spline)
   - Forward pass: y = sum phi_i(x_i) onde phi_i é spline learnable

3. Testes: verificar que output shape está correto para Stage 1 (2 classes) 
   e Stage 2 (3 classes). Verificar que gate inicializa em 0.5.

Restrições:
- Keras 3 / TensorFlow 2.16+
- Não usar Radix UI
- Seed 42 em todos os inicializadores
- Documentar parâmetros totais e FLOPs estimados
```

### Subagente: Data Engineer (Atualizado)
```
Você é o Data Engineer do projeto ECG v3.0+.
Implemente augmentation fisiológica + SMOTE híbrido.

Entregáveis:
1. src/data/augment_ecg.py com:
   - time_shift(±25 amostras)
   - amplitude_scale(0.9–1.1)
   - gaussian_noise(SNR=20dB)
   - mild_warping(±5%)
   - Probabilidade de aplicação por classe: F=0.8, V=0.6, S=0.4, N=0.2

2. src/data/smote_sequence_v2.py com:
   - SMOTE aplicado APENAS após augmentation
   - k=5 vizinhos, mas com distância DTW (Dynamic Time Warping) 
     em vez de Euclidean (preserva forma temporal)
   - Oversampling_ratio: F=5x, V=2x, S=1.5x, N=1x
   - Validação: plotar 10 amostras sintéticas de F para verificação 
     visual da morfologia P-QRS-T

3. src/data/nested_cv.py com:
   - StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
   - Bootstrap CI (n=1000) para F1-macro e F1 por classe
   - Exportar JSON: reports/nested_cv_results.json

Restrições:
- Nunca leakage entre pacientes
- SMOTE apenas no treino de cada fold
- Validar que amostras sintéticas não excedem range [-5, +5] Z-score
```

### Subagente: ML Engineer (Atualizado)
```
Você é o ML Engineer do projeto ECG v3.0+.
Treine o pipeline completo: Teacher -> KD Student -> QAT.

Entregáveis:
1. src/models/train_teacher.py:
   - HCTG-KAN teacher, 150 epochs, focal loss gamma=2.0, alpha=0.25
   - EarlyStopping(patience=15, monitor='val_f1_macro')
   - Salvar melhor modelo: models/v3.0/teacher_fp32_v3.0.keras

2. src/models/train_student_kd.py:
   - Student HCTG-KAN leve (~60K params)
   - Loss = 0.7 * focal_loss + 0.3 * KL(teacher_soft, student_soft, T=4)
   - 100 epochs, LR=1e-3
   - Salvar: models/v3.0/student_fp32_v3.0.keras

3. src/quantization/qat_kd_pipeline.py:
   - QAT no student por 10 epochs, LR=1e-4
   - Conversão TFLite INT8 com representative_dataset estratificado
   - Fallback: se ΔF1 > 0.02, output layer em FP16 (mixed precision)
   - Exportar: .tflite, .h, quantization_params.json

4. src/evaluation/ensemble_architectural.py:
   - Especialista A: CNN-BiGRU (otimizado para F1(F), F1(S))
   - Especialista B: CNN-Transformer leve (otimizado para F1(V))
   - Meta-learner: pesos por classe aprendidos via grid search em validação
   - Fusão: y_final = w_A * y_A + w_B * y_B

Métricas de saída:
- F1-macro Stage 1 >= 0.65 (CI 95%)
- F1-macro Stage 2 >= 0.55 (CI 95%)
- F1(F) >= 0.35 (CI 95%)
- F1(V) >= 0.70 (CI 95%)
- ΔF1 pós-QAT <= 0.02
```

### Subagente: Edge Engineer (Atualizado)
```
Você é o Edge Engineer do projeto ECG v3.0+.
Otimize o modelo para firmware CI com QAT + Mixed Precision.

Entregáveis:
1. src/quantization/mixed_precision_qat.py:
   - Identificar camadas sensíveis (primeira conv, última dense/KAN)
   - Aplicar QAT global, mas marcar camadas sensíveis como 
     tfmot.quantization.keras.QuantizeConfig personalizada com 
     supported_types=[tf.float16] para fallback seletivo
   - Converter para TFLite com schema de quantização por camada

2. src/quantization/representative_dataset_v2.py:
   - 500 amostras por classe, com augmentation fisiológica aplicada
   - Estratificação: 20% N, 20% S, 20% V, 20% F, 20% Anormal
   - Verificar que amostras cobrem range completo de Z-score (-3, +3)

3. src/quantization/benchmark_edge.py:
   - Simular inferência em ARM Cortex-M4 (120MHz)
   - Medir: latência (ms), RAM usage (KB), Flash usage (KB)
   - Target: < 50ms, < 64KB RAM, < 60KB Flash

4. Firmware headers (.h):
   - stage1_int8_qat_v3.0.h com scales/zero_points inline
   - stage2_int8_qat_v3.0.h com scales/zero_points inline
   - Macros: MODEL_INPUT_SCALE, MODEL_INPUT_ZP, MODEL_OUTPUT_SCALE, etc.

Restrições:
- TensorFlow Lite Micro compatível (sem ops customizados)
- KAN layer deve ser convertido para lookup table ou MLP equivalente 
  se TFLite Micro não suportar spline ops
```

---

## 4. Quality Gates v3.0+ (Expandidos)

| Gate | Critério | Método | Owner | Prioridade |
|---|---|---|---|---|
| **QG1** | F1-macro Stage 1 >= 0.65 | Bootstrap CI 95% | ML Eng | 🔴 |
| **QG2** | F1(Anormal) >= 0.50 | Confusion matrix | ML Eng | 🔴 |
| **QG3** | F1-macro Stage 2 >= 0.55 | Bootstrap CI 95% | ML Eng | 🔴 |
| **QG4** | F1(S) >= 0.60, F1(V) >= 0.70, F1(F) >= 0.35 | Por classe CI 95% | ML Eng | 🔴 |
| **QG5** | Variância inter-fold < 0.08 | Std F1-macro 5 folds | Data Eng | 🟡 |
| **QG6** | QAT INT8: ΔF1-macro <= 0.02 | Holdout set | Edge Eng | 🟡 |
| **QG7** | Mean Drift < 0.1, PSI < 0.25 | Scaler drift | QA | 🟡 |
| **QG8** | Tempo inferência < 50ms (edge) | Benchmark simulado | Edge Eng | 🟡 |
| **QG9** | 100% cobertura testes unitários | pytest --cov | QA | 🟢 |
| **QG10** | Doc SDD/PRD assinada | Review técnico | Arquiteto | 🟢 |
| **QG11** | Augmentation fisiológica validada | Plot 100 amostras sintéticas | Data Eng | 🟡 |
| **QG12** | KAN layer convergente | Verificar gradientes não-NaN | ML Eng | 🟡 |
| **QG13** | KD: student F1 >= 95% do teacher | Validação holdout | ML Eng | 🟡 |
| **QG14** | Mixed-precision fallback funcional | Teste de regressão TFLite | Edge Eng | 🟡 |

---

## 5. Riscos & Mitigações Atualizados

| Risco | Prob | Impacto | Mitigação | Trigger |
|---|---|---|---|---|
| KAN layer instável (gradientes NaN/Explosão) | Média | Alto | Gradient clipping (max_norm=1.0); fallback para MLP se não convergir em 5 epochs | QG12 falha |
| Gated Fusion converge para gate=0 ou 1 (colapso) | Média | Alto | Regularização L2 no gate (lambda=0.01); inicialização bias=0.5 | t-SNE mostra cluster único |
| Augmentation fisiológica cria artefatos em F | Baixa | Médio | Validação visual obrigatória; rejeitar amostras com Z-score > ±5 | QG11 falha |
| KD não converge (student underfitting) | Baixa | Médio | Aumentar T de 4->8; aumentar peso KL de 0.3->0.5; aumentar epochs student | F1 student < 90% teacher |
| TFLite Micro não suporta KAN/spline ops | Média | Alto | Converter KAN para lookup table pré-computada (256 entradas, int8) ou substituir por MLP 2-camadas | Falha na conversão |
| Ensemble arquitetural aumenta tempo >50ms | Média | Alto | Executar especialistas em paralelo (dual-core MCU) ou reduzir para single model com gate adaptativo | Benchmark >50ms |
| QAT + Mixed Precision aumenta tamanho >60KB | Baixa | Médio | Quantizar KAN lookup table para 4-bit; reduzir d_model de 128->64 | Flash >60KB |

---

## 6. Referências Bibliográficas Consolidadas

1. **ECGBert: CNN-BERT Hybrid for ECG Arrhythmia** — PMC12739122, 2025. F1-macro 95.90%, F1(F) 97.83%, mas F1(V) 84.47%.
2. **HCTG-Net: CNN-Transformer Gated Fusion** — MDPI Bioengineering 12(11):1268, 2025. Residual blocks, positional encoding, adaptive pooling.
3. **MAK-Net: Multi-Scale KAN + BiGRU** — PMC12252100, 2025. Accuracy 99.80%, F1 98.88% no MIT-BIH. KAN layers com spline learnable.
4. **Focal Loss vs Class Weight vs Balance** — CINC 2023/Seoul Asan Medical Center. Focal Loss F1=0.95 vs Class Weight 0.93 vs Balance 0.86.
5. **Imbalance-aware Loss Functions in Medical Imaging** — ML Research v250, 2024. Focal loss supera SMOTE em imagens médicas; SMOTE linear cria artefatos.
6. **TinyML with QAT + KD** — Nature Scientific Reports 2025. QAT+KD: 11% tamanho, F1 83.89 vs PTQ 73.84.
7. **EdgeML Literature Review** — ScienceDirect 2025. Mixed-precision, vector quantization, ultra-low bit (2-bit) para edge.
8. **Advances in ML/DL for ECG Classification** — Frontiers Digital Health 2025. Hierarquia: loss-level > augmentation fisiológica > transfer learning > ensemble.

---

## 7. Próximos Passos Imediatos

1. **PoC KAN Layer** (24h): Implementar `src/models/kan_layer.py` e testar convergência em subset de 1000 amostras MIT-BIH.
2. **PoC Gated Fusion** (24h): Implementar HCTG leve (sem KAN) e validar que gate não colapsa.
3. **Baseline Augmentation** (12h): Implementar augmentation fisiológica e medir F1(F) com CNN-BiGRU simples.
4. **Decision Gate**: Se F1(F) > 0.25 com augmentation + focal loss, prosseguir para HCTG-KAN completo. Se não, investigar GAN sequence-aware ou transfer learning do Chapman-Shaoxing.

---

> **Nota LGPD**: Todos os datasets (MIT-BIH, INCART, Chapman-Shaoxing, PTB-XL) são públicos e anonimizados. Scalers e modelos não contêm PII. O firmware embarcado processa sinais localmente — zero transmissão de dados brutos.