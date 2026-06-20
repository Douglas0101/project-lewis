# Project-Lewis — Camada 3: Segmentação e Feature Engineering
## Responsável: Ciência de Dados + Engenharia de Dados

**Versão:** 1.1 | **Data:** 2026-06-09 | **Arquiteto:** Douglas Souza

---

## 3.1 Objetivo
Segmentar batimentos em janelas de 1000ms (500 amostras @ 500Hz), detectar picos R via AMPT, extrair features temporais e morfológicas com fundamento clínico, aplicar augmentation controlado e balanceamento via SMOTE no espaço de features, garantindo rastreabilidade de linhagem e reprodutibilidade do pipeline.

---

## 3.2 Decisão Arquitetural: Janela de 1000ms

A literatura de deep learning para ECG adota janelas variadas conforme a arquitetura:

| Arquitetura | Janela | Amostras @ 500Hz | Contexto | Referência |
| :--- | :--- | :--- | :--- | :--- |
| CNN simples | 360ms | 180 | 1 batimento centrado | Yan et al. (SNN+CNN)citeweb_search:12#10 |
| CNN+LSTM | 2400ms | 1200 | 3 batimentos (contexto) | Deep CNN ECG beats (PMC)citeweb_search:12#8 |
| Waveform Segmentation | 2000ms | 1000 | P+QRS+T completas | Boston Scientific (HRS 2021)citeweb_search:11#6 |
| **Project-Lewis (padrão)** | **1000ms** | **500** | 1 batimento + metade do anterior/posterior | Escolha arquitetural |
| **Project-Lewis (fallback)** | **600ms** | **300** | 1 batimento (batimentos curtos / taquicardia) | Escolha arquitetural |

**Justificativa:** 1000ms captura o batimento alvo, a onda T subsequente e parte do batimento anterior, fornecendo contexto temporal suficiente para modelos CNN/Transformer sem o custo computacional de 2.4s. Em taquicardia (RR < 600ms), a janela de 600ms é usada para evitar overlap de múltiplos batimentos. A janela é sempre centrada no pico R.

---

## 3.3 Segmentação

### src/data/segmenter.py
```python
class ECGSegmenter:
    def __init__(self, fs: float = 500.0, window_ms: float = 1000.0, min_window_ms: float = 600.0):
        self.fs = fs
        self.half_len = int((window_ms * fs) / 2000)  # 250 amostras para 1000ms
        self.min_half_len = int((min_window_ms * fs) / 2000)  # 150 amostras para 600ms
        self.window_len = 2 * self.half_len + 1  # 501 amostras (ímpar, R no centro)

    def segment_with_labels(self, sig, r_peaks, labels, rr_intervals_ms) -> Tuple[np.ndarray, np.ndarray]:
        """Segmentar batimentos em janelas centradas no R-peak.

        1. Para cada batimento i:
           a. Calcular RR_interval atual (ms)
           b. Se RR_interval < 600ms: usar min_window (600ms) para evitar overlap
           c. Senão: usar window padrão (1000ms)
           d. Verificar se há amostras suficientes antes e depois do R-peak
           e. Se bordas insuficientes: descartar (nunca padding com zeros)
        2. Retornar X shape (n_segments, window_len) float32, y shape (n_segments,)
        3. Logar: n_descartados_bordas, n_usados_600ms, n_usados_1000ms
        """
```

> **Regra de ouro:** Padding com zeros é proibido. Zeros introduzem descontinuidades artificiais que confundem filtros e CNNs. Batimentos nas bordas do registro são descartados.

---

## 3.4 Detecção QRS — AMPT @ 500Hz

### Correção Crítica: Parâmetros do AMPT

O documento v1.0 afirmava que AMPT usa "banda 5-25 Hz (estendida vs 5-15 Hz clássico)". **Isso está incorreto.** O AMPT (AccYouRate Modified Pan-Tompkins) utiliza os **mesmos filtros** do Pan-Tompkins original (5–15 Hz), mas simplifica a fase de decisão:

1. Remove a análise dupla de sinais (bandpass + filtered signal); analisa apenas o sinal filtrado final.
2. Elimina o segundo RR average (para arritmias) e usa apenas um RR average1 para search-back.
3. Redefine os thresholds SPKF, NPKF, THRESHOLDF1 e THRESHOLDF2 com coeficientes simplificados.citeweb_search:12#5

A banda 5–18 Hz é do **Pan-Tompkins++** (arXiv 2024), não do AMPT.citeweb_search:11#0

### src/features/ampt_500hz.py
```python
class AMPTDetector:
    """AccYouRate Modified Pan-Tompkins (Neri et al., 2023).

    Referência: https://github.com/Accyourate-Group-S-p-A/acy_ampt
    """
    def __init__(self, fs: float = 500.0):
        self.fs = fs
        # Bandpass: 5-15 Hz (MESMO do Pan-Tompkins original)
        self.b_band, self.a_band = scipy.signal.butter(
            2, [5.0/(fs/2), 15.0/(fs/2)], btype="band"
        )
        # Derivative: 5-point (Pan-Tompkins standard)
        # Squaring: point-by-point
        # Moving Window Integration: 150ms = 75 amostras @ 500Hz
        self.mwi_window = int(0.150 * fs)
        # Refratariedade / T-wave discrimination: 360ms = 180 amostras @ 500Hz
        self.refractory = int(0.360 * fs)
        # Search-back missed limit: 1.66 * RR_average1
        # Learning phase: 2s inicial

    def detect(self, sig: np.ndarray) -> np.ndarray:
        # 1. Bandpass filter (filtfilt para zero-phase)
        # 2. 5-point derivative
        # 3. Squaring
        # 4. Moving average integration (75 amostras, rectangular window)
        # 5. Adaptive thresholding (SPKF, NPKF, THRESHOLDF1, THRESHOLDF2)
        # 6. T-wave discrimination (slope check within 360ms window)
        # 7. Search-back for missed QRS (1.66 * RR_average1)
        # Retornar: np.ndarray de índices de R-peaks

    def evaluate(self, sig, r_true, tol_ms: float = 150.0) -> dict:
        """Avaliar AMPT contra ground truth (.atr).

        tol_ms: tolerância padrão de 150ms para detecção de batimento
                (equivale a 75 amostras @ 500Hz)
        """
        tol_samples = int(tol_ms * self.fs / 1000.0)
        # TP: predição dentro de tol_samples de um true não-matched
        # FN: true não-matched
        # FP: predição não-matched
        # Sens = TP / (TP + FN)
        # PPV = TP / (TP + FP)
        # Retornar dict com TP, FN, FP, Sens, PPV, n_pred, n_true
```

**Quality Gate QG2:**

| Dataset | Sensibilidade | PPV | F1 | Fonte Esperada |
| :--- | :--- | :--- | :--- | :--- |
| MIT-BIH (High Quality) | > 99.5% | > 99.7% | > 99.6% | AMPT: Sens 99.80%, PPV 99.82%citeweb_search:12#5 |
| MIT-BIH (Arrhythmias) | > 96.0% | > 99.5% | > 97.5% | AMPT: Sens 96.80%, PPV 99.83%citeweb_search:12#5 |
| MIT-BIH (Paced) | > 94.0% | > 94.0% | > 94.0% | AMPT: Sens 96.90%, PPV 95.07%citeweb_search:12#5 |
| FP global | < 1% do total | — | — | Pan-Tompkins: < 1% FP |

> **Nota:** A tolerância de 150ms (75 amostras) é o padrão AAMI/PhysioNet para avaliação de QRS detectors. Uma tolerância de 3 amostras (6ms) é usada apenas para validação de precisão de localização do pico R, não para detecção de batimento.

---

## 3.5 Features Temporais

### src/features/time_domain.py
```python
class TimeDomainFeatures:
    def extract(self, r_peaks: np.ndarray, fs: float = 500.0) -> List[Dict]:
        """Extrair features temporais baseadas em intervalos RR.

        rr_intervals = np.diff(r_peaks) / fs * 1000.0  # ms
        Para cada batimento i:
          rr_prev = rr_ms[i-1] (0 se i==0)
          rr_next = rr_ms[i] (0 se último)
          rr_ratio = rr_prev / rr_next (1.0 se inválido)
          rr_local_mean = np.mean(rr_ms[max(0,i-2):i+3])  # 5 batimentos
          rr_local_std = np.std(rr_ms[max(0,i-2):i+3])
          rmssd = np.sqrt(np.mean(np.diff(rr_ms[max(0,i-4):i+1])**2))
          heart_rate = 60000.0 / rr_prev  # BPM
        """
```

| Feature | Descrição | Relevância Clínica | Unidade |
| :--- | :--- | :--- | :--- |
| `rr_prev` | Intervalo RR anterior | Bradicardia / Taquicardia | ms |
| `rr_next` | Intervalo RR posterior | Arritmia irregular | ms |
| `rr_ratio` | rr_prev / rr_next | Prematuroidade (PVC: ratio >> 1) | adimensional |
| `rr_local_mean` | Média RR em janela de 5 batimentos | Tendência de ritmo | ms |
| `rr_local_std` | Desvio padrão RR local | Instabilidade de ritmo | ms |
| `rmssd` | Root mean square of successive differences | HRV (parasimpática) | ms |
| `heart_rate` | Frequência cardíaca instantânea | Taquicardia / Bradicardia | BPM |

---

## 3.6 Features Morfológicas

### Fundamentação Clínica

A extração de features morfológicas segue padrões clínicos estabelecidos:

- **QRS Width:** Medida entre o onset (início da onda Q) e o offset (fim da onda S). O método padrão é o threshold a 50% da amplitude do pico R no envelope do sinal filtrado.citeweb_search:13#3 Janela de busca: 300ms antes do R para onset, 150ms após o R para offset.citeweb_search:13#11
- **ST Slope:** Medido a 60ms após o J-point (junção QRS-ST), que é o ponto padrão ACC/AHA/HRS para detecção de isquemia.citeweb_search:13#7citeweb_search:13#8 A medição entre J+60ms e J+80ms maximiza a capacidade discriminativa para doença coronariana.citeweb_search:13#7

### src/features/morphological.py
```python
class MorphologicalFeatures:
    def extract(self, segments: np.ndarray, fs: float = 500.0, r_idx: int = None) -> list:
        """Extrair features morfológicas de segmentos ECG.

        segments: (n_segments, window_len) — já centrados no R-peak
        r_idx: índice do pico R no segmento (default: argmax(abs(seg)))

        Para cada segmento:
          1. r_idx = np.argmax(np.abs(seg))  # aproximação pico R
          2. r_amplitude = seg[r_idx]  # mV

          3. QRS onset (busca 300ms antes do R):
             - Envelope do sinal filtrado bandpass [5, 30] Hz
             - Threshold = 50% de |r_amplitude|
             - Onset = último ponto antes de r_idx onde envelope < threshold

          4. QRS offset (busca 150ms após o R):
             - Mesmo threshold
             - Offset = primeiro ponto após r_idx onde envelope < threshold

          5. qrs_width_ms = (offset - onset) / fs * 1000.0

          6. q_depth = np.min(seg[max(0, r_idx-50):r_idx])  # 100ms antes
          7. t_amplitude = np.max(seg[r_idx:min(len(seg), r_idx+150)])  # 300ms depois

          8. J-point = offset (fim do QRS)
          9. st_start = J-point + int(0.060 * fs)  # 60ms após J
          10. st_end = J-point + int(0.080 * fs)   # 80ms após J
          11. st_slope = np.polyfit(range(st_start, st_end), seg[st_start:st_end], 1)[0]
              # slope em mV/amostra; converter para mV/s multiplicando por fs

          12. qrs_area = np.trapezoid(np.abs(seg[onset:offset]), dx=1.0/fs)
        """
```

| Feature | Descrição | Relevância Clínica | Unidade |
| :--- | :--- | :--- | :--- |
| `r_amplitude` | Pico máximo do segmento | Amplitude relativa | mV |
| `q_depth` | Mínimo antes do R (ond Q) | Infarto anterior | mV |
| `t_amplitude` | Máximo após o R (onda T) | Repolarização anormal | mV |
| `qrs_width_ms` | Largura do complexo QRS (onset→offset @ 50% amp) | Bloqueio de ramo / PVC / Hipertrófia | ms |
| `qrs_area` | Área sob a curva do QRS | Energia do batimento | mV·s |
| `st_slope_mV_s` | Inclinação do segmento ST (J+60ms → J+80ms) | Isquemia / STEMI | mV/s |
| `j_point` | Offset do QRS (junção QRS-ST) | Referência para ST | amostra |

> **Caveat:** A detecção automática de onset/offset do QRS é sensível a ruído. Em caso de falha (onset >= offset), marcar `qrs_width_ms = np.nan` e excluir do treinamento de modelos tabulares, mas manter para CNNs (que não dependem dessa feature).

---

## 3.7 Data Augmentation

**Aplicável APENAS no fine-tuning (MIT-BIH+). NUNCA no pré-treino (Chapman) ou teste.**

As técnicas de augmentation para ECG são bem estabelecidas na literatura.citeweb_search:11#13 Os parâmetros são calibrados para não distorcer a morfologia clínica:

### src/features/augmentation.py
```python
class ECGAugmenter:
    def jitter(self, x: np.ndarray, std_factor: float = 0.01) -> np.ndarray:
        """Adicionar ruído Gaussiano proporcional ao desvio padrão do sinal."""
        noise = np.random.normal(0, std_factor * np.std(x), size=x.shape)
        return x + noise

    def baseline_wander(self, x: np.ndarray, fs: float = 500.0) -> np.ndarray:
        """Adicionar senoide de baixa frequência (0.05–0.5 Hz) simulando respiração."""
        t = np.arange(len(x)) / fs
        freq = np.random.uniform(0.05, 0.5)  # Hz
        phase = np.random.uniform(0, 2 * np.pi)
        amplitude = np.random.uniform(0.05, 0.2)  # mV
        wander = amplitude * np.sin(2 * np.pi * freq * t + phase)
        return x + wander

    def powerline_noise(self, x: np.ndarray, fs: float = 500.0, freq: float = 60.0) -> np.ndarray:
        """Adicionar interferência de rede (50 Hz ou 60 Hz) + harmônicos."""
        t = np.arange(len(x)) / fs
        amplitude = np.random.uniform(0.02, 0.05)  # mV
        harmonic = np.random.choice([1, 2, 3])  # 1º, 2º ou 3º harmônico
        noise = amplitude * np.sin(2 * np.pi * freq * harmonic * t)
        return x + noise

    def time_warp(self, x: np.ndarray, max_stretch: float = 0.05) -> np.ndarray:
        """Esticar/comprimir levemente o sinal ao longo do tempo (DTW-like)."""
        from scipy.interpolate import interp1d
        stretch = np.random.uniform(1 - max_stretch, 1 + max_stretch)
        old_len = len(x)
        new_len = int(old_len * stretch)
        f = interp1d(np.arange(old_len), x, kind='cubic', fill_value='extrapolate')
        x_warped = f(np.linspace(0, old_len - 1, new_len))
        # Reamostrar para o tamanho original
        f2 = interp1d(np.arange(new_len), x_warped, kind='cubic', fill_value='extrapolate')
        return f2(np.linspace(0, new_len - 1, old_len))

    def apply(self, x: np.ndarray, fs: float = 500.0, p: float = 0.5) -> np.ndarray:
        """Aplicar cada augmentação com probabilidade p, em sequência."""
        if np.random.rand() < p:
            x = self.jitter(x)
        if np.random.rand() < p:
            x = self.baseline_wander(x, fs)
        if np.random.rand() < p:
            x = self.powerline_noise(x, fs)
        if np.random.rand() < p:
            x = self.time_warp(x)
        return x
```

> **Regras de augmentation:**
> - Aplicar APENAS no conjunto de treinamento do fine-tuning.
> - Nunca aplicar no pré-treino (Chapman) — o backbone deve aprender representações limpas.
> - Nunca aplicar no conjunto de teste/validação.
> - Parâmetros de amplitude são limitados para preservar a morfologia clínica (ex: baseline wander < 0.2 mV, que é o limite fisiológico normal).

---

## 3.8 Balanceamento de Classes — SMOTE no Espaço de Features

O MIT-BIH é severamente desbalanceado: classe N domina (~75%), enquanto F e Q são minoritárias (< 1% cada). SMOTE (Synthetic Minority Oversampling Technique) deve ser aplicado **no espaço de features extraídas**, nunca no sinal bruto.citeweb_search:11#4citeweb_search:12#9

### src/features/balancer.py
```python
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

class ECGBalancer:
    def __init__(self, strategy: str = "smote+rus", random_state: int = 42):
        self.strategy = strategy
        self.random_state = random_state

    def balance(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Balancear dataset de features (n_samples, n_features).

        Estratégias:
        - "smote": SMOTE puro (oversampling minoritárias até igualar majoritária)
        - "smote+rus": SMOTE + RandomUnderSampler (oversampling + undersampling)
        - "adasyn": ADASYN (foco em amostras difíceis)

        Retornar: X_balanced, y_balanced
        """
        if self.strategy == "smote":
            sampler = SMOTE(random_state=self.random_state, k_neighbors=5)
        elif self.strategy == "smote+rus":
            sampler = ImbPipeline([
                ("over", SMOTE(random_state=self.random_state, k_neighbors=5)),
                ("under", RandomUnderSampler(random_state=self.random_state))
            ])
        elif self.strategy == "adasyn":
            from imblearn.over_sampling import ADASYN
            sampler = ADASYN(random_state=self.random_state, n_neighbors=5)
        else:
            raise ValueError(f"Estratégia desconhecida: {self.strategy}")

        return sampler.fit_resample(X, y)
```

> **Aviso:** SMOTE no sinal bruto (waveform) gera artefatos sintéticos não-fisiológicos que degradam a performance de CNNs. A literatura majoritária concorda: SMOTE no espaço de features para classificadores tradicionais (SVM, XGBoost, RF); augmentation no sinal bruto para CNNs.citeweb_search:12#5

---

## 3.9 Mapeamento AAMI EC57

O mapeamento segue o padrão ANSI/AAMI EC57:1998, conforme implementação de referência `bxb` da PhysioNet.citeweb_search:8#5 (Ver Camada 2 para tabela completa.)

### src/features/aami_mapper.py
```python
AAMI_MAP = {
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
    "V": "V", "E": "V",
    "A": "S", "a": "S", "J": "S", "S": "S",
    "F": "F",
    "/": "Q", "f": "Q", "Q": "Q", "|": "Q",
}

AAMI_CLASSES = ["N", "S", "V", "F", "Q"]

AAMI_DESCRIPTION = {
    "N": "Normal / Bundle branch block / Escape",
    "S": "Supraventricular ectopic",
    "V": "Ventricular ectopic",
    "F": "Fusion beat",
    "Q": "Paced / Unclassifiable / Artifact",
}

def map_annotations(symbols: List[str]) -> Tuple[List[str], Dict]:
    """Mapear símbolos WFDB para classes AAMI.

    Retorna: (labels_aami, stats)
    stats: {n_total, n_unmapped, n_by_class, n_by_symbol}
    """
```

---

## 3.10 Linhagem de Features

Cada batimento segmentado gera um registro de linhagem em `data/lineage/features/{dataset}/{record_id}_{beat_idx}.json`:

```json
{
  "record_id": "100",
  "beat_idx": 42,
  "dataset": "mitdb",
  "segment": {
    "window_ms": 1000,
    "fs": 500,
    "r_peak_sample": 250,
    "start_sample": 0,
    "end_sample": 500
  },
  "ampt": {
    "r_peak_detected": 250,
    "detection_error_ms": 0,
    "tol_ms": 150
  },
  "features": {
    "temporal": {"rr_prev": 856.0, "rr_next": 840.0, "rr_ratio": 1.02, "heart_rate": 70.1},
    "morphological": {"r_amplitude": 1.23, "qrs_width_ms": 92.0, "st_slope_mV_s": -0.45, "qrs_area": 0.087}
  },
  "label": {"wfdb": "N", "aami": "N"},
  "augmentation": {"applied": false, "methods": []},
  "preprocess_config": "config/preprocess_v1.0.yaml",
  "timestamp": "2026-06-09T22:15:00Z"
}
```

---

## 3.11 Quality Gate QG3

| Critério | Valor | Como Validar |
| :--- | :--- | :--- |
| Janela de segmentação | 1000ms (600ms fallback) | `pytest tests/test_segmenter.py` |
| Sem padding com zeros | 0 ocorrências | `assert not np.any(seg == 0.0)` para bordas artificiais |
| AMPT Sensibilidade (MIT-BIH) | > 96.5% | `pytest tests/test_ampt.py` vs .atr (tol=150ms) |
| AMPT PPV (MIT-BIH) | > 99.0% | `pytest tests/test_ampt.py` |
| AMPT F1 (MIT-BIH) | > 97.5% | `pytest tests/test_ampt.py` |
| Features por batimento | >= 10 dimensões | `len(df.columns) >= 10` |
| Sem NaN/Inf | 0 ocorrências | `df.isnull().sum().sum() == 0` |
| Range fisiológico | `rr_prev` > 0, `qrs_width_ms` ∈ [40, 200] | `assert` por coluna |
| QRS width válido | > 95% dos batimentos | `np.isnan(qrs_width).sum() / len(qrs_width) < 0.05` |
| SMOTE aplicado | Apenas em features, nunca em sinal bruto | `pytest tests/test_smote.py` |
| Augmentation | Apenas treino fine-tuning, nunca teste/pré-treino | `pytest tests/test_augmentation.py` |
| Linhagem | 100% dos batimentos | `len(lineage_dir.glob("*.json")) == n_beats` |

**Teste:** `pytest tests/test_feature_engineering.py -v`

---

## 3.12 Referências Verificadas

- AMPT (AccYouRate Modified Pan-Tompkins): Neri et al., Sensors 2023, 23(3), 1625. https://pmc.ncbi.nlm.nih.gov/articles/PMC9920820/ — Simplificação do Pan-Tompkins com mesmos filtros (5–15 Hz), removendo análise dupla de sinais e search-back duplo. F1 99.62% (High Quality), 96.66% (Arrhythmias).citeweb_search:12#5
- AMPT GitHub: https://github.com/Accyourate-Group-S-p-A/acy_ampt — Implementação Python oficial.citeweb_search:12#6
- Pan-Tompkins++ (arXiv 2024): https://arxiv.org/html/2211.03171v3 — Banda 5–18 Hz (NÃO é AMPT), flattop window 60ms, lower threshold 0.4× higher threshold.citeweb_search:11#0
- Pan-Tompkins Original (IEEE TBME 1985): https://www.robots.ox.ac.uk/~gari/teaching/cdt/A3/readings/ECG/Pan+Tompkins.pdf — Banda 5–15 Hz, MWI 150ms, refratariedade 200ms.citeweb_search:11#10
- ECG Segmentation 2.4s (PMC): https://pmc.ncbi.nlm.nih.gov/articles/PMC8155180/ — 865 amostras @ 360Hz para CNN+LSTM com 3 batimentos.citeweb_search:12#8
- ECG Segmentation 2s (Boston Scientific): https://www.bostonscientific.com/content/dam/bostonscientific/ep/general/news---events/hrs/2021/Deep%20Learning%20for%20ECG%20Waveform%20Segmentation_withBS.pdf — 2s windows para P/QRS/T segmentation.citeweb_search:11#6
- SMOTE no espaço de features (MDPI): https://www.mdpi.com/2673-7426/6/3/33 — "SMOTE operates in the extracted feature space rather than on the raw ECG signal".citeweb_search:11#4
- ECG Augmentation Survey (PMC): https://pmc.ncbi.nlm.nih.gov/articles/PMC10256074/ — Jitter, baseline wander (0.05–0.5 Hz), powerline (50/60 Hz), time warping.citeweb_search:11#13
- QRS Width / Onset-Offset (CiC 2008): https://cinc.org/archives/2008/pdf/0857.pdf — Envelope method, search window 300ms (onset) / 150ms (offset).citeweb_search:13#11
- ST Slope Measurement (Int J Cardiol 1997): https://www.sciencedirect.com/science/article/abs/pii/S0167527397001575 — J+60ms é o ponto ótimo para discriminação de doença coronariana.citeweb_search:13#7
- ST Segment / J-Point (NCBI StatPearls): https://www.ncbi.nlm.nih.gov/books/NBK459364/ — "ST segment is frequently evaluated at ST, which is the ST segment at 60 ms after the J point."citeweb_search:13#8
