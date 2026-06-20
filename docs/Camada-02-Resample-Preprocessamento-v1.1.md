# Project-Lewis — Camada 2: Resample, Lead Selection e Pré-Processamento
## Responsável: Engenharia de Dados + DSP

**Versão:** 1.1 | **Data:** 2026-06-09 | **Arquiteto:** Douglas Souza

---

## 2.1 Objetivo
Unificar frequências de amostragem (→ 500 Hz), selecionar lead equivalente a MLII, filtrar ruído, remover tendência, normalizar sinais e persistir metadados de linhagem para consumo do modelo, garantindo reprodutibilidade, rastreabilidade e idempotência do pipeline.

---

## 2.2 Decisão Arquitetural: 500 Hz

Todos os datasets são resampleados para **500 Hz** (frequência nativa do Chapman-Shaoxing e do ADS1292R no firmware). A escolha de 500 Hz como denominador comum é tecnicamente justificada: captura ondas P, QRS e T com fidelidade (Nyquist = 250 Hz), é compatível com o hardware alvo e simplifica o pipeline unificado.citeweb_search:7#0

| Dataset | Fs Original | Resample | up | down | Método | Nota Técnica |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| MIT-BIH | 360 Hz | 500 Hz | 25 | 18 | `scipy.signal.resample_poly` | Fração irredutível; FIR interno com atenuação > 40 dB na banda de rejeição |
| SVDB | **250 Hz** | 500 Hz | 2 | 1 | `scipy.signal.resample_poly` | **Correção:** Fs original é 250 Hz (valor documentado na v1.0 estava incorreto)citeweb_search:9#7 |
| AFDB | 250 Hz | 500 Hz | 2 | 1 | `scipy.signal.resample_poly` | Upsampling trivial por interpolação FIR |
| INCART | 257 Hz | 500 Hz | 500 | 257 | `scipy.signal.resample_poly` | Fração grande; validar custo computacional do FIR |
| Chapman | 500 Hz | — | — | — | Nativo | Nenhuma transformação de taxa |

> **Anti-aliasing:** `resample_poly` aplica filtro FIR interno automaticamente. Para ECG — sinal não-periódico — `resample_poly` é preferível a `resample` (FFT-based), que assume periodicidade e pode introduzir artefatos de borda.citeweb_search:8#0 Em todos os casos, validar via espectrograma que componentes > 250 Hz (Nyquist pós-resample) são atenuados > 40 dB.

---

## 2.3 Lead Selection

A seleção de lead único é necessária para compatibilizar datasets multi-lead com a arquitetura do modelo single-lead (MLII-equivalente). A escolha segue o princípio da **maximização da correspondência anatômica** com o lead MLII (bipolar: RA–LL).

| Dataset | Lead Selecionado | Índice | Justificativa Anatômica |
| :--- | :--- | :--- | :--- |
| Chapman (12-lead) | **II** | `lead_names.index("II")` | Lead bipolar RA–LL, anatomicamente idêntico a MLII |
| MIT-BIH (2-lead) | **MLII** | `0` | Lead padrão do dataset; já é MLII |
| SVDB (2-lead) | **ECG1** | `0` | Lead 0 do dataset; nomenclatura genérica (ECG1/ECG2) sem documentação oficial de equivalência anatômicaciteweb_search:8#9 |
| AFDB (2-lead) | **ECG1** | `0` | Lead 0 do dataset; mesmo caveat do SVDB |
| INCART (12-lead) | **II** | `lead_names.index("II")` | Lead bipolar RA–LL, mais próximo anatomicamente de MLII do que Lead I (RA–LA) |

> **Aviso de proveniência:** A equivalência exata ECG1 ↔ MLII no SVDB/AFDB não é documentada oficialmente pela PhysioNet.citeweb_search:8#9 O mapeamento para índice 0 é uma convenção adotada pela literatura majoritária (incluindo Kaggle datasets derivados).citeweb_search:8#7 Para máxima rigorosidade, recomenda-se validar a morfologia QRS de ECG1 contra padrões conhecidos de MLII em amostra piloto antes do treinamento.

---

## 2.4 Parâmetros de Pré-Processamento Versionados

Todo pré-processamento deve ser determinístico e reprodutível. Os parâmetros são versionados em `config/preprocess_v{major}.{minor}.yaml`:

```yaml
preprocess_v1.0:
  resample:
    target_fs: 500.0
    method: "resample_poly"
    padtype: "line"  # zero-padding para ECG não-periódico
  filter:
    type: "butterworth"
    order: 4
    lowcut: 0.5   # Hz — remove baseline wander (respiração, movimento)
    highcut: 40.0 # Hz — remove EMG e ruído de alta frequência
    implementation: "filtfilt"  # zero-phase, sem latência
  detrend:
    type: "linear"  # scipy.signal.detrend
  normalization:
    type: "zscore_global"
    eps: 1.0e-12
    per_record: false  # global sobre todo o dataset; true = por registro
  loader:
    default_gain: 200.0      # counts/mV (MIT-BIH padrão)
    default_baseline: 1024.0 # ADC zero (MIT-BIH 11-bit)
    physical_unit: "mV"
    validate_range: [-5.0, 5.0]  # mV
```

> **Fundamentação clínica:** O bandpass 0.5–40 Hz é o padrão ACC/AHA/HRS para ECG digital de adultos.citeweb_search:7#6 O filtro high-pass a 0.5 Hz remove baseline wander induzido por respiração e movimento do paciente sem distorcer o segmento ST. O filtro low-pass a 40 Hz preserva o complexo QRS enquanto atenua artefatos musculares (EMG) e interferência de alta frequência.

---

## 2.5 Módulos

### src/data/resampler.py
```python
def resample_to_500hz(sig: np.ndarray, fs_orig: float, padtype: str = "line") -> np.ndarray:
    """Resample sinal para 500 Hz usando scipy.signal.resample_poly.

    1. Calcular fração reduzida up/down = Fraction(500, fs_orig).limit_denominator(1000)
    2. Aplicar scipy.signal.resample_poly(sig, up, down, padtype=padtype)
       - padtype="line" para ECG não-periódico (evita artefatos de wrap)
    3. Validar: espectrograma Welch mostra atenuação > 40 dB acima de 250 Hz
    4. Logar metadados: fs_orig, up, down, padtype, len_in, len_out
    5. Retornar float64, len_out = len(sig) * (500/fs_orig) arredondado
    """
```

### src/data/lead_selector.py
```python
def select_lead(record, dataset_name: str) -> Tuple[np.ndarray, str]:
    """Extrair lead específico de registro multi-lead.

    Mapeamento:
    - Chapman: lead "II" (bipolar RA-LL, equivalente MLII)
    - MIT-BIH: lead "MLII" (índice 0)
    - SVDB: lead "ECG1" (índice 0; caveat: equivalência não documentada)
    - AFDB: lead "ECG1" (índice 0; caveat: equivalência não documentada)
    - INCART: lead "II" (bipolar RA-LL, mais próximo de MLII)

    Retornar: (np.ndarray 1D em mV, lead_name)
    """
```

### src/data/loader.py
```python
class MITBIHLoader:
    """Loader unificado para MIT-BIH family + Chapman.

    NÃO assume ganho/baseline fixos. Sempre lê do .hea via wfdb.rdheader()
    e aplica: physical = (digital - baseline) / gain.
    """
    FS_TARGET = 500.0  # Hz PADRÃO APÓS RESAMPLE

    def load_signal(record_path: Path, channel: int = 0, units: str = "physical") -> np.ndarray:
        # Usar wfdb.rdrecord(..., channels=[channel], units=units)
        # Se units="physical", wfdb já converte para mV usando adc_gain e baseline do .hea
        # Se units="digital", retorna ADC counts
        # Validar: se physical, assert range [-5, +5] mV
        # Emitir WARN se adc_gain ou baseline divergirem drasticamente dos padrões MIT-BIH

    def load_annotations(record_path: Path) -> Tuple[np.ndarray, np.ndarray, dict]:
        # Retorna (r_samples, aami_labels, metadata)
        # Mapear símbolos WFDB → AAMI EC57 test labels (N, V, F, Q, O, X, S)
        # Filtrar apenas batimentos (beat annotations)
        # Remover: rhythm changes ('+'), signal quality ('~'), non-beat markers
        # metadata: total_beats, noise_segments, paced_ratio

    def get_record_names(raw_dir: Path) -> List[str]:
        # Listar stems de .hea. Esperado >= 226 registros (MIT-BIH family)
```

### src/data/preprocessor.py
```python
class ECGPreprocessor:
    def __init__(self, config_path: Path):
        self.cfg = load_yaml(config_path)
        self.fs = self.cfg["filter"]["target_fs"]  # 500.0
        self.lowcut = self.cfg["filter"]["lowcut"]   # 0.5
        self.highcut = self.cfg["filter"]["highcut"] # 40.0
        self.order = self.cfg["filter"]["order"]     # 4
        # Pré-computar coeficientes Butterworth para evitar recálculo por registro
        nyq = self.fs / 2.0
        self.b_band, self.a_band = scipy.signal.butter(
            self.order, [self.lowcut/nyq, self.highcut/nyq], btype="band"
        )

    def filter(self, x: np.ndarray) -> np.ndarray:
        # scipy.signal.filtfilt(self.b_band, self.a_band, x) — zero-phase, sem latência
        # filtfilt processa forward + backward, cancelando fase

    def detrend(self, x: np.ndarray) -> np.ndarray:
        # scipy.signal.detrend(x, type="linear") — remove drift DC linear

    def normalize(self, x: np.ndarray) -> np.ndarray:
        # Z-score global: (x - mean) / (std + eps)
        # Alternativa: min-max por registro se cfg["normalization"]["per_record"]=true

    def process(self, x: np.ndarray, record_id: str) -> Tuple[np.ndarray, dict]:
        # 1. Resample (se fs != 500)
        # 2. filter → detrend → normalize
        # 3. Gerar metadata: input_range_mV, output_range_mV, mean, std, duration_sec
        # 4. Retornar (x_processed, metadata)
```

### src/data/segmenter.py

```python
class ECGSegmenter:
    """Segmenta batimentos em janelas centradas nos R-peaks.

    Parâmetros padrão:
    - fs = 500 Hz
    - window_ms = 1000 ms → half_len = 250 → window_len = 2*half_len + 1 = 501
    - min_window_ms = 600 ms → half_len = 150 → min_window_len = 301
    """
```

> **Decisão arquitetural pendente — input shape 500 vs 501:**
> O segmentador atual gera janelas ímpares (501 amostras) para manter o R-peak
> exatamente no centro. O modelo 1D-CNN, porém, espera `(500, 1)`.
> Recomenda-se padronizar o pipeline em **500 amostras**: alterar o segmentador
> para `window_len = 2 * half_len` (sem o `+1`), resultando em 500 amostras para
> 1000 ms @ 500 Hz. Isso evita reshape/adaptação no modelo e mantém a
> compatibilidade com a especificação da Camada 4. A implementação efetiva é
> responsabilidade da frente `input_shape`.

---

## 2.6 Mapeamento AAMI EC57

O padrão ANSI/AAMI EC57:1998 define os rótulos de teste para avaliação de algoritmos de detecção de arritmia. O mapeamento WFDB → AAMI é obrigatório para garantir interoperabilidade com métricas da literatura.citeweb_search:8#3citeweb_search:8#5

| WFDB Symbol | AAMI Test Label | Descrição | Incluir no Treino? |
| :--- | :--- | :--- | :--- |
| N | N | Normal beat | Sim |
| L | N | Left bundle branch block | Sim |
| R | N | Right bundle branch block | Sim |
| e | N | Atrial escape beat | Sim |
| j | N | Nodal (junctional) escape beat | Sim |
| V | V | Premature ventricular contraction | Sim |
| E | V | Ventricular escape beat | Sim |
| A | S | Atrial premature contraction | Sim |
| a | S | Aberrated atrial premature beat | Sim |
| J | S | Nodal (junctional) premature beat | Sim |
| S | S | Premature/ectopic supraventricular beat | Sim |
| F | F | Fusion of ventricular and normal beat | Sim |
| / | Q | Paced beat | Opcional (excluir se modelo não suportar pacing) |
| f | Q | Fusion of paced and normal beat | Opcional |
| Q | Q | Unclassifiable beat | Sim (classe residual) |
| ~ | — | Signal quality change | **Não** (non-beat) |
| + | — | Rhythm change | **Não** (non-beat) |
| &#124; | — | Isolated QRS-like artifact | **Não** (non-beat) |
| x | — | Non-conducted P-wave | **Não** (non-beat) |

> **Regra de filtragem:** Apenas anotações do tipo `beat` (códigos 0–29 no formato MIT) são incluídas. Anotações de ritmo, qualidade de sinal e artefatos são descartadas para o treinamento de classificação de batimentos, mas podem ser preservadas para análise de qualidade do sinal (QG2).

---

## 2.7 Linhagem, Rastreabilidade e Metadados

Cada registro processado gera um arquivo de linhagem JSON em `data/lineage/{dataset}/{record_id}.json`:

```json
{
  "record_id": "100",
  "dataset": "mitdb",
  "version": "1.0.0",
  "raw_checksum": "sha256:abc123...",
  "preprocess_config": "config/preprocess_v1.0.yaml",
  "pipeline": [
    {"step": "load", "fs_orig": 360, "gain": 200, "baseline": 1024, "lead": "MLII", "unit": "mV"},
    {"step": "resample", "method": "resample_poly", "up": 25, "down": 18, "fs_out": 500},
    {"step": "filter", "type": "butterworth", "order": 4, "bandpass": [0.5, 40.0], "implementation": "filtfilt"},
    {"step": "detrend", "type": "linear"},
    {"step": "normalize", "type": "zscore_global", "mean": -0.003, "std": 0.847}
  ],
  "output": {
    "path": "data/processed/mitdb/100_II.npy",
    "shape": [540000, 1],
    "dtype": "float64",
    "duration_sec": 1080.0,
    "range_mV": [-4.12, 3.89]
  },
  "quality_gate": {"QG1_pass": true, "QG2_pass": null},
  "timestamp": "2026-06-09T22:15:00Z"
}
```

**Benefícios:**
- **Reprodutibilidade:** qualquer registro pode ser reconstruído exatamente a partir do raw + config.
- **Auditoria:** permite identificar qual passo introduziu artefatos em caso de regressão.
- **Idempotência:** se `lineage.json` existe e `raw_checksum` não mudou, pular reprocessamento.

---

## 2.8 Dead Letter Queue (DLQ) para Pré-Processamento

Registros que falham em qualquer etapa do pré-processamento são enfileirados em `data/.dlq/preprocess_failures.jsonl`:

```python
{
  "record_id": "203",
  "dataset": "mitdb",
  "step": "resample",
  "error": "ValueError: up*down too large for resample_poly",
  "traceback": "...",
  "raw_path": "data/raw_mitbih/203.hea",
  "timestamp": "2026-06-09T22:15:00Z"
}
```

**Regras:**
- Falha em `resample` → geralmente sinal muito curto ou fs inconsistente; requer análise manual.
- Falha em `filter` → possível NaN/Inf no sinal; verificar integridade do raw.
- Falha em `normalize` → std = 0 (sinal flatline); marcar para exclusão do treinamento.
- DLQ é consumida diariamente por job de reprocessamento com config alternativa.

---

## 2.9 Quality Gate QG1

| Critério | Valor | Como Validar |
| :--- | :--- | :--- |
| Resample 360→500 | RMSE < 1e-6 vs referência `resample` (FFT) | `pytest tests/test_resampler.py` sobre segmento sintético de 10 s |
| Resample 250→500 | RMSE < 1e-6 vs referência | Idem para SVDB/AFDB |
| Resample 257→500 | RMSE < 1e-6 vs referência | Idem para INCART |
| Ganho ADC | Lido do .hea (não hardcoded) | `wfdb.rdheader().adc_gain[0]` |
| Baseline ADC | Lido do .hea (não hardcoded) | `wfdb.rdheader().adc_zero[0]` ou `baseline` |
| Range sinal pós-conversão | [-5, +5] mV | `assert sig.min() >= -5.0 and sig.max() <= 5.0` |
| Range sinal pós-filtro | [-5, +5] mV | Filtro passa-banda não deve clipar |
| Zero-phase filter | Fase linear confirmada | `scipy.signal.group_delay` ou correlação cruzada vs sinal original |
| Z-score | mean ≈ 0, std ≈ 1 | `abs(mean) < 1e-6`, `abs(std - 1.0) < 1e-4` |
| 48 MIT-BIH + 78 SVDB + 25 AFDB + 75 INCART | 226 registros processados | `len(lineage_dir.glob("*.json")) >= 226` |
| Chapman | >= 45.000 registros | `len(lineage_dir.glob("*.json")) >= 45000` |
| DLQ vazia | 0 falhas | `data/.dlq/preprocess_failures.jsonl` não existe ou vazio |
| Metadados de linhagem | 100% | Cada registro processado tem `lineage.json` válido |

**Teste:** `pytest tests/test_preprocessing.py -v`

---

## 2.10 Pipeline Idempotente e Cache

```python
# Pseudocódigo do pipeline principal
def process_record(record_path: Path, dataset: str, config: dict) -> Path:
    lineage_path = PROCESSED_DIR / dataset / f"{record_path.stem}_lineage.json"

    # 1. Idempotência: já processado?
    if lineage_path.exists():
        lineage = json.load(lineage_path)
        if lineage["preprocess_config"] == config["version"]:
            raw_checksum = compute_sha256(record_path.with_suffix(".dat"))
            if lineage["raw_checksum"] == raw_checksum:
                return PROCESSED_DIR / dataset / f"{record_path.stem}_II.npy"

    # 2. Load → Lead Select → Resample → Filter → Detrend → Normalize
    # 3. Salvar .npy + lineage.json
    # 4. Se falhar: DLQ
```

**Cache:**
- `data/processed/` entra no `.gitignore`.
- GitHub Actions: cachear `data/processed/` por `hashFiles('config/preprocess_v*.yaml')` para evitar reprocessamento em CI.

---

## 2.11 Referências Verificadas

- PhysioNet MIT-BIH Arrhythmia Database Directory (ADC 11-bit, 0–2047, baseline 1024 = 0V): https://physionet.org/physiobank/database/html/mitdbdir/intro.htm — "The ADCs were unipolar, with 11-bit resolution over a ±5 mV range. Sample values thus range from 0 to 2047 inclusive, with a value of 1024 corresponding to zero volts."citeweb_search:7#3
- WFDB Format — Signal Data (gain, baseline, units, conversion formula): https://cran.r-project.org/web/packages/EGM/vignettes/wfdb-guide.html — `physical = (digital - baseline) / gain`citeweb_search:7#1
- SciPy `resample` vs `resample_poly` (sinais não-periódicos): https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample.html — "For non-periodic signals, resample_poly may be a better choice."citeweb_search:8#0
- ECG Bandpass Filter Standards (ACC/AHA/HRS): https://emedicine.medscape.com/article/1894014-overview — "low-frequency and high-frequency cutoff values are 0.5 Hz and 150 Hz, respectively, for adult ECG."citeweb_search:7#6
- Zero-Phase Butterworth Filtering for ECG: https://pmc.ncbi.nlm.nih.gov/articles/PMC12660802/ — "We decide to use Butterworth and zero-phase filtering to protect ECG morphology while effectively removing baseline drift."citeweb_search:7#0
- AAMI EC57 Annotation Mapping: https://cran.r-project.org/web/packages/EGM/vignettes/annotation-guide.html — beat symbols table (N, L, R, V, F, A, S, etc.)citeweb_search:8#2
- WFDB Applications Guide (bxb — AAMI-standard comparator): https://physionet.org/files/wfdb/10.7.0/wag.pdf — "bxb implements the beat-by-beat comparison algorithms described in ANSI/AAMI EC57:1998"citeweb_search:8#5
- MIT-BIH SVDB 250 Hz (correção do valor obsoleto documentado na v1.0): https://pmc.ncbi.nlm.nih.gov/articles/PMC10542398/ — "Each recording contains two signals sampled at 250 samples/second"citeweb_search:9#7
- SVDB Lead Ambiguity (GitHub Issue): https://github.com/MIT-LCP/physionet/issues/112 — "the ECG1 and ECG2 belong to which channel? MLII or V2 or V5?" (sem resposta oficial)citeweb_search:8#9
- Kaggle ECG Lead II Dataset (SVDB + INCART + MITDB processados): https://www.kaggle.com/datasets/nelsonsharma/ecg-lead-2-dataset-physionet-open-access — usa Lead II para todos os datasetsciteweb_search:8#7
