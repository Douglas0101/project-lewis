# Project-Lewis — Camada 6: Validação e Quality Gates
## Responsável: QA / DevOps / Arquiteto

**Versão:** 1.1 | **Data:** 2026-06-09 | **Arquiteto:** Douglas Souza

---

## 6.1 Objetivo
Definir, implementar e executar quality gates rigorosos que bloqueiam avanço entre camadas do pipeline. Nenhum código, dataset ou modelo avança sem passar no gate correspondente. Os gates são fundamentados no padrão ANSI/AAMI EC57:1998, que exige avaliação "hands-off" (sem intervenção humana) e reprodutível.citeweb_search:22#6

---

## 6.2 Quality Gates Consolidados e Corrigidos

Os gates abaixo refletem as correções técnicas das Camadas 1–5. Valores obsoletos do v1.0 foram atualizados.

| Gate | Nome | Critério | Dataset | Como Validar | Bloqueia |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **QG0** | Download | Chapman ≥ 45.000 registros; MIT-BIH 48, SVDB 78, AFDB 25, INCART 75; checksums SHA256 válidos; DLQ vazia | Todos | `pytest tests/test_download.py -v` | Pré-treino |
| **QG1** | Resample + Pré-processamento | Fs = 500 Hz (MIT-BIH 360→500, SVDB 250→500, AFDB 250→500, INCART 257→500); ganho/baseline lidos do .hea (não hardcoded); range ±5 mV; zero-phase filter; Z-score global; linhagem 100% | MIT-BIH+ | `pytest tests/test_loader.py -v` + `tests/test_resampler.py` + `tests/test_preprocessing.py` | Segmentação |
| **QG2** | AMPT @ 500Hz | Sens > 96.5% vs .atr (tol = 150 ms), PPV > 99.0%, F1 > 97.5%, FP < 1% do total; banda 5–15 Hz (não 5–25 Hz); MWI 150ms; refratariedade 360ms | MIT-BIH | `pytest tests/test_ampt.py -v` | Feature Engineering |
| **QG3** | Features | Janela 1000ms (600ms fallback para RR < 600ms); ≥ 10 dimensões; sem NaN/Inf; QRS width via envelope method (onset 300ms antes, offset 150ms depois, threshold 50% \|R\|); ST slope J+60ms → J+80ms; SMOTE apenas em feature space; augmentation apenas treino fine-tuning | MIT-BIH+ | `pytest tests/test_features.py -v` + `tests/test_segmenter.py` | Modelagem |
| **QG4** | Pré-treino | AUC-ROC macro > 0.85; loss < 0.15; 5 superclasses SCP-ECG (NORM, CD, MI, HYP, STTC); seed 42; determinístico | Chapman | `pytest tests/test_pretrain.py -v` | Fine-tuning |
| **QG5** | Fine-tuning | Acc global > 93% (inter-patient); F1-macro > 85%; MCC > 0.80; Sens N > 96%, V > 90%, S > 75%, F > 60%, Q > 70%; FPR global < 5%; GroupKFold std F1-macro < 3%; DLQ vazia; linhagem 100% | MIT-BIH+ | `pytest tests/test_finetune.py -v` | Quantização |
| **QG6** | Quantização + Exportação | Per-channel INT8; ΔAcc global < 1%; ΔF1-macro < 2%; ΔSens N < 0.5%, V/S/F/Q < 3%; FlatBuffer < 64KB; arena TFLM < 64KB; CMSIS-NN ativado; header C compilável (`-Werror`); parâmetros de quantização extraídos; alinhamento 16 bytes | MIT-BIH+ subset | `pytest tests/test_quantization.py -v` + `tests/test_tflm_integration.py` | Firmware (Fase 2) |

> **Nota sobre métricas inter-patient:** O padrão AAMI EC57:1998 exige que a avaliação seja "hands-off" e reprodutível.citeweb_search:22#6 Métricas intra-patient (shuffle aleatório) são inválidas para validação final, pois produzem Acc ~99% artificialmente.citeweb_search:22#3 O Project-Lewis adota GroupKFold por paciente como padrão obrigatório.

---

## 6.3 Estratégia de Testes: Pirâmide de Qualidade

A pirâmide de testes para ML segue a proporção 70/20/10: 70% unit tests, 20% integration tests, 10% e2e tests.citeweb_search:22#4

```
        /\
       /  \     E2E Pipeline (10%)
      /____\
     /      \   Integration (20%)
    /________\
   /          \ Unit Tests (70%)
  /____________\
```

### 6.3.1 Unit Tests (70%)

Testam funções individuais com fixtures parametrizadas.citeweb_search:22#7

```python
# tests/conftest.py — Fixtures compartilhadas
import pytest
import numpy as np
from pathlib import Path

@pytest.fixture(scope="session")
def raw_dir():
    return Path("data/raw_mitbih")

@pytest.fixture(scope="session")
def processed_dir():
    return Path("data/processed")

@pytest.fixture(scope="session")
def config_preprocess():
    return Path("config/preprocess_v1.0.yaml")

@pytest.fixture(scope="session")
def sample_record_100():
    """Registro 100 do MIT-BIH como fixture padrão para testes rápidos."""
    from src.data.loader import MITBIHLoader
    loader = MITBIHLoader()
    sig = loader.load_signal(Path("data/raw_mitbih/100"))
    r_peaks, labels = loader.load_annotations(Path("data/raw_mitbih/100"))
    return {"sig": sig, "r_peaks": r_peaks, "labels": labels, "record_id": "100"}

# Fixture parametrizada para múltiplos registros
@pytest.fixture(scope="session", params=["100", "101", "103", "200", "233"])
def sample_record_param(request):
    from src.data.loader import MITBIHLoader
    loader = MITBIHLoader()
    sig = loader.load_signal(Path(f"data/raw_mitbih/{request.param}"))
    r_peaks, labels = loader.load_annotations(Path(f"data/raw_mitbih/{request.param}"))
    return {"sig": sig, "r_peaks": r_peaks, "labels": labels, "record_id": request.param}
```

```python
# tests/test_download.py
import pytest
from pathlib import Path

class TestDownload:
    @pytest.mark.qg0
    def test_chapman_count(self, raw_chapman_dir):
        assert len(list(raw_chapman_dir.rglob("*.hea"))) >= 45000

    @pytest.mark.qg0
    @pytest.mark.parametrize("dataset,expected", [
        ("mitdb", 48), ("svdb", 78), ("afdb", 25), ("incartdb", 75)
    ])
    def test_record_count(self, raw_dir, dataset, expected):
        hea_files = list((raw_dir / dataset).glob("*.hea"))
        assert len(hea_files) == expected, f"{dataset}: esperado {expected}, obtido {len(hea_files)}"

    @pytest.mark.qg0
    def test_checksums_valid(self, checksums_file):
        from src.data.checksums import verify_checksum
        import json
        with open(checksums_file) as f:
            checksums = json.load(f)
        for filename, meta in checksums.items():
            assert verify_checksum(Path(filename), meta["sha256"])

    @pytest.mark.qg0
    def test_dlq_vazia(self, dlq_dir):
        assert not any(dlq_dir.glob("*.jsonl")) or all(
            f.stat().st_size == 0 for f in dlq_dir.glob("*.jsonl")
        )
```

```python
# tests/test_loader.py
import pytest
import numpy as np

class TestLoader:
    @pytest.mark.qg1
    def test_gain_read_from_header(self, sample_record_100):
        """Ganho deve ser lido do .hea, não hardcoded."""
        from src.data.loader import MITBIHLoader
        loader = MITBIHLoader()
        header = loader.read_header(Path("data/raw_mitbih/100"))
        assert header.adc_gain[0] == pytest.approx(200.0, rel=0.01)

    @pytest.mark.qg1
    def test_sampling_rate_500hz(self, sample_record_100):
        sig = sample_record_100["sig"]
        # Após resample, fs deve ser 500 Hz
        assert len(sig) == pytest.approx(30 * 60 * 500, rel=0.01)  # 30 min @ 500Hz

    @pytest.mark.qg1
    def test_signal_range(self, sample_record_100):
        sig = sample_record_100["sig"]
        assert sig.min() >= -5.0
        assert sig.max() <= 5.0

    @pytest.mark.qg1
    def test_annotations_aami_mapped(self, sample_record_100):
        labels = sample_record_100["labels"]
        assert set(labels).issubset({"N", "V", "S", "F", "Q"})
        assert len(labels) == len(sample_record_100["r_peaks"])
```

```python
# tests/test_ampt.py
import pytest

class TestAMPT:
    @pytest.mark.qg2
    @pytest.mark.parametrize("record_id", ["100", "101", "103", "105", "200", "233"])
    def test_sensitivity_per_record(self, record_id):
        from src.data.loader import MITBIHLoader
        from src.features.ampt_500hz import AMPTDetector
        loader = MITBIHLoader()
        sig = loader.load_signal(Path(f"data/raw_mitbih/{record_id}"))
        r_true, _ = loader.load_annotations(Path(f"data/raw_mitbih/{record_id}"))

        det = AMPTDetector(fs=500.0)
        result = det.evaluate(sig, r_true, tol_ms=150.0)

        assert result["Sens"] >= 0.965, f"Sens {result['Sens']:.3f} < 0.965 em {record_id}"
        assert result["PPV"] >= 0.990, f"PPV {result['PPV']:.3f} < 0.990 em {record_id}"
        assert result["FP"] <= int(0.01 * result["n_true"]), f"FP excessivo em {record_id}"
```

```python
# tests/test_segmenter.py
import pytest
import numpy as np

class TestSegmenter:
    @pytest.mark.qg3
    def test_window_size(self, sample_record_100):
        from src.data.segmenter import ECGSegmenter
        seg = ECGSegmenter(fs=500.0, window_ms=1000.0)
        assert seg.window_len == 501  # 500 amostras + 1 (centro ímpar)

    @pytest.mark.qg3
    def test_no_zero_padding(self, sample_record_100):
        from src.data.segmenter import ECGSegmenter
        seg = ECGSegmenter(fs=500.0)
        sig = sample_record_100["sig"]
        r_peaks = sample_record_100["r_peaks"]
        X, _ = seg.segment_with_labels(sig, r_peaks, labels=None)
        # Verificar que nenhum segmento tem zeros artificiais nas bordas
        for seg_arr in X:
            assert not np.all(seg_arr[:10] == 0.0), "Padding detectado no início"
            assert not np.all(seg_arr[-10:] == 0.0), "Padding detectado no fim"

    @pytest.mark.qg3
    def test_fallback_600ms(self, sample_record_100):
        from src.data.segmenter import ECGSegmenter
        seg = ECGSegmenter(fs=500.0, window_ms=1000.0, min_window_ms=600.0)
        sig = sample_record_100["sig"]
        # Simular RR < 600ms (taquicardia)
        r_peaks_fast = np.array([100, 200, 300, 400])  # RR = 100ms (200 BPM)
        X, _ = seg.segment_with_labels(sig, r_peaks_fast, labels=None)
        assert X.shape[1] == 301  # 600ms = 300 amostras + 1
```

### 6.3.2 Schema Tests (Integração de Dados)

Validam a estrutura e os tipos dos dados em cada camada.citeweb_search:22#9

```python
# tests/test_schema.py
import pytest
import pandas as pd
from pandera import DataFrameSchema, Column, Check

RAW_SCHEMA = DataFrameSchema({
    "record_id": Column(str),
    "fs": Column(float, Check.greater_than(0)),
    "n_sig": Column(int, Check.greater_than(0)),
    "sig_len": Column(int, Check.greater_than(0)),
    "adc_gain": Column(float, Check.greater_than(0)),
    "baseline": Column(int),
    "units": Column(str, Check.isin(["mV", "mV*2", "mV/2", "mV/4", "mV/6", "mV/12"])),
})

PROCESSED_SCHEMA = DataFrameSchema({
    "segment": Column(object),  # np.ndarray
    "label": Column(str, Check.isin(["N", "V", "S", "F", "Q"])),
    "rr_prev": Column(float, Check.greater_than_or_equal_to(0)),
    "qrs_width_ms": Column(float, Check.in_range(40, 200)),
    "st_slope_mV_s": Column(float),
    "r_amplitude": Column(float),
})

class TestSchema:
    @pytest.mark.qg1
    def test_raw_catalog_schema(self, raw_catalog_path):
        df = pd.read_json(raw_catalog_path, lines=True)
        RAW_SCHEMA.validate(df)

    @pytest.mark.qg3
    def test_processed_features_schema(self, features_path):
        df = pd.read_parquet(features_path)
        PROCESSED_SCHEMA.validate(df)
```

### 6.3.3 Integration Tests (20%)

Testam a integração entre camadas.

```python
# tests/test_integration.py
import pytest

class TestIntegration:
    @pytest.mark.qg1
    def test_end_to_end_download_to_preprocess(self, tmp_path):
        """Download → Resample → Filter → Detrend → Normalize → Linhagem."""
        from src.data.download_mitbih import download_mitbih_family
        from src.data.preprocessor import ECGPreprocessor
        from src.data.loader import MITBIHLoader

        raw_dir = tmp_path / "raw"
        download_mitbih_family(raw_dir, datasets={"mitdb": ["100"]})

        loader = MITBIHLoader()
        sig = loader.load_signal(raw_dir / "100")

        pre = ECGPreprocessor(config_path="config/preprocess_v1.0.yaml")
        sig_proc, metadata = pre.process(sig, record_id="100")

        assert sig_proc.shape == sig.shape
        assert metadata["fs"] == 500.0
        assert abs(metadata["mean"]) < 1e-6
        assert abs(metadata["std"] - 1.0) < 1e-4
        assert Path("data/lineage/preprocess/100.json").exists()

    @pytest.mark.qg3
    def test_segmentation_to_features(self, sample_record_100):
        """Segmentação → Features → Sem NaN."""
        from src.data.segmenter import ECGSegmenter
        from src.features.time_domain import TimeDomainFeatures
        from src.features.morphological import MorphologicalFeatures

        seg = ECGSegmenter(fs=500.0)
        td = TimeDomainFeatures()
        morph = MorphologicalFeatures()

        sig = sample_record_100["sig"]
        r_peaks = sample_record_100["r_peaks"]
        labels = sample_record_100["labels"]

        X, y = seg.segment_with_labels(sig, r_peaks, labels)
        temporal = td.extract(r_peaks, fs=500.0)
        morphological = morph.extract(X, fs=500.0)

        assert len(temporal) == len(morphological) == len(y)
        assert not any(np.isnan(list(m.values())) for m in temporal)
        assert not any(np.isnan(list(m.values())) for m in morphological)
```

### 6.3.4 End-to-End Tests (10%)

Testam o pipeline completo em um subconjunto pequeno.

```python
# tests/test_e2e.py
import pytest

class TestE2E:
    @pytest.mark.slow
    @pytest.mark.qg5
    def test_full_pipeline_mitbih_subset(self, tmp_path):
        """Pipeline completo: download → preprocess → segment → features → train → evaluate."""
        # Usar apenas 3 registros MIT-BIH para teste rápido (10–20 min)
        records = ["100", "101", "103"]
        # ... pipeline completo
        # Assert: modelo treinado, métricas AAMI calculadas, F1-macro > 0.80
```

---

## 6.4 CI/CD com Cache Externo para Datasets Grandes

O GitHub Actions tem limite de 5GB para cache local e 2GB para artifacts.citeweb_search:22#2 Datasets como Chapman (~5GB raw) excedem esse limite. A estratégia recomendada é cache externo (S3/GCS/Azure Blob) + subset local para CI.citeweb_search:22#2

### .github/workflows/ci.yml
```yaml
name: Project-Lewis CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

env:
  PYTHON_VERSION: "3.12"
  DATASET_MIRROR_URL: "s3://project-lewis-mirrors/datasets/"  # ou GCS/Azure
  CI_SUBSET_SIZE: "1000"  # amostras para testes rápidos

jobs:
  unit-tests:
    runs-on: ubuntu-24.04
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Cache pip dependencies
        uses: actions/cache@v4
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-${{ hashFiles('requirements.txt') }}
          restore-keys: ${{ runner.os }}-pip-

      - name: Cache dataset subset (local)
        uses: actions/cache@v4
        with:
          path: data/.ci_cache/
          key: datasets-ci-${{ hashFiles('src/data/checksums.json') }}
          restore-keys: datasets-ci-

      - name: Download dataset subset
        run: |
          if [ ! -d "data/.ci_cache/mitbih_subset" ]; then
            aws s3 cp ${{ env.DATASET_MIRROR_URL }}/mitbih_ci_subset.tar.gz data/.ci_cache/
            tar xzf data/.ci_cache/mitbih_ci_subset.tar.gz -C data/
          fi
        env:
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}

      - name: Install dependencies
        run: make env

      - name: Run unit + schema tests
        run: pytest tests/test_*.py -m "not slow and not qg4 and not qg5 and not qg6" -v --tb=short

      - name: Upload test results
        uses: actions/upload-artifact@v4
        if: always()
        with:
          name: pytest-results-unit
          path: reports/pytest.xml

  integration-tests:
    runs-on: ubuntu-24.04
    needs: unit-tests
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Download full MIT-BIH mirror
        run: |
          aws s3 cp ${{ env.DATASET_MIRROR_URL }}/mitbih_family_mirror.tar.gz data/
          tar xzf data/mitbih_family_mirror.tar.gz -C data/

      - name: Run integration tests
        run: pytest tests/test_integration.py -v --tb=short

  quality-gates:
    runs-on: ubuntu-24.04
    needs: [unit-tests, integration-tests]
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Download full datasets
        run: |
          aws s3 cp ${{ env.DATASET_MIRROR_URL }}/chapman_mirror.tar.gz data/
          aws s3 cp ${{ env.DATASET_MIRROR_URL }}/mitbih_family_mirror.tar.gz data/
          tar xzf data/chapman_mirror.tar.gz -C data/
          tar xzf data/mitbih_family_mirror.tar.gz -C data/

      - name: Run QG4 (pré-treino) — slow
        run: pytest tests/test_pretrain.py -v --tb=short -m qg4
        timeout-minutes: 180

      - name: Run QG5 (fine-tuning) — slow
        run: pytest tests/test_finetune.py -v --tb=short -m qg5
        timeout-minutes: 120

      - name: Run QG6 (quantização) — slow
        run: pytest tests/test_quantization.py tests/test_tflm_integration.py -v --tb=short -m qg6
        timeout-minutes: 60

      - name: Generate quality report
        run: python scripts/generate_quality_report.py --output reports/quality_report.md

      - name: Upload quality report
        uses: actions/upload-artifact@v4
        with:
          name: quality-report
          path: reports/quality_report.md
```

> **Estratégia de cache:** 70.9% dos repositórios GitHub usam `actions/cache` explicitamente.citeweb_search:22#0 Para datasets > 1GB, o cache externo (S3/GCS) é obrigatório; o cache local do GitHub Actions é usado apenas para metadados e subsets.

---

## 6.5 Dead Letter Queue (DLQ) para CI/CD

Falhas de CI são logadas em `data/.dlq/ci_failures.jsonl` para análise post-mortem:

```json
{
  "workflow": "Project-Lewis CI",
  "job": "quality-gates",
  "run_id": "123456789",
  "commit": "abc123def456",
  "branch": "main",
  "failed_gate": "QG5",
  "test": "tests/test_finetune.py::TestFineTune::test_f1_macro",
  "error": "AssertionError: F1-macro 0.823 < 0.850",
  "metrics": {"f1_macro": 0.823, "acc": 0.941, "sens_N": 0.972, "sens_V": 0.891},
  "timestamp": "2026-06-09T22:15:00Z"
}
```

**Regras:**
- Falha em QG0–QG3: bug de código ou regressão de dados → bloquear PR imediatamente.
- Falha em QG4: instabilidade de pré-treino → revisar LR, batch_size, ou arquitetura.
- Falha em QG5: overfitting ou underfitting → revisar regularização, class weights, ou augmentation.
- Falha em QG6: op não suportado ou degradação excessiva → revisar arquitetura para TFLM.

---

## 6.6 Relatório de Qualidade Automatizado

O script `scripts/generate_quality_report.py` consolida todos os gates em um markdown:

```markdown
# Quality Report — Project-Lewis
**Data:** 2026-06-09 22:15 UTC | **Commit:** abc123 | **Branch:** main

| Gate | Status | Valor | Threshold | Pass?
| :--- | :--- | :--- | :--- | :--- |
| QG0 | ✅ | 45.152 registros | ≥ 45.000 | Sim |
| QG1 | ✅ | Fs = 500 Hz | 500 Hz | Sim |
| QG2 | ✅ | Sens = 99.1% | ≥ 96.5% | Sim |
| QG3 | ✅ | 12 features | ≥ 10 | Sim |
| QG4 | ✅ | AUC-ROC = 0.91 | ≥ 0.85 | Sim |
| QG5 | ✅ | F1-macro = 0.87 | ≥ 0.85 | Sim |
| QG6 | ✅ | FlatBuffer = 24.5 KB | < 64 KB | Sim |

**DLQ:** 0 falhas pendentes.
**Tempo total de pipeline:** 4h 23min.
```

---

## 6.7 Quality Gate Final (QG-Final)

| Critério | Valor | Como Validar |
| :--- | :--- | :--- |
| Todos os QGs (0–6) | Pass | `pytest tests/ -m "qg0 or qg1 or qg2 or qg3 or qg4 or qg5 or qg6"` |
| Cobertura de testes | > 80% | `pytest --cov=src --cov-report=xml` |
| Testes parametrizados | 100% dos registros MIT-BIH | `pytest tests/test_ampt.py -v` |
| Schema validation | 100% dos datasets | `pytest tests/test_schema.py -v` |
| CI pass | Sim | GitHub Actions badge verde |
| DLQ vazia | 0 falhas | `data/.dlq/ci_failures.jsonl` vazio |
| Relatório de qualidade | Gerado | `reports/quality_report.md` existe |
| Artifact retention | 30 dias | GitHub Actions settings |

**Teste:** `make quality-report` (equivale a `pytest tests/ + python scripts/generate_quality_report.py`)

---

## 6.8 Referências Verificadas

- AAMI EC57:1998 — "hands-off" evaluation (sem intervenção humana): https://mdcpp.com/doc/standard/ANSIAAMIEC57-1998(R)2003.pdf — "A credible evaluation must be reproducible... evaluations of these devices shall be performed without human intervention."citeweb_search:22#6
- AAMI EC57 Evaluation Methodology (WFDB bxb, rxr, mxm, epic): https://physionet.org/physiotools/wpg/wpg_67.htm — "EC38 and EC57 specify the use of 'bxb', 'rxr', 'mxm', and 'epic' to perform evaluations."citeweb_search:22#1
- Systematic Review AAMI EC57 Metrics: https://arxiv.org/html/2503.07276v1 — "few studies in the literature follow the recommendations of AAMI... ANSI/AAMI/ISO EC57 standard provides guidelines for assessing arrhythmia classification algorithms."citeweb_search:22#3
- Pytest Best Practices for ML Pipelines: https://www.fuzzylabs.ai/blog-post/the-art-of-testing-machine-learning-pipelines — fixtures, parametrização, testes de pipeline.citeweb_search:22#4
- Pytest Parametrization (S/A level): https://towardsdatascience.com/testing-best-practices-for-machine-learning-libraries-41b7d0362c95 — `@pytest.mark.parametrize`, fixtures compartilhadas.citeweb_search:22#7
- Schema Tests for Data Pipelines: https://eugeneyan.com/writing/testing-pipelines/ — row-level, column-level, table-level, schema tests.citeweb_search:22#9
- GitHub Actions Cache Study (arXiv 2026): https://arxiv.org/html/2604.13129v1 — "70.9% explicit caching via actions/cache... build and test jobs most prominent."citeweb_search:22#0
- Large Files in CI/CD (StackOverflow): https://stackoverflow.com/questions/66474531/large-files-for-github-cicd — "Store Large Files Externally possibly into AWS S3/Google Drive / GCS... Modify CI/CD Workflow to Download Data from these external sources."citeweb_search:22#2
