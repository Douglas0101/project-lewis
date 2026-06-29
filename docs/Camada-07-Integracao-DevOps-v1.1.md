# Project-Lewis — Camada 7: Integração, DevOps e Reprodutibilidade
## Responsável: DevOps / Arquiteto

**Versão:** 1.1 | **Data:** 2026-06-09 | **Arquiteto:** Douglas Souza

---

## 7.1 Objetivo
Orquestrar todo o pipeline via Makefile, gerenciar dependências com lockfile determinístico, configurar CI/CD com cache externo para datasets grandes, containerizar o ambiente para reprodutibilidade total, e definir entregáveis estruturados para a Fase 2 (Firmware), garantindo que "funciona na minha máquina" seja impossível.

---

## 7.2 Gestão de Dependências: `uv` + `pyproject.toml`

O documento v1.0 usava `requirements.txt` com versões congeladas via `pip freeze`. **Isso é obsoleto em 2026.** O ecossistema Python consolidou-se em torno de `uv` (Astral), `Poetry` e `pip-tools`.citeweb_search:24#2 Para o Project-Lewis, adota-se `uv` como gerenciador primário:

- **Performance:** `uv` instala 10–100× mais rápido que `pip` (30s → 300ms).citeweb_search:24#2
- **Lockfile nativo:** `uv.lock` garante builds determinísticos sem `pip freeze` manual.citeweb_search:24#2
- **Compatibilidade:** `uv` consome `pyproject.toml` (PEP 621) e pode exportar `requirements.txt` para CI legado.citeweb_search:24#2
- **Adoção:** 75M downloads/mês no PyPI, superando Poetry (66M).citeweb_search:24#2

### pyproject.toml
```toml
[project]
name = "project-lewis"
version = "1.1.0"
description = "ECG Arrhythmia Classification Pipeline for Edge Devices"
requires-python = ">=3.12"
readme = "README.md"
license = {text = "MIT"}
authors = [
    {name = "Douglas Souza", email = "douglas@example.com"}
]

# Dependências de produção (mínimas para inference)
dependencies = [
    "numpy>=1.26,<2.0",
    "scipy>=1.11,<2.0",
    "pandas>=2.1,<3.0",
    "wfdb>=4.1,<5.0",
    "tensorflow>=2.16,<2.17",
    "scikit-learn>=1.4,<2.0",
    "imbalanced-learn>=0.12,<1.0",
    "joblib>=1.3,<2.0",
    "pyarrow>=14.0,<15.0",
    "matplotlib>=3.8,<4.0",
    "seaborn>=0.13,<1.0",
    "kagglehub>=0.3,<1.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0,<9.0",
    "pytest-cov>=4.1,<5.0",
    "pytest-xdist>=3.5,<4.0",
    "black>=24.0,<25.0",
    "isort>=5.13,<6.0",
    "flake8>=7.0,<8.0",
    "mypy>=1.8,<2.0",
    "bandit>=1.7,<2.0",
    "pre-commit>=3.6,<4.0",
    "pandera>=0.18,<1.0",
]

[tool.uv]
# Configurações específicas do uv
python-downloads = "automatic"

[tool.black]
line-length = 100
target-version = ["py312"]
include = '\.pyi?$'
exclude = '''
/(
    \.venv
  | \.git
  | \.mypy_cache
  | data/
  | notebooks/
)/
'''

[tool.isort]
profile = "black"
line_length = 100
known_first_party = ["src"]

[tool.mypy]
python_version = "3.12"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
ignore_missing_imports = true
exclude = ["data/", "notebooks/", "tests/"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --tb=short --strict-markers"
markers = [
    "qg0: Quality Gate 0 — Download",
    "qg1: Quality Gate 1 — Preprocessing",
    "qg2: Quality Gate 2 — AMPT",
    "qg3: Quality Gate 3 — Features",
    "qg4: Quality Gate 4 — Pretrain",
    "qg5: Quality Gate 5 — Finetune",
    "qg6: Quality Gate 6 — Quantization",
    "slow: Tests that take > 1 minute",
]

[tool.bandit]
exclude_dirs = ["tests/", "notebooks/"]
skips = ["B101"]  # assert_used em testes é aceitável
```

### Comandos uv
```bash
# Instalar uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Criar ambiente e instalar dependências
uv sync                    # instala tudo (prod + dev)
uv sync --no-dev           # apenas produção (CI de deploy)

# Adicionar dependência
uv add scipy
uv add --dev pytest

# Lockfile
uv lock                    # gera/atualiza uv.lock
uv export > requirements.txt  # exporta para CI legado

# Executar scripts
uv run python -m src.data.download_mitbih
uv run pytest tests/ -v
```

> **Fallback:** Se `uv` não estiver disponível (ex: máquina corporativa com restrições), usar `pip-tools`:
> ```bash
> pip install pip-tools
> # requirements.in (dependências de alto nível)
> pip-compile requirements.in --generate-hashes --output-file requirements.txt
> pip-sync requirements.txt
> ```citeweb_search:24#12

---

## 7.3 Makefile com Paralelismo e Dependências

O Makefile v1.0 era sequencial e não verificava pré-requisitos. O v1.1 adiciona:
- **Paralelismo:** `make -j4` para etapas independentes (download MIT-BIH + download Chapman podem rodar em paralelo).
- **Verificação de pré-requisitos:** checa se `uv` está instalado, se `data/mirrors/` existe, etc.
- **Targets por camada:** cada camada é um target independente, permitindo reexecução seletiva.
- **Idempotência:** targets verificam se o output já existe e está válido (via checksum).

```makefile
# Project-Lewis — Makefile v1.1
# Uso: make -j4 all  (paralelismo com 4 jobs)

.PHONY: env check-env download-all download-chapman download-mitbih         process pretrain finetune quantize export test test-unit         test-integration test-e2e quality-report clean docker-build

PYTHON := uv run python
PYTEST := uv run pytest
NPROCS := $(shell nproc 2>/dev/null || echo 4)

# 1. Verificação de ambiente
check-env:
	@which uv > /dev/null || (echo "Erro: uv não instalado. Execute: curl -LsSf https://astral.sh/uv/install.sh | sh" && exit 1)
	@python3 --version | grep -q "3.13" || (echo "Erro: Python 3.13 requerido" && exit 1)

# 2. Ambiente
env: check-env pyproject.toml
	uv sync
	@echo "Ambiente pronto. Execute: uv run python ..."

# 3. Download (paralelizável)
download-all: download-chapman download-mitbih

download-chapman: check-env data/mirrors/chapman_mirror.tar.gz
	@echo "Chapman: verificando mirror..."
	@test -d data/raw_chapman/.checksum_valid || $(PYTHON) -m src.data.download_chapman

download-mitbih: check-env data/mirrors/mitbih_family_mirror.tar.gz
	@echo "MIT-BIH family: verificando mirror..."
	@test -d data/raw_mitbih/.checksum_valid || $(PYTHON) -m src.data.download_mitbih

# 4. Processamento (depende de download-all)
process: download-all config/preprocess_v1.0.yaml
	$(PYTHON) -m src.data.aggregator
	@echo "Processamento completo."

# 5. Pré-treino (depende de process)
pretrain: process
	$(PYTHON) -m src.models.pretrain_chapman

# 6. Fine-tuning (depende de pretrain)
finetune: pretrain
	$(PYTHON) -m src.models.finetune_mitbih

# 7. Quantização (depende de finetune)
quantize: finetune
	$(PYTHON) -m src.quantization.ptq

# 8. Exportação (depende de quantize)
export: quantize
	$(PYTHON) -m src.quantization.export_tflite
	@echo "Entregáveis firmware gerados em firmware/src/ml/"

# 9. Pipeline completo (NÃO rodar em CI — demora horas)
all: env download-all process pretrain finetune quantize export
	@echo "Pipeline completo finalizado. Verifique reports/quality_report.md"

# 10. Testes
TEST_ARGS := -v --tb=short --strict-markers

test-unit: env
	$(PYTEST) tests/ -m "not slow and not qg4 and not qg5 and not qg6" $(TEST_ARGS)

test-integration: env
	$(PYTEST) tests/test_integration.py $(TEST_ARGS)

test-e2e: env
	$(PYTEST) tests/test_e2e.py -m "slow" $(TEST_ARGS) --timeout=3600

test: test-unit test-integration

# 11. Quality report
quality-report: test
	$(PYTHON) scripts/generate_quality_report.py --output reports/quality_report.md

# 12. Docker
docker-build:
	docker build -t project-lewis:latest -f Dockerfile .

docker-run:
	docker run --rm -it -v $(PWD)/data:/app/data -v $(PWD)/models:/app/models project-lewis:latest

# 13. Pre-commit hooks
pre-commit-install: env
	uv run pre-commit install
	uv run pre-commit run --all-files

# 14. Limpeza (preserva raw/ e mirrors/)
clean:
	rm -rf data/processed/* data/features/* data/lineage/*
	rm -rf models/*.keras models/*.tflite models/*.h
	rm -rf reports/*.md reports/*.xml
	rm -rf .pytest_cache .coverage htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

clean-all: clean
	rm -rf data/raw_chapman/ data/raw_mitbih/ data/raw_svdb/ data/raw_afdb/ data/raw_incart/
	@echo "ATENCAO: datasets brutos removidos. Reexecute 'make download-all' para recuperar."
```

---

## 7.4 Pre-Commit Hooks

Hooks são obrigatórios para garantir qualidade de código antes de cada commit.citeweb_search:24#9citeweb_search:24#11

### .pre-commit-config.yaml
```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-json
      - id: check-added-large-files
        args: ["--maxkb=1000"]  # bloquear arquivos > 1MB
      - id: check-merge-conflict

  - repo: https://github.com/psf/black
    rev: 24.4.2
    hooks:
      - id: black
        language_version: python3.12

  - repo: https://github.com/pycqa/isort
    rev: 5.13.2
    hooks:
      - id: isort
        args: ["--profile", "black"]

  - repo: https://github.com/pycqa/flake8
    rev: 7.1.0
    hooks:
      - id: flake8
        args: ["--max-line-length=100", "--extend-ignore=E203,W503"]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.10.0
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
        args: ["--ignore-missing-imports"]

  - repo: https://github.com/pycqa/bandit
    rev: 1.7.9
    hooks:
      - id: bandit
        args: ["-c", "pyproject.toml"]
        additional_dependencies: ["bandit[toml]"]
```

**Instalação:**
```bash
make pre-commit-install
# ou manualmente:
uv run pre-commit install
uv run pre-commit run --all-files
```

---

## 7.5 Docker para Reprodutibilidade Total

Docker elimina o problema "funciona na minha máquina" — essencial para pipelines médicos.citeweb_search:24#3citeweb_search:24#4

### Dockerfile
```dockerfile
# Project-Lewis — Dockerfile v1.1
FROM python:3.12-slim-bookworm

# Instalar dependências de sistema
RUN apt-get update && apt-get install -y --no-install-recommends     gcc g++ make     libhdf5-dev     && rm -rf /var/lib/apt/lists/*

# Instalar uv
RUN pip install --no-cache-dir uv

# Diretório de trabalho
WORKDIR /app

# Copiar configurações de dependências
COPY pyproject.toml uv.lock ./

# Instalar dependências (aproveita cache Docker)
RUN uv sync --no-dev

# Copiar código-fonte
COPY src/ ./src/
COPY tests/ ./tests/
COPY config/ ./config/
COPY scripts/ ./scripts/
COPY Makefile ./

# Variáveis de ambiente
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=2

# Entrypoint padrão
CMD ["uv", "run", "pytest", "tests/", "-v", "--tb=short"]
```

### docker-compose.yml
```yaml
version: "3.8"

services:
  project-lewis:
    build: .
    image: project-lewis:latest
    container_name: lewis-pipeline
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./reports:/app/reports
    environment:
      - PYTHONPATH=/app
      - SEED=42
    # Para treinamento com GPU (se disponível):
    # deploy:
    #   resources:
    #     reservations:
    #       devices:
    #         - driver: nvidia
    #           count: 1
    #           capabilities: [gpu]
```

**Uso:**
```bash
make docker-build    # builda imagem
make docker-run      # roda container interativo
# ou manualmente:
docker-compose up --build
```

---

## 7.6 DVC (Data Version Control)

O `.gitignore` v1.0 simplesmente ignorava datasets. **Isso é insuficiente para reprodutibilidade.** DVC rastreia versionamento de dados grandes via Git, sem versionar os binários.citeweb_search:24#13

### Setup DVC
```bash
# Inicializar (uma vez)
uv add --dev dvc
uv run dvc init

# Rastrear datasets brutos
uv run dvc add data/raw_chapman/
uv run dvc add data/raw_mitbih/
uv run dvc add data/raw_svdb/
uv run dvc add data/raw_afdb/
uv run dvc add data/raw_incart/

# Os .dvc files (pequenos) são versionados no Git
# Os dados grandes ficam no remote (S3, GCS, local)
git add data/*.dvc .gitignore

# Configurar remote (S3)
uv run dvc remote add -d s3remote s3://project-lewis-dvc/datasets
uv run dvc push

# Em outra máquina:
uv run dvc pull   # baixa datasets na versão correta
```

### .dvcignore
```gitignore
# Ignorar processados (gerados pelo pipeline, não dados brutos)
data/processed/
data/features/
data/lineage/
data/.cache/
data/.dlq/
data/.ci_cache/
```

---

## 7.7 .gitignore Completo

```gitignore
# DVC
*.dvc
.dvc/
.dvcignore

# Dados brutos e processados (versionados via DVC, não Git)
data/raw_chapman/
data/raw_mitbih/
data/raw_svdb/
data/raw_afdb/
data/raw_incart/
data/processed/
data/features/
data/lineage/
data/mirrors/
data/.cache/
data/.dlq/
data/.ci_cache/

# Modelos e artifacts (versionados via DVC ou artifacts CI)
models/*.keras
models/*.tflite
models/*.h5
models/*.pb
models/*.h
models/*.o
*.tar.gz
*.zip
*.rar

# Ambiente Python
.venv/
__pycache__/
*.pyc
*.pyo
*.egg-info/
dist/
build/

# uv
uv.lock.bak

# Docker
.dockerignore

# Notebooks
notebooks/.ipynb_checkpoints/
*.ipynb_checkpoints/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*.swn

# OS
.DS_Store
Thumbs.db

# Relatórios temporários
reports/*.xml
reports/*.html
htmlcov/
.coverage
.pytest_cache/

# MLflow / Wandb (se usado futuramente)
mlruns/
wandb/
```

---

## 7.8 Entregáveis para Fase 2 (Firmware)

Os entregáveis devem ser headers C auto-contidos, compiláveis e validados.

| Entregável | Origem | Destino | Formato | Validação | Nota |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `model_data.h` | Camada 5 | `firmware/src/ml/` | `alignas(16) const unsigned char[]` + `len` | `arm-none-eabi-gcc -c -Werror` | FlatBuffer INT8 < 64KB |
| `quantization_params.h` | Camada 5 | `firmware/src/ml/` | `struct { float scale; int zero_point; }` | `sizeof()` == 8 bytes | Input/output dequantização |
| `filter_coeffs_q31.h` | Camada 2 | `firmware/src/dsp/` | Coeficientes IIR/FIR Q31 para CMSIS-DSP | RMSE < 1e-6 vs SciPy | Butterworth bandpass 0.5–40 Hz |
| `features_config.h` | Camada 3 | `firmware/src/features/` | `struct` com offsets, escalas, janelas | `sizeof()` alinhado a 4 bytes | AMPT params @ 500Hz |
| `normalization_params.h` | Camada 4 | `firmware/src/ml/` | `float mean, std;` ou `float min, max;` | Consistência treino vs inference | Z-score global |
| `mitbih_100.h` | Camada 1 | `firmware/tests/hil/` | Array de amostras em mV (500Hz, 30min) | Playback determinístico | Registro 100 para HIL test |

### Exemplo: `quantization_params.h`
```c
#ifndef QUANTIZATION_PARAMS_H
#define QUANTIZATION_PARAMS_H

#include <stdint.h>

/* Project-Lewis Quantization Parameters
 * Generated: 2026-06-09T22:15:00Z
 * Model: model_int8_v1.0.tflite
 * SHA256: abc123...
 */

typedef struct {
    float scale;
    int32_t zero_point;
} quant_params_t;

static const quant_params_t INPUT_QUANT = {
    .scale = 0.0039215689f,
    .zero_point = 0
};

static const quant_params_t OUTPUT_QUANT = {
    .scale = 0.0039215689f,
    .zero_point = 0
};

/* Macros inline para conversão rápida */
#define Q_FLOAT_TO_INT8(val)     (int8_t)((val) / INPUT_QUANT.scale + INPUT_QUANT.zero_point)

#define Q_INT8_TO_FLOAT(q)     (((float)(q) - OUTPUT_QUANT.zero_point) * OUTPUT_QUANT.scale)

#endif /* QUANTIZATION_PARAMS_H */
```

### Script de Validação de Entregáveis
```python
# scripts/validate_firmware_deliverables.py
import subprocess
from pathlib import Path

HEADERS = [
    "firmware/src/ml/model_data.h",
    "firmware/src/ml/quantization_params.h",
    "firmware/src/dsp/filter_coeffs_q31.h",
    "firmware/src/features/features_config.h",
    "firmware/src/ml/normalization_params.h",
]

def validate_header(path: Path) -> bool:
    result = subprocess.run(
        ["arm-none-eabi-gcc", "-c", "-mcpu=cortex-m4", "-mthumb", "-O3", "-Werror", "-Wall", str(path)],
        capture_output=True, text=True
    )
    return result.returncode == 0

for h in HEADERS:
    assert validate_header(Path(h)), f"Falha de compilacao: {h}"
print("Todos os entregáveis compilam com sucesso.")
```

---

## 7.9 CI/CD GitHub Actions (Completo)

### .github/workflows/ci.yml
```yaml
name: Project-Lewis CI/CD

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]
  schedule:
    - cron: "0 3 * * 1"  # Semanal: segunda 03:00 UTC

env:
  PYTHON_VERSION: "3.12"
  UV_VERSION: "0.4.x"
  DVC_REMOTE: "s3://project-lewis-dvc/datasets"

jobs:
  # ——— Job 1: Lint & Pre-commit ———
  lint:
    runs-on: ubuntu-24.04
    timeout-minutes: 10
    steps:
      - uses: actions/checkout@v4
      - name: Setup uv
        uses: astral-sh/setup-uv@v3
        with:
          uv-version: ${{ env.UV_VERSION }}
      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
      - name: Install dependencies
        run: uv sync --only-dev
      - name: Run pre-commit
        run: uv run pre-commit run --all-files --show-diff-on-failure

  # ——— Job 2: Unit Tests (rápido, sem datasets) ———
  unit-tests:
    runs-on: ubuntu-24.04
    needs: lint
    timeout-minutes: 15
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
        with:
          uv-version: ${{ env.UV_VERSION }}
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
      - name: Install dependencies
        run: uv sync
      - name: Run unit tests
        run: uv run pytest tests/ -m "not slow and not qg4 and not qg5 and not qg6" -v --tb=short
      - name: Upload coverage
        uses: actions/upload-artifact@v4
        if: always()
        with:
          name: coverage-unit
          path: .coverage

  # ——— Job 3: Integration Tests (datasets via DVC/S3) ———
  integration-tests:
    runs-on: ubuntu-24.04
    needs: unit-tests
    timeout-minutes: 30
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
        with:
          uv-version: ${{ env.UV_VERSION }}
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: us-east-1
      - name: Pull datasets via DVC
        run: |
          uv sync
          uv run dvc pull data/raw_mitbih.dvc
          uv run dvc pull data/raw_svdb.dvc
          uv run dvc pull data/raw_afdb.dvc
          uv run dvc pull data/raw_incart.dvc
      - name: Run integration tests
        run: uv run pytest tests/test_integration.py -v --tb=short
      - name: Upload test results
        uses: actions/upload-artifact@v4
        if: always()
        with:
          name: pytest-integration
          path: reports/pytest.xml

  # ——— Job 4: Quality Gates (apenas na main, self-hosted ou large runner) ———
  quality-gates:
    runs-on: ubuntu-24.04
    needs: [unit-tests, integration-tests]
    if: github.ref == 'refs/heads/main'
    timeout-minutes: 360
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
        with:
          uv-version: ${{ env.UV_VERSION }}
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
      - uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: us-east-1
      - name: Pull all datasets
        run: |
          uv sync
          uv run dvc pull
      - name: QG4 — Pré-treino
        run: uv run pytest tests/test_pretrain.py -m qg4 -v --tb=short
        timeout-minutes: 180
      - name: QG5 — Fine-tuning
        run: uv run pytest tests/test_finetune.py -m qg5 -v --tb=short
        timeout-minutes: 120
      - name: QG6 — Quantização + Exportação
        run: |
          uv run pytest tests/test_quantization.py -m qg6 -v --tb=short
          uv run pytest tests/test_tflm_integration.py -m qg6 -v --tb=short
        timeout-minutes: 60
      - name: Validar entregáveis firmware
        run: uv run python scripts/validate_firmware_deliverables.py
      - name: Generate quality report
        run: uv run python scripts/generate_quality_report.py --output reports/quality_report.md
      - name: Upload quality report
        uses: actions/upload-artifact@v4
        with:
          name: quality-report
          path: reports/quality_report.md
      - name: Upload models
        uses: actions/upload-artifact@v4
        with:
          name: models
          path: |
            models/*.keras
            models/*.tflite
            firmware/src/ml/*.h
```

---

## 7.10 Artifact Retention e Versionamento

| Artifact | Retenção | Destino |
| :--- | :--- | :--- |
| Modelos (.keras, .tflite) | 30 dias | GitHub Actions + S3 (`s3://project-lewis-artifacts/models/`) |
| Headers C (.h) | 30 dias | GitHub Actions + S3 |
| Quality Report (.md) | 90 dias | GitHub Actions + Wiki do repo |
| Coverage (.xml) | 7 dias | GitHub Actions (análise via codecov) |

**Versionamento semântico:**
- `v1.0.0` — Release inicial (todas as camadas passando QG0–QG6)
- `v1.1.0` — Aprimoramento de arquitetura (Camada 4)
- `v1.1.1` — Hotfix de correção de dados (Camada 1)
- Tags Git anotadas: `git tag -a v1.1.0 -m "Release v1.1.0 — QG0-QG6 pass"`

---

## 7.11 Dead Letter Queue (DLQ) para DevOps

Falhas de CI/CD são logadas em `data/.dlq/devops_failures.jsonl`:

```json
{
  "workflow": "Project-Lewis CI/CD",
  "job": "quality-gates",
  "run_id": "123456789",
  "commit": "abc123def456",
  "branch": "main",
  "failed_job": "QG5",
  "error": "Timeout after 120 minutes",
  "runner": "ubuntu-24.04",
  "uv_version": "0.4.15",
  "timestamp": "2026-06-09T22:15:00Z"
}
```

**Regras:**
- Timeout em QG4/QG5 → usar self-hosted runner ou GitHub Actions larger runner (4-core+).
- DVC pull falha → verificar credenciais AWS e permissões do bucket.
- Pre-commit falha → corrigir formatação antes de push; nunca bypassar com `--no-verify`.

---

## 7.12 Quality Gate QG-DevOps

| Critério | Valor | Como Validar |
| :--- | :--- | :--- |
| `pyproject.toml` válido | Sim | `uv sync` executa sem erro |
| `uv.lock` presente | Sim | `test -f uv.lock` |
| Pre-commit pass | Sim | `uv run pre-commit run --all-files` |
| Docker build | Sim | `docker build -t project-lewis .` |
| Docker run test | Sim | `docker run --rm project-lewis pytest -v` |
| DVC initialized | Sim | `test -d .dvc` |
| Makefile targets | Todos funcionam | `make -n all` (dry-run) |
| Entregáveis firmware | Compilam | `scripts/validate_firmware_deliverables.py` |
| CI pass | Sim | GitHub Actions badge verde |
| Artifact retention | Configurado | `.github/workflows/ci.yml` com `retention-days` |
| DLQ vazia | 0 falhas | `data/.dlq/devops_failures.jsonl` vazio |

---

## 7.13 Dependência do TensorFlow Lite Micro (TFLM)

O código-fonte do TFLM **não é versionado no Git**. Ele é clonado sob demanda em `firmware/third_party/tflite-micro/` e compilado localmente para as plataformas host (`linux_x86_64`) e ARM (`cortex-m4+fp`).

### Por que não submódulo

- O repositório `tensorflow/tflite-micro` é grande e não publica releases/tags estáveis.
- Shallow clone é difícil de configurar como submódulo.
- Nem todos os desenvolvedores precisam trabalhar com firmware; o submódulo obrigaria todo mundo a baixar TFLM.

### Pin por commit SHA

A versão exata do TFLM é controlada por:

```text
firmware/third_party/tflite-micro.commit
```

Exemplo de conteúdo:

```text
348eed01b6485f6282b805672ebf1e2a88589830
```

Alterar esse arquivo invalida o cache do CI e força um novo build validado.

### Script de instalação

```bash
# Local
make firmware-tflm
# ou
./firmware/scripts/install_tflm.sh
```

O script `firmware/scripts/install_tflm.sh`:

1. Lê o commit SHA de `firmware/third_party/tflite-micro.commit`.
2. Faz shallow clone de `tensorflow/tflite-micro` no commit especificado.
3. Builda a biblioteca nativa (`libtensorflow-microlite.a` para x86_64).
4. Builda a biblioteca ARM com CMSIS-NN (`cortex_m_generic_cortex-m4+fp`).
5. Usa `-j1` por padrão para evitar `internal compiler error` por OOM em runners com pouca RAM.

### CI / GitHub Actions

O workflow `ci.yml` possui um cache dedicado para TFLM:

```yaml
- name: Cache TFLM build
  id: cache-tflm
  uses: actions/cache@v4
  with:
    path: firmware/third_party/tflite-micro
    key: tflm-${{ runner.os }}-${{ hashFiles('firmware/third_party/tflite-micro.commit', 'firmware/scripts/install_tflm.sh') }}
    restore-keys: |
      tflm-${{ runner.os }}-

- name: Install / cache TFLM
  run: uv run make firmware-tflm
```

A chave de cache leva em conta tanto o commit SHA quanto o script de instalação, garantindo invalidação quando qualquer um dos dois mudar.

---

## 7.14 Referências Verificadas

- `uv` (Astral) — Python Package Manager 2026: https://cuttlesoft.com/blog/2026/01/27/python-dependency-management-in-2026/ — "uv is the most significant change to Python tooling in years... 75 million monthly downloads on PyPI, surpassing Poetry's approximately 66 million."citeweb_search:24#2
- Poetry vs Conda vs Pip: https://www.geeksforgeeks.org/python/conda-vs-poetry-in-python/ — "Poetry provides deterministic dependency resolution... ensuring that dependency conflicts are minimized and builds are reproducible."citeweb_search:24#1
- pip-tools + Hash Pinning: https://xygeni.io/blog/hidden-dangers-of-requirements-txt-how-dependency-pinning-can-save-you/ — "Dependency pinning is more than a best practice. It's your first line of defense against supply chain attacks."citeweb_search:24#12
- Docker para ML Reprodutível: https://blog.jyotiprakash.org/reproducible-ml-environments-with-docker — "Docker ensures that your environment is precisely defined and can be recreated identically anywhere."citeweb_search:24#3
- Pre-commit Hooks (Black, isort, flake8): https://medium.com/staqu-dev-logs/keeping-python-code-clean-with-pre-commit-hooks-black-flake8-and-isort-cac8b01e0ea1 — "isort sorts imports then black formats the code and at last flake8 checks code compliance with PEP8."citeweb_search:24#9
- Bandit Security Linter: https://github.com/gabrielsantello/black-isort-flake8 — "use 'bandit', 'flake8', 'black', 'isort' in pre-commit hooks."citeweb_search:24#11
- DVC + ML Reprodutibilidade: https://in.ncu.edu.tw/~hhchen/courses/old/2024_fall_data_science/Reproducible_ML_environment____a_tutorial.pdf — "version control using Git, experiment tracking with MLflow, creating isolated experiment environments with Conda for virtual environments and Docker for containerization."citeweb_search:24#13
