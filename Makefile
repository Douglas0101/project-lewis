.PHONY: help env setup doctor dev download-all download-chapman download-mitbih mirror mirror-restore \
        catalog qg0 dlq-replay test clean clean-raw clean-mirrors \
        process pretrain finetune quantize export provenance all \
        docker-build docker-run docker-shell pre-commit-install lint format type-check \
        firmware-deps firmware-tflm-lib firmware-build firmware-native firmware-native-tflm firmware-native-stub \
        firmware-run firmware-test hard-gates hard-gates-ci check-strict-markers check-no-stub \
        verify-renode

# Detecta ambiente virtual se existente; caso contrário usa python3/pytest do sistema.
ifeq ($(wildcard .venv/bin/python),)
    PYTHON  := python3
    PYTEST  := pytest
else
    PYTHON  := .venv/bin/python
    PYTEST  := .venv/bin/pytest
endif
UV      := uv
DATA    := data

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------
help: ## Show this help message
	@echo "Project-Lewis Makefile targets:"
	@grep -E '^[a-zA-Z0-9_-]+:.*##.*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "} {printf "  %-24s %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# Onboarding
# ---------------------------------------------------------------------------
setup: env pre-commit-install ## Setup completo para novo contribuinte

doctor: ## Verifica se o ambiente local atende aos pre-requisitos
	$(PYTHON) scripts/check_environment.py

dev: ## Abre shell no container Docker de desenvolvimento
	docker compose up -d app && docker compose exec app bash

# ---------------------------------------------------------------------------
# Ambiente reprodutível (uv)
# ---------------------------------------------------------------------------
env: ## Create/sync the reproducible Python environment with uv
	$(UV) sync --frozen

# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------
docker-build: ## Build the project Docker image
	docker build -t project-lewis:latest .

docker-run: docker-build ## Build and run the project in a Docker container
	docker run --rm -it -v $(PWD):/app -v lewis-data:/app/data project-lewis:latest

docker-shell: docker-build ## Build and open a bash shell in the Docker container
	docker run --rm -it -v $(PWD):/app -v lewis-data:/app/data project-lewis:latest bash

# ---------------------------------------------------------------------------
# Git hooks e qualidade de código
# ---------------------------------------------------------------------------
pre-commit-install: ## Install pre-commit Git hooks
	$(UV) run pre-commit install

lint: ## Run flake8, mypy and bandit checks
	$(UV) run flake8 src tests --max-line-length=100
	$(UV) run mypy src --ignore-missing-imports
	$(UV) run bandit -c pyproject.toml -r src

format: ## Format Python code with black and isort
	$(UV) run black src tests
	$(UV) run isort src tests

type-check: ## Run static type checks with mypy
	$(UV) run mypy src --ignore-missing-imports

# ---------------------------------------------------------------------------
# Pipeline de dados (Fase 1)
# ---------------------------------------------------------------------------
download-chapman: ## Download the Chapman ECG dataset
	$(PYTHON) -m src.data.download_chapman

download-mitbih: ## Download MIT-BIH, SVDB, AFDB and INCART datasets
	$(PYTHON) -m src.data.download_mitbih

download-all: download-chapman download-mitbih ## Download all ECG datasets

mirror: ## Create compressed mirrors of raw datasets
	mkdir -p $(DATA)/mirrors
	tar czf $(DATA)/mirrors/chapman_mirror.tar.gz        -C $(DATA)/raw_chapman .
	tar czf $(DATA)/mirrors/mitbih_family_mirror.tar.gz  -C $(DATA)/raw_mitbih . \
	                                                        -C $(DATA)/raw_svdb   . \
	                                                        -C $(DATA)/raw_afdb   . \
	                                                        -C $(DATA)/raw_incart .

mirror-restore: ## Restore raw datasets from compressed mirrors
	mkdir -p $(DATA)/raw_chapman $(DATA)/raw_mitbih $(DATA)/raw_svdb \
	         $(DATA)/raw_afdb $(DATA)/raw_incart
	tar xzf $(DATA)/mirrors/chapman_mirror.tar.gz        -C $(DATA)/raw_chapman/
	tar xzf $(DATA)/mirrors/mitbih_family_mirror.tar.gz  -C $(DATA)/raw_mitbih/

catalog: ## Build the dataset catalog
	$(PYTHON) -c "from src.data._catalog import build_catalog; build_catalog()"

qg0: ## Run QG0 download integrity tests
	$(PYTEST) tests/test_download.py -v

dlq-replay: ## Replay dead-letter queue failed downloads
	$(PYTHON) -m src.data._downloader_replay

provenance: ## Write data provenance manifest
	$(PYTHON) -c "from src.data._compliance import write_provenance; import json; from pathlib import Path; m = json.loads(Path('src/data/checksums.json').read_text()); write_provenance(m)"

process: ## Run resample and preprocessing pipeline
	$(PYTHON) -m src.data.aggregator

features: ## Run feature engineering pipeline
	$(PYTHON) -m src.features.pipeline

audit-training-data: ## Audit training data quality
	$(PYTHON) scripts/audit_training_data.py

pretrain: ## Pre-train model on Chapman dataset
	$(PYTHON) -m src.models.pretrain_chapman

finetune: ## Fine-tune model on MIT-BIH family datasets
	$(PYTHON) -m src.models.finetune_mitbih

quantize: ## Run INT8 post-training quantization
	$(PYTHON) -m src.quantization.ptq

export: ## Export quantized model to TFLite FlatBuffer
	$(PYTHON) -m src.quantization.export_tflite

test: ## Run the Python test suite
	$(PYTEST) tests/ -q --tb=short

quality-report: ## Generate project quality report
	$(UV) run python scripts/generate_quality_report.py

# ---------------------------------------------------------------------------
# Firmware / Simulacao (Fase 2)
# ---------------------------------------------------------------------------
RENOD_DIR := firmware/tools/renode-1.15.3
RENODE_BIN := $(RENOD_DIR)/renode

verify-renode: ## Verify Renode 1.15.3 installation
	@if [ ! -x "$(RENODE_BIN)" ]; then \
	    echo "ERROR: Renode nao encontrado em $(RENODE_BIN)"; \
	    exit 1; \
	fi
	@RENODE_VERSION_OUTPUT=$$($(RENODE_BIN) --version | head -n1); \
	RENODE_VERSION=$$(echo "$$RENODE_VERSION_OUTPUT" | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -n1); \
	if [ "$$RENODE_VERSION" != "1.15.3" ]; then \
	    echo "ERROR: Renode version mismatch (expected 1.15.3, got $$RENODE_VERSION)"; \
	    echo "Output: $$RENODE_VERSION_OUTPUT"; \
	    exit 1; \
	fi; \
	echo "Renode 1.15.3 confirmed"

firmware-deps: ## Install firmware build dependencies
	$(MAKE) -C firmware firmware-deps

firmware-tflm-lib: ## Build TensorFlow Lite Micro library
	$(MAKE) -C firmware tflm-lib

firmware-build: firmware-tflm-lib ## Build STM32F4 firmware binary
	$(MAKE) -C firmware stm32f4

firmware-native: ## Build firmware native simulator
	$(MAKE) -C firmware native

firmware-native-tflm: ## Build native simulator with TFLM
	$(MAKE) -C firmware native-tflm

firmware-native-stub: ## Build native simulator with TFLM stub
	$(MAKE) -C firmware ALLOW_STUB=1 native

firmware-run: ## Run firmware in Renode emulation
	$(MAKE) -C firmware firmware-run

firmware-test: firmware-tflm-lib ## Run firmware tests under Renode
	$(MAKE) -C firmware LEWIS_USE_TFLM=1 RENODE_SIMULATION=1 firmware-test

# ---------------------------------------------------------------------------
# Hard Gates (HG-01..HG-06)
# ---------------------------------------------------------------------------
check-strict-markers: ## Verify --strict-markers is enabled in pytest
	@echo "Verificando --strict-markers em pyproject.toml..."
	@$(PYTHON) -c "import tomllib, pathlib, sys; cfg = tomllib.loads(pathlib.Path('pyproject.toml').read_text(encoding='utf-8')); addopts = cfg.get('tool', {}).get('pytest', {}).get('ini_options', {}).get('addopts', ''); sys.exit(0 if '--strict-markers' in addopts.split() else (print('ERROR: --strict-markers nao encontrado em pyproject.toml') or 1))" && echo "OK: --strict-markers configurado"

check-no-stub: ## Verify TFLM stub is not present in firmware ELF
	@if [ ! -f firmware/build/stm32f4/lewis.elf ]; then \
	    echo "SKIP: firmware/build/stm32f4/lewis.elf ainda nao existe"; \
	    exit 0; \
	fi
	@if command -v strings >/dev/null 2>&1; then \
	    if strings firmware/build/stm32f4/lewis.elf | grep -q "STUB_TFLM"; then \
	        echo "ERROR: STUB_TFLM encontrado no firmware ELF"; \
	        exit 1; \
	    fi; \
	    echo "OK: nenhum STUB_TFLM no firmware ELF"; \
	else \
	    echo "SKIP: binario strings nao disponivel"; \
	fi

hard-gates: verify-renode ## Run hard quality gates (HG-01..HG-06)
	PYTEST=$(PYTEST) ALLOW_STUB=0 CI=1 $(PYTHON) scripts/run_hard_gates.py

hard-gates-ci: check-strict-markers hard-gates check-no-stub ## Run CI hard gates including marker/stub checks

all: env download-all catalog test quality-report ## Run full pipeline: env, download, catalog, test and report

clean: ## Remove processed data, features and model artifacts
	rm -rf $(DATA)/processed/* $(DATA)/features/* models/*.h5 models/*.keras models/*.tflite

clean-raw: ## Remove all raw downloaded datasets
	rm -rf $(DATA)/raw_*

clean-mirrors: ## Remove dataset mirror archives
	rm -rf $(DATA)/mirrors/*
