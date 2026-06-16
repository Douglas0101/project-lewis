.PHONY: env download-all download-chapman download-mitbih mirror mirror-restore \
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
# Ambiente reprodutível (uv)
# ---------------------------------------------------------------------------
env:
	$(UV) sync --frozen

# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------
docker-build:
	docker build -t project-lewis:latest .

docker-run: docker-build
	docker run --rm -it -v $(PWD):/app -v lewis-data:/app/data project-lewis:latest

docker-shell: docker-build
	docker run --rm -it -v $(PWD):/app -v lewis-data:/app/data project-lewis:latest bash

# ---------------------------------------------------------------------------
# Git hooks e qualidade de código
# ---------------------------------------------------------------------------
pre-commit-install:
	$(UV) run pre-commit install

lint:
	$(UV) run flake8 src tests --max-line-length=100
	$(UV) run mypy src --ignore-missing-imports
	$(UV) run bandit -c pyproject.toml -r src

format:
	$(UV) run black src tests
	$(UV) run isort src tests

type-check:
	$(UV) run mypy src --ignore-missing-imports

# ---------------------------------------------------------------------------
# Pipeline de dados (Fase 1)
# ---------------------------------------------------------------------------
download-chapman:
	$(PYTHON) -m src.data.download_chapman

download-mitbih:
	$(PYTHON) -m src.data.download_mitbih

download-all: download-chapman download-mitbih

mirror:
	mkdir -p $(DATA)/mirrors
	tar czf $(DATA)/mirrors/chapman_mirror.tar.gz        -C $(DATA)/raw_chapman .
	tar czf $(DATA)/mirrors/mitbih_family_mirror.tar.gz  -C $(DATA)/raw_mitbih . \
	                                                        -C $(DATA)/raw_svdb   . \
	                                                        -C $(DATA)/raw_afdb   . \
	                                                        -C $(DATA)/raw_incart .

mirror-restore:
	mkdir -p $(DATA)/raw_chapman $(DATA)/raw_mitbih $(DATA)/raw_svdb \
	         $(DATA)/raw_afdb $(DATA)/raw_incart
	tar xzf $(DATA)/mirrors/chapman_mirror.tar.gz        -C $(DATA)/raw_chapman/
	tar xzf $(DATA)/mirrors/mitbih_family_mirror.tar.gz  -C $(DATA)/raw_mitbih/

catalog:
	$(PYTHON) -c "from src.data._catalog import build_catalog; build_catalog()"

qg0:
	$(PYTEST) tests/test_download.py -v

dlq-replay:
	$(PYTHON) -m src.data._downloader_replay

provenance:
	$(PYTHON) -c "from src.data._compliance import write_provenance; import json; from pathlib import Path; m = json.loads(Path('src/data/checksums.json').read_text()); write_provenance(m)"

process:
	$(PYTHON) -m src.data.aggregator

pretrain:
	$(PYTHON) -m src.models.pretrain_chapman

finetune:
	$(PYTHON) -m src.models.finetune_mitbih

quantize:
	$(PYTHON) -m src.quantization.ptq

export:
	$(PYTHON) -m src.quantization.export_tflite

test:
	$(PYTEST) tests/ -q --tb=short

quality-report:
	$(UV) run python scripts/generate_quality_report.py

# ---------------------------------------------------------------------------
# Firmware / Simulacao (Fase 2)
# ---------------------------------------------------------------------------
RENOD_DIR := firmware/tools/renode-1.15.3
RENODE_BIN := $(RENOD_DIR)/renode

verify-renode:
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

firmware-deps:
	$(MAKE) -C firmware firmware-deps

firmware-tflm-lib:
	$(MAKE) -C firmware tflm-lib

firmware-build: firmware-tflm-lib
	$(MAKE) -C firmware stm32f4

firmware-native:
	$(MAKE) -C firmware native

firmware-native-tflm:
	$(MAKE) -C firmware native-tflm

firmware-native-stub:
	$(MAKE) -C firmware ALLOW_STUB=1 native

firmware-run:
	$(MAKE) -C firmware firmware-run

firmware-test: firmware-tflm-lib
	$(MAKE) -C firmware LEWIS_USE_TFLM=1 RENODE_SIMULATION=1 firmware-test

# ---------------------------------------------------------------------------
# Hard Gates (HG-01..HG-06)
# ---------------------------------------------------------------------------
check-strict-markers:
	@echo "Verificando --strict-markers em pyproject.toml..."
	@$(PYTHON) -c "import tomllib, pathlib, sys; cfg = tomllib.loads(pathlib.Path('pyproject.toml').read_text(encoding='utf-8')); addopts = cfg.get('tool', {}).get('pytest', {}).get('ini_options', {}).get('addopts', ''); sys.exit(0 if '--strict-markers' in addopts.split() else (print('ERROR: --strict-markers nao encontrado em pyproject.toml') or 1))" && echo "OK: --strict-markers configurado"

check-no-stub:
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

hard-gates: verify-renode
	PYTEST=$(PYTEST) ALLOW_STUB=0 CI=1 $(PYTHON) scripts/run_hard_gates.py

hard-gates-ci: check-strict-markers hard-gates check-no-stub

all: env download-all catalog test quality-report

clean:
	rm -rf $(DATA)/processed/* $(DATA)/features/* models/*.h5 models/*.keras models/*.tflite

clean-raw:
	rm -rf $(DATA)/raw_*

clean-mirrors:
	rm -rf $(DATA)/mirrors/*
