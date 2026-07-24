.PHONY: help env setup doctor dev download-all download-chapman download-mitbih mirror mirror-restore \
        catalog qg0 dlq-replay test clean clean-raw clean-mirrors \
        process pretrain finetune quantize export provenance all \
        docker-build docker-run docker-shell pre-commit-install lint format type-check \
        firmware-deps firmware-tflm firmware-tflm-lib firmware-build firmware-native firmware-native-tflm firmware-native-stub \
        firmware-run firmware-test hard-gates hard-gates-ci check-strict-markers check-no-stub \
        verify-renode \
        knowledge-index knowledge-query knowledge-status knowledge-test knowledge-clean knowledge-validate \
        knowledge-reindex-if-docs-changed \
        hybrid-eval \
        ragas-eval \
        memory-commit \
        observability-up observability-down \
        test-e2e \
        mlp-logs-dir mlp-prepare-features mlp-train-stage1 mlp-train-stage2 mlp-train \
        mlp-select-best mlp-quantize mlp-validate-quantized mlp-test-qg5 \
        mlp-features mlp-pipeline mlp-pipeline-with-features mlp-clean

# Detecta ambiente virtual se existente; caso contrario usa python3/pytest do sistema.
ifeq ($(wildcard .venv/bin/python),)
    PYTHON  := python3
    PYTEST  := pytest
else
    PYTHON  := .venv/bin/python
    PYTEST  := .venv/bin/pytest
endif
UV      := uv
DATA    := data
FIRMWARE_DIR := firmware

# Garante que tf.keras use Keras 3 standalone, evitando redirecionamento
# para tf_keras causado por sentence-transformers / tensorflow_model_optimization.
export TF_USE_LEGACY_KERAS := 0

# ---------------------------------------------------------------------------
# MLP v2.3 — variáveis configuráveis
# ---------------------------------------------------------------------------
MLP_HIDDEN_UNITS       ?= 64
MLP_STAGE1_MAX_WEIGHT  ?= 20
MLP_STAGE2_MAX_WEIGHT  ?= 10
MLP_F_OVERSAMPLE_RATIO ?= 0.75
MLP_N_CAL              ?= 500
MLP_LOG                ?= logs/mlp_v2.3.log
MLP_STAGE1_EXP         ?= experiments/stage1_mlp_features_v2.3
MLP_STAGE2_EXP         ?= experiments/stage2_mlp_features_v2.3

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
# Ambiente reprodutivel (uv)
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
# Git hooks e qualidade de codigo
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

# ---------------------------------------------------------------------------
# MLP v2.3 — pipeline de treinamento, seleção, quantização e QG5'
# ---------------------------------------------------------------------------

mlp-logs-dir:
	@mkdir -p logs

mlp-prepare-features: mlp-logs-dir ## Prepara NPZs de features para treino MLP v2.3
	$(PYTHON) scripts/prepare_stage1_features.py 2>&1 | tee -a $(MLP_LOG)
	$(PYTHON) scripts/prepare_stage2_features.py 2>&1 | tee -a $(MLP_LOG)

mlp-train-stage1: mlp-logs-dir ## Treina Estágio 1 MLP v2.3 (N vs Anormal)
	$(PYTHON) scripts/train_stage1_mlp.py \
		--hidden-units $(MLP_HIDDEN_UNITS) \
		--max-weight $(MLP_STAGE1_MAX_WEIGHT) \
		--output-dir $(MLP_STAGE1_EXP) 2>&1 | tee -a $(MLP_LOG)

mlp-train-stage2: mlp-logs-dir ## Treina Estágio 2 MLP v2.3 (S vs V vs F)
	$(PYTHON) scripts/train_stage2_mlp.py \
		--hidden-units $(MLP_HIDDEN_UNITS) \
		--f-oversample-ratio $(MLP_F_OVERSAMPLE_RATIO) \
		--max-weight $(MLP_STAGE2_MAX_WEIGHT) \
		--output-dir $(MLP_STAGE2_EXP) 2>&1 | tee -a $(MLP_LOG)

mlp-train: mlp-train-stage1 mlp-train-stage2 ## Treina ambos os estágios

mlp-select-best: mlp-logs-dir ## Seleciona melhor fold e publica em models/
	$(PYTHON) scripts/select_best_mlp_fold.py \
		--stage1-exp $(MLP_STAGE1_EXP) \
		--stage2-exp $(MLP_STAGE2_EXP) 2>&1 | tee -a $(MLP_LOG)

mlp-quantize: mlp-logs-dir ## Quantiza modelos v2.3 para INT8
	$(PYTHON) scripts/quantize_mlp_features.py --n-cal $(MLP_N_CAL) 2>&1 | tee -a $(MLP_LOG)

mlp-validate-quantized: mlp-logs-dir ## Valida ΔF1-macro < 2% entre float32 e INT8
	$(PYTHON) scripts/validate_quantized_mlp.py 2>&1 | tee -a $(MLP_LOG)

mlp-test-qg5: mlp-logs-dir ## Roda testes QG5' do pipeline MLP v2.3
	$(PYTEST) tests/test_two_stage_mlp_qg5.py tests/test_morphological_features.py -v 2>&1 | tee -a $(MLP_LOG)

mlp-features: features mlp-prepare-features ## Feature engineering completa + preparação de NPZs

mlp-pipeline: mlp-prepare-features mlp-train mlp-select-best mlp-quantize mlp-validate-quantized mlp-test-qg5 ## Pipeline padrão MLP v2.3 (usa features já geradas)

mlp-pipeline-with-features: mlp-features mlp-train mlp-select-best mlp-quantize mlp-validate-quantized mlp-test-qg5 ## Pipeline MLP v2.3 com regeneração de features

mlp-clean: ## Remove artefatos v2.3 de experiments e models
	rm -rf $(MLP_STAGE1_EXP) $(MLP_STAGE2_EXP)
	rm -f models/*_v2.3*
	rm -f models/quantized/*_v2.3*
	rm -f $(MLP_LOG)

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

test-e2e: ## Run slow and integration tests
	$(PYTEST) tests/ -m "slow or integration" -v --tb=short

stress-test: ## Run stress tests (RAG, SQL, Timeline)
	$(PYTEST) tests/stress/ -v -m stress --timeout=120 -x

stress-test-p1: ## Run stress tests for Ponte 1 (RAG)
	$(PYTEST) tests/stress/test_ponte1_rag.py -v -m stress --timeout=120 -x

stress-test-p2: ## Run stress tests for Ponte 2 (NL -> SQL)
	$(PYTEST) tests/stress/test_ponte2_sql.py -v -m stress --timeout=120 -x

stress-test-p3: ## Run stress tests for Ponte 3 (Timeline)
	$(PYTEST) tests/stress/test_ponte3_timeline.py -v -m stress --timeout=120 -x

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

firmware-tflm: ## Install/cache TensorFlow Lite Micro
	$(FIRMWARE_DIR)/scripts/install_tflm.sh

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

# ---------------------------------------------------------------------------
# Camada C11 — Knowledge Layer (RAG + sqlite-vec + MCP)
# ---------------------------------------------------------------------------
knowledge-index:
	@echo "[C11] Reindexando knowledge base..."
	$(UV) run python -m src.knowledge.cli reindex

knowledge-query:
	@read -p "Query: " q; $(UV) run python -m src.knowledge.cli query "$$q"

knowledge-status:
	$(UV) run python -m src.knowledge.cli status

knowledge-test:
	$(UV) run pytest tests/test_knowledge/ -v --tb=short

knowledge-clean:
	rm -f data/knowledge.db
	rm -rf data/lineage/knowledge/
	rm -f logs/knowledge_queries.jsonl
	rm -f data/.dlq/knowledge_rejected.jsonl

knowledge-validate:
	$(UV) run python scripts/validate_knowledge_index.py

knowledge-reindex-if-docs-changed:
	@mkdir -p data/lineage
	@CURRENT=$$(find docs src/knowledge -type f \( -name '*.md' -o -name '*.py' \) | sort | xargs sha256sum | sha256sum | awk '{print $$1}'); \
	if [ -f data/lineage/.knowledge_checksum ] && [ "$$(cat data/lineage/.knowledge_checksum)" = "$$CURRENT" ]; then \
		echo "[C11] Knowledge sources unchanged; skipping reindex."; \
	else \
		echo "[C11] Knowledge sources changed; reindexing..."; \
		$(UV) run python -m src.knowledge.cli reindex; \
		echo "$$CURRENT" > data/lineage/.knowledge_checksum; \
	fi

hybrid-eval: ## Avalia hybrid search no golden dataset
	$(UV) run python scripts/eval_hybrid.py

ragas-eval: ## Avalia qualidade do RAG com golden dataset
	$(UV) run python -m src.observability.ragas_eval_cli data/eval/golden_dataset.json

# ---------------------------------------------------------------------------
# Memory / ArtifactRegistry
# ---------------------------------------------------------------------------
memory-commit:
	$(UV) run python scripts/memory_commit.py --run-id $(RUN_ID) --path $(ARTIFACT_PATH) --type $(ARTIFACT_TYPE)

# ---------------------------------------------------------------------------
# Observabilidade (Prometheus + Grafana)
# ---------------------------------------------------------------------------
observability-up: ## Sobe Prometheus + Grafana + app de métricas
	docker compose up -d observability prometheus grafana

observability-down: ## Derruba stack de observabilidade
	docker compose down observability prometheus grafana

all: env download-all catalog test quality-report ## Run full pipeline: env, download, catalog, test and report

clean: ## Remove processed data, features and model artifacts
	rm -rf $(DATA)/processed/* $(DATA)/features/* models/*.h5 models/*.keras models/*.tflite

clean-raw: ## Remove all raw downloaded datasets
	rm -rf $(DATA)/raw_*

clean-mirrors: ## Remove dataset mirror archives
	rm -rf $(DATA)/mirrors/*
