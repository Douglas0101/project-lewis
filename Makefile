# Project-Lewis Makefile — camada de orquestração padronizada (FASE 7)
#
# Convenções:
# - Alvos públicos possuem descrição `##` e aparecem em `make help` por seção (##@).
# - Alvos internos/auxiliares não possuem `##` e ficam ocultos do help.
# - Aliases legados executam o alvo canônico via sub-make com aviso DEPRECATED.
# - Flags padronizadas: DRY_RUN=1 FORCE=1 RUN_ID=... STAGE=e065|e07 JSON=1
#   (domínio-específicas: FEATURES=1 TFLM=1 STUB=1).

.PHONY: help setup doctor check lint format type-check test test-e2e clean status watch \
        env pre-commit-install dev docker-build docker-run docker-shell \
        data-download-all data-download-chapman data-download-mitbih data-catalog \
        data-verify-chapman \
        data-process data-features data-audit-train data-qg0 data-provenance \
        data-mirror-create data-mirror-restore data-dlq-replay \
        mlp-run mlp-train mlp-train-stage1 mlp-train-stage2 mlp-select-best \
        mlp-quantize mlp-validate-quant mlp-qg5 mlp-clean mlp-logs-dir mlp-prepare-features \
        e07r e07r-check e07r-status e07r-freeze e07r-e065 e07r-e07 e07r-report e07r-watch \
        fw-build fw-test fw-native fw-run fw-verify-renode fw-check-markers fw-check-no-stub \
        gates-firmware gates-ci \
        kb-index kb-query kb-status kb-test kb-validate kb-clean kb-reindex-changed \
        rag-eval-hybrid rag-eval-ragas \
        obs-up obs-down memory-commit \
        all clean-raw clean-mirrors quality-report stress-test stress-test-p1 stress-test-p2 stress-test-p3 \
        pretrain pretrain-smoke pretrain-check pretrain-validate pretrain-export-smoke pretrain-qg \
        finetune quantize export \
        download-all download-chapman download-mitbih catalog process features \
        audit-training-data qg0 provenance mirror mirror-restore dlq-replay \
        mlp-pipeline mlp-pipeline-with-features mlp-features mlp-test-qg5 mlp-validate-quantized \
        e07r-preflight e07r-all e07r-e065-dry e07r-e065-fresh e07r-e07-dry \
        firmware-deps firmware-tflm firmware-tflm-lib firmware-build firmware-native \
        firmware-native-tflm firmware-native-stub firmware-run firmware-test verify-renode \
        check-strict-markers check-no-stub hard-gates hard-gates-ci \
        knowledge-index knowledge-query knowledge-status knowledge-test knowledge-validate \
        knowledge-clean knowledge-reindex-if-docs-changed \
        hybrid-eval ragas-eval observability-up observability-down

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
# E07R — variáveis (RUN_ID sobrescrevível por alvo; defaults via `or`)
# ---------------------------------------------------------------------------
E07R_EXP          := experiments/stage2_v2.4_research
E07R_PD_CLI       := scripts/run_stage2_e07r_pd.py
E07R_FRESH_SUFFIX := $(shell date +%Y%m%d_%H%M%S)
STAGE             ?= e065
EXTRA             ?=

##@ Geral

help: ## Mostra esta ajuda por seção
	@echo "Uso: make <alvo> [DRY_RUN=1] [FORCE=1] [RUN_ID=...] [STAGE=e065|e07] [JSON=1]"
	@echo "     [FEATURES=1] [TFLM=1] [STUB=1]"
	@awk 'BEGIN {FS = ":.*?## "} \
		/^##@/ {printf "\n== %s ==\n", substr($$0, 5); next} \
		/^[a-zA-Z0-9_-]+:.*?## / {printf "  %-26s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

setup: env pre-commit-install ## Setup completo para novo contribuinte

doctor: ## Verifica se o ambiente local atende aos pré-requisitos
	$(PYTHON) scripts/check_environment.py

check: lint fw-check-markers e07r-check ## Verificação rápida: lint + markers + integridade E07R

lint: ## Roda flake8, mypy e bandit
	$(UV) run flake8 src tests --max-line-length=100
	$(UV) run mypy src --ignore-missing-imports
	$(UV) run bandit -c pyproject.toml -r src

test: ## Roda a suíte pytest completa
	$(PYTEST) tests/ -q --tb=short

test-e2e: ## Roda testes slow e de integração
	$(PYTEST) tests/ -m "slow or integration" -v --tb=short

clean: ## Remove dados processados, features e artefatos de modelos
	rm -rf $(DATA)/processed/* $(DATA)/features/* models/*.h5 models/*.keras models/*.tflite

status: e07r-status ## Atalho: painel de status do projeto (E07R)

watch: e07r-watch

##@ Dados

data-download-all: data-download-chapman data-download-mitbih ## Baixa todos os datasets ECG

data-download-chapman: ## Baixa o dataset Chapman ECG (idempotente; FORCE=1 re-baixa)
	$(PYTHON) -m src.data.download_chapman $(if $(FORCE),--force,)

data-verify-chapman: ## Verifica integridade local do Chapman (offline, sem rede)
	$(PYTHON) -m src.data.download_chapman --verify

data-download-mitbih: ## Baixa MIT-BIH, SVDB, AFDB e INCART
	$(PYTHON) -m src.data.download_mitbih

data-catalog:
	$(PYTHON) -c "from src.data._catalog import build_catalog; build_catalog()"

data-process:
	$(PYTHON) -m src.data.aggregator

data-features:
	$(PYTHON) -m src.features.pipeline

data-audit-train:
	$(PYTHON) scripts/audit_training_data.py

data-qg0:
	$(PYTEST) tests/test_download.py -v

data-provenance:
	$(PYTHON) -c "from src.data._compliance import write_provenance; import json; from pathlib import Path; m = json.loads(Path('src/data/checksums.json').read_text()); write_provenance(m)"

data-mirror-create:
	mkdir -p $(DATA)/mirrors
	tar czf $(DATA)/mirrors/chapman_mirror.tar.gz        -C $(DATA)/raw_chapman .
	tar czf $(DATA)/mirrors/mitbih_family_mirror.tar.gz  -C $(DATA)/raw_mitbih . \
	                                                        -C $(DATA)/raw_svdb   . \
	                                                        -C $(DATA)/raw_afdb   . \
	                                                        -C $(DATA)/raw_incart .

data-mirror-restore:
	mkdir -p $(DATA)/raw_chapman $(DATA)/raw_mitbih $(DATA)/raw_svdb \
	         $(DATA)/raw_afdb $(DATA)/raw_incart
	tar xzf $(DATA)/mirrors/chapman_mirror.tar.gz        -C $(DATA)/raw_chapman/
	tar xzf $(DATA)/mirrors/mitbih_family_mirror.tar.gz  -C $(DATA)/raw_mitbih/

data-dlq-replay:
	$(PYTHON) -m src.data._downloader_replay

##@ MLP v2.3

mlp-run: ## Pipeline v2.3 completo (FEATURES=1 regenera features)
ifeq ($(FEATURES),1)
mlp-run: data-features
endif
mlp-run: mlp-prepare-features mlp-train mlp-select-best mlp-quantize mlp-validate-quant mlp-qg5
	@true

mlp-train: mlp-train-stage1 mlp-train-stage2

mlp-train-stage1: mlp-logs-dir
	$(PYTHON) scripts/train_stage1_mlp.py \
		--hidden-units $(MLP_HIDDEN_UNITS) \
		--max-weight $(MLP_STAGE1_MAX_WEIGHT) \
		--output-dir $(MLP_STAGE1_EXP) 2>&1 | tee -a $(MLP_LOG)

mlp-train-stage2: mlp-logs-dir
	$(PYTHON) scripts/train_stage2_mlp.py \
		--hidden-units $(MLP_HIDDEN_UNITS) \
		--f-oversample-ratio $(MLP_F_OVERSAMPLE_RATIO) \
		--max-weight $(MLP_STAGE2_MAX_WEIGHT) \
		--output-dir $(MLP_STAGE2_EXP) 2>&1 | tee -a $(MLP_LOG)

mlp-select-best: mlp-logs-dir
	$(PYTHON) scripts/select_best_mlp_fold.py \
		--stage1-exp $(MLP_STAGE1_EXP) \
		--stage2-exp $(MLP_STAGE2_EXP) 2>&1 | tee -a $(MLP_LOG)

mlp-quantize: mlp-logs-dir
	$(PYTHON) scripts/quantize_mlp_features.py --n-cal $(MLP_N_CAL) 2>&1 | tee -a $(MLP_LOG)

mlp-validate-quant: mlp-logs-dir
	$(PYTHON) scripts/validate_quantized_mlp.py 2>&1 | tee -a $(MLP_LOG)

mlp-qg5: mlp-logs-dir ## Roda testes QG5' do pipeline MLP v2.3
	$(PYTEST) tests/test_two_stage_mlp_qg5.py tests/test_morphological_features.py -v 2>&1 | tee -a $(MLP_LOG)

mlp-clean:
	rm -rf $(MLP_STAGE1_EXP) $(MLP_STAGE2_EXP)
	rm -f models/*_v2.3*
	rm -f models/quantized/*_v2.3*
	rm -f $(MLP_LOG)

##@ E07R — pipeline patient-disjoint

e07r: e07r-check e07r-e065 e07r-status ## Fluxo completo: preflight → E06.5-PD → status

e07r-check:
	@$(PYTHON) scripts/check_e07r_status.py || { \
		echo "E07R INTEGRITY BLOCKED — fail-closed; veja o check BLOCKED no painel acima (não é erro do Makefile)"; \
		exit 1; \
	}

e07r-status:
	@$(PYTHON) scripts/check_e07r_status.py || { \
		echo "E07R INTEGRITY BLOCKED — fail-closed; veja o check BLOCKED no painel acima (não é erro do Makefile)"; \
		exit 1; \
	}

e07r-freeze:
	@if [ -f $(E07R_EXP)/integrity/e07r_freeze_manifest.json ]; then \
		echo "E07R freeze já publicado (write-once) — nada a fazer"; \
	else \
		$(PYTHON) scripts/freeze_e07r_integrity_v4.py; \
	fi

e07r-e065: ## Executa/resume E06.5-PD 4×5×5 (DRY_RUN=1, FORCE=1, RUN_ID=...)
ifeq ($(DRY_RUN),1)
	$(PYTHON) $(E07R_PD_CLI) e065-pd --run-id $(or $(RUN_ID),e065pd-audit-v2) --dry-run
else ifeq ($(FORCE),1)
	@for d in E06_5_PD cache_pd; do \
		if [ -d $(E07R_EXP)/$$d ]; then \
			mv $(E07R_EXP)/$$d $(E07R_EXP)/$${d}_archive_$(E07R_FRESH_SUFFIX); \
		fi; \
	done
	$(PYTHON) $(E07R_PD_CLI) e065-pd --run-id $(or $(RUN_ID),e065pd-fresh-$(E07R_FRESH_SUFFIX))
else
	$(PYTHON) $(E07R_PD_CLI) e065-pd --run-id $(or $(RUN_ID),e065pd-audit-v2)
endif

e07r-e07: ## Executa E07-PD 6×5×5 (DRY_RUN=1, RUN_ID=...; BLOCKED pré-registrado não é falha)
	@status=0; \
	$(PYTHON) $(E07R_PD_CLI) e07-pd --run-id $(or $(RUN_ID),e07pd-audit-v1) \
		$(if $(filter 1,$(DRY_RUN)),--dry-run) || status=$$?; \
	if [ $$status -eq 10 ]; then \
		echo "E07-PD BLOCKED (pré-registro: sem H*-PD válido) — comportamento esperado"; \
	elif [ $$status -ne 0 ]; then \
		exit $$status; \
	fi

e07r-report:
	@echo "pacote de evidência: $(E07R_EXP)/integrity/e07r_evidence_package.json"
	@echo "relatório consolidado: docs/e07r_evidence_report.md"
	@$(PYTHON) -m json.tool $(E07R_EXP)/E07R_final_checkpoint_20260726.json

e07r-watch: ## Dashboard TUI dos treinamentos (STAGE=e065|e07, EXTRA="--once")
	$(PYTHON) scripts/e07r_watch.py --stage $(STAGE) $(EXTRA)

##@ Firmware & Gates

RENOD_DIR := firmware/tools/renode-1.15.3
RENODE_BIN := $(RENOD_DIR)/renode

fw-build: firmware-tflm-lib ## Compila o binário do firmware STM32F4
	$(MAKE) -C firmware stm32f4

fw-test: firmware-tflm-lib ## Roda testes do firmware sob Renode
	$(MAKE) -C firmware LEWIS_USE_TFLM=1 RENODE_SIMULATION=1 firmware-test

fw-native:
ifeq ($(STUB),1)
	$(MAKE) -C firmware ALLOW_STUB=1 native
else ifeq ($(TFLM),1)
	$(MAKE) -C firmware native-tflm
else
	$(MAKE) -C firmware native
endif

fw-run:
	$(MAKE) -C firmware firmware-run

fw-verify-renode:
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

fw-check-markers:
	@echo "Verificando --strict-markers em pyproject.toml..."
	@$(PYTHON) -c "import tomllib, pathlib, sys; cfg = tomllib.loads(pathlib.Path('pyproject.toml').read_text(encoding='utf-8')); addopts = cfg.get('tool', {}).get('pytest', {}).get('ini_options', {}).get('addopts', ''); sys.exit(0 if '--strict-markers' in addopts.split() else (print('ERROR: --strict-markers nao encontrado em pyproject.toml') or 1))" && echo "OK: --strict-markers configurado"

fw-check-no-stub:
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

gates-firmware: fw-verify-renode ## Hard gates HG-01..HG-06 do firmware
	PYTEST=$(PYTEST) ALLOW_STUB=0 CI=1 $(PYTHON) scripts/run_hard_gates.py

gates-ci: fw-check-markers gates-firmware fw-check-no-stub ## Hard gates de CI (markers + stub)

##@ Knowledge & RAG

kb-index: ## Reindexa a knowledge base (C11)
	@echo "[C11] Reindexando knowledge base..."
	$(UV) run python -m src.knowledge.cli reindex

kb-query: ## Consulta interativa na knowledge base
	@read -p "Query: " q; $(UV) run python -m src.knowledge.cli query "$$q"

kb-status: ## Status da knowledge base
	$(UV) run python -m src.knowledge.cli status

kb-test:
	$(UV) run pytest tests/test_knowledge/ -v --tb=short

kb-validate:
	$(UV) run python scripts/validate_knowledge_index.py

kb-clean:
	rm -f data/knowledge.db
	rm -rf data/lineage/knowledge/
	rm -f logs/knowledge_queries.jsonl
	rm -f data/.dlq/knowledge_rejected.jsonl

kb-reindex-changed:
	@mkdir -p data/lineage
	@CURRENT=$$(find docs src/knowledge -type f \( -name '*.md' -o -name '*.py' \) | sort | xargs sha256sum | sha256sum | awk '{print $$1}'); \
	if [ -f data/lineage/.knowledge_checksum ] && [ "$$(cat data/lineage/.knowledge_checksum)" = "$$CURRENT" ]; then \
		echo "[C11] Knowledge sources unchanged; skipping reindex."; \
	else \
		echo "[C11] Knowledge sources changed; reindexing..."; \
		$(UV) run python -m src.knowledge.cli reindex; \
		echo "$$CURRENT" > data/lineage/.knowledge_checksum; \
	fi

rag-eval-hybrid:
	$(UV) run python scripts/eval_hybrid.py

rag-eval-ragas:
	$(UV) run python -m src.observability.ragas_eval_cli data/eval/golden_dataset.json

##@ Observabilidade & Memória

obs-up: ## Sobe Prometheus + Grafana + app de métricas
	docker compose up -d observability prometheus grafana

obs-down: ## Derruba stack de observabilidade
	docker compose down observability prometheus grafana

memory-commit: ## Registra artefato no ArtifactRegistry (RUN_ID=... ARTIFACT_PATH=... ARTIFACT_TYPE=...)
	$(UV) run python scripts/memory_commit.py --run-id $(RUN_ID) --path $(ARTIFACT_PATH) --type $(ARTIFACT_TYPE)

# ===========================================================================
# Alvos internos/auxiliares (funcionais, ocultos do help)
# ===========================================================================

env:
	$(UV) sync --frozen

pre-commit-install:
	$(UV) run pre-commit install

format:
	$(UV) run black src tests
	$(UV) run isort src tests

type-check:
	$(UV) run mypy src --ignore-missing-imports

dev:
	docker compose up -d app && docker compose exec app bash

docker-build:
	docker build -t project-lewis:latest .

docker-run: docker-build
	docker run --rm -it -v $(PWD):/app -v lewis-data:/app/data project-lewis:latest

docker-shell: docker-build
	docker run --rm -it -v $(PWD):/app -v lewis-data:/app/data project-lewis:latest bash

mlp-logs-dir:
	@mkdir -p logs

mlp-prepare-features: mlp-logs-dir
	$(PYTHON) scripts/prepare_stage1_features.py 2>&1 | tee -a $(MLP_LOG)
	$(PYTHON) scripts/prepare_stage2_features.py 2>&1 | tee -a $(MLP_LOG)

firmware-deps:
	$(MAKE) -C firmware firmware-deps

firmware-tflm:
	$(FIRMWARE_DIR)/scripts/install_tflm.sh

firmware-tflm-lib:
	$(MAKE) -C firmware tflm-lib

all: env data-download-all data-catalog test quality-report

quality-report:
	$(UV) run python scripts/generate_quality_report.py

stress-test:
	$(PYTEST) tests/stress/ -v -m stress --timeout=120 -x

stress-test-p1:
	$(PYTEST) tests/stress/test_ponte1_rag.py -v -m stress --timeout=120 -x

stress-test-p2:
	$(PYTEST) tests/stress/test_ponte2_sql.py -v -m stress --timeout=120 -x

stress-test-p3:
	$(PYTEST) tests/stress/test_ponte3_timeline.py -v -m stress --timeout=120 -x

clean-raw:
	rm -rf $(DATA)/raw_*

clean-mirrors:
	rm -rf $(DATA)/mirrors/*

##@ Pré-treino

pretrain: ## Pré-treino Chapman via wrapper (30 épocas; exit 0 se execução OK)
	$(PYTHON) scripts/pretrain_wrapper.py

pretrain-qg: ## Pré-treino com gate QG4 bloqueante (exit 10 se QG4 falhar)
	$(PYTHON) scripts/pretrain_wrapper.py --enforce-qg4

pretrain-smoke: ## Smoke de engenharia do pré-treino (1 época; QG4 informativo)
	$(PYTHON) scripts/pretrain_wrapper.py --smoke

pretrain-check: ## Lint + testes rápidos do pipeline de pré-treino
	$(UV) run flake8 src/models/pretrain_chapman.py src/models/chapman_dataset.py src/models/pretrain_provenance.py src/models/pretrain_losses.py src/models/pretrain_evaluation.py src/models/backbones/ scripts/pretrain_wrapper.py scripts/validate_pretrain_artifacts.py --max-line-length=100
	$(PYTEST) tests/test_chapman_dataset.py tests/test_pretrain.py tests/test_pretrain_pipeline.py tests/test_pretrain_artifacts.py tests/test_backbone_budget.py tests/test_pretrain_evaluation.py tests/test_qg4.py -q -m "not slow"

pretrain-validate: ## Valida artefatos do último run de pré-treino
	$(PYTHON) scripts/validate_pretrain_artifacts.py

pretrain-export-smoke: ## Exporta TFLite float32/INT8 e valida FlatBuffer < 64KB
	$(PYTHON) scripts/export_tflite_smoke.py

finetune:
	$(PYTHON) -m src.models.finetune_mitbih

quantize:
	$(PYTHON) -m src.quantization.ptq

export:
	$(PYTHON) -m src.quantization.export_tflite

# ===========================================================================
# Aliases legados (DEPRECATED) — compatibilidade total via sub-make
# ===========================================================================

download-all:
	@echo "DEPRECATED: 'make download-all' → 'make data-download-all'" >&2
	@$(MAKE) --no-print-directory data-download-all

download-chapman:
	@echo "DEPRECATED: 'make download-chapman' → 'make data-download-chapman'" >&2
	@$(MAKE) --no-print-directory data-download-chapman

download-mitbih:
	@echo "DEPRECATED: 'make download-mitbih' → 'make data-download-mitbih'" >&2
	@$(MAKE) --no-print-directory data-download-mitbih

catalog:
	@echo "DEPRECATED: 'make catalog' → 'make data-catalog'" >&2
	@$(MAKE) --no-print-directory data-catalog

process:
	@echo "DEPRECATED: 'make process' → 'make data-process'" >&2
	@$(MAKE) --no-print-directory data-process

features:
	@echo "DEPRECATED: 'make features' → 'make data-features'" >&2
	@$(MAKE) --no-print-directory data-features

audit-training-data:
	@echo "DEPRECATED: 'make audit-training-data' → 'make data-audit-train'" >&2
	@$(MAKE) --no-print-directory data-audit-train

qg0:
	@echo "DEPRECATED: 'make qg0' → 'make data-qg0'" >&2
	@$(MAKE) --no-print-directory data-qg0

provenance:
	@echo "DEPRECATED: 'make provenance' → 'make data-provenance'" >&2
	@$(MAKE) --no-print-directory data-provenance

mirror:
	@echo "DEPRECATED: 'make mirror' → 'make data-mirror-create'" >&2
	@$(MAKE) --no-print-directory data-mirror-create

mirror-restore:
	@echo "DEPRECATED: 'make mirror-restore' → 'make data-mirror-restore'" >&2
	@$(MAKE) --no-print-directory data-mirror-restore

dlq-replay:
	@echo "DEPRECATED: 'make dlq-replay' → 'make data-dlq-replay'" >&2
	@$(MAKE) --no-print-directory data-dlq-replay

mlp-pipeline:
	@echo "DEPRECATED: 'make mlp-pipeline' → 'make mlp-run'" >&2
	@$(MAKE) --no-print-directory mlp-run

mlp-pipeline-with-features:
	@echo "DEPRECATED: 'make mlp-pipeline-with-features' → 'make mlp-run FEATURES=1'" >&2
	@$(MAKE) --no-print-directory mlp-run FEATURES=1

mlp-features:
	@echo "DEPRECATED: 'make mlp-features' → 'make mlp-run FEATURES=1'" >&2
	@$(MAKE) --no-print-directory mlp-run FEATURES=1

mlp-test-qg5:
	@echo "DEPRECATED: 'make mlp-test-qg5' → 'make mlp-qg5'" >&2
	@$(MAKE) --no-print-directory mlp-qg5

mlp-validate-quantized:
	@echo "DEPRECATED: 'make mlp-validate-quantized' → 'make mlp-validate-quant'" >&2
	@$(MAKE) --no-print-directory mlp-validate-quant

e07r-preflight:
	@echo "DEPRECATED: 'make e07r-preflight' → 'make e07r-check'" >&2
	@$(MAKE) --no-print-directory e07r-check

e07r-all:
	@echo "DEPRECATED: 'make e07r-all' → 'make e07r'" >&2
	@$(MAKE) --no-print-directory e07r

e07r-e065-dry:
	@echo "DEPRECATED: 'make e07r-e065-dry' → 'make e07r-e065 DRY_RUN=1'" >&2
	@$(MAKE) --no-print-directory e07r-e065 DRY_RUN=1

e07r-e065-fresh:
	@echo "DEPRECATED: 'make e07r-e065-fresh' → 'make e07r-e065 FORCE=1'" >&2
	@$(MAKE) --no-print-directory e07r-e065 FORCE=1

e07r-e07-dry:
	@echo "DEPRECATED: 'make e07r-e07-dry' → 'make e07r-e07 DRY_RUN=1'" >&2
	@$(MAKE) --no-print-directory e07r-e07 DRY_RUN=1

firmware-build:
	@echo "DEPRECATED: 'make firmware-build' → 'make fw-build'" >&2
	@$(MAKE) --no-print-directory fw-build

firmware-test:
	@echo "DEPRECATED: 'make firmware-test' → 'make fw-test'" >&2
	@$(MAKE) --no-print-directory fw-test

firmware-native:
	@echo "DEPRECATED: 'make firmware-native' → 'make fw-native'" >&2
	@$(MAKE) --no-print-directory fw-native

firmware-native-tflm:
	@echo "DEPRECATED: 'make firmware-native-tflm' → 'make fw-native TFLM=1'" >&2
	@$(MAKE) --no-print-directory fw-native TFLM=1

firmware-native-stub:
	@echo "DEPRECATED: 'make firmware-native-stub' → 'make fw-native STUB=1'" >&2
	@$(MAKE) --no-print-directory fw-native STUB=1

firmware-run:
	@echo "DEPRECATED: 'make firmware-run' → 'make fw-run'" >&2
	@$(MAKE) --no-print-directory fw-run

verify-renode:
	@echo "DEPRECATED: 'make verify-renode' → 'make fw-verify-renode'" >&2
	@$(MAKE) --no-print-directory fw-verify-renode

check-strict-markers:
	@echo "DEPRECATED: 'make check-strict-markers' → 'make fw-check-markers'" >&2
	@$(MAKE) --no-print-directory fw-check-markers

check-no-stub:
	@echo "DEPRECATED: 'make check-no-stub' → 'make fw-check-no-stub'" >&2
	@$(MAKE) --no-print-directory fw-check-no-stub

hard-gates:
	@echo "DEPRECATED: 'make hard-gates' → 'make gates-firmware'" >&2
	@$(MAKE) --no-print-directory gates-firmware

hard-gates-ci:
	@echo "DEPRECATED: 'make hard-gates-ci' → 'make gates-ci'" >&2
	@$(MAKE) --no-print-directory gates-ci

knowledge-index:
	@echo "DEPRECATED: 'make knowledge-index' → 'make kb-index'" >&2
	@$(MAKE) --no-print-directory kb-index

knowledge-query:
	@echo "DEPRECATED: 'make knowledge-query' → 'make kb-query'" >&2
	@$(MAKE) --no-print-directory kb-query

knowledge-status:
	@echo "DEPRECATED: 'make knowledge-status' → 'make kb-status'" >&2
	@$(MAKE) --no-print-directory kb-status

knowledge-test:
	@echo "DEPRECATED: 'make knowledge-test' → 'make kb-test'" >&2
	@$(MAKE) --no-print-directory kb-test

knowledge-validate:
	@echo "DEPRECATED: 'make knowledge-validate' → 'make kb-validate'" >&2
	@$(MAKE) --no-print-directory kb-validate

knowledge-clean:
	@echo "DEPRECATED: 'make knowledge-clean' → 'make kb-clean'" >&2
	@$(MAKE) --no-print-directory kb-clean

knowledge-reindex-if-docs-changed:
	@echo "DEPRECATED: 'make knowledge-reindex-if-docs-changed' → 'make kb-reindex-changed'" >&2
	@$(MAKE) --no-print-directory kb-reindex-changed

hybrid-eval:
	@echo "DEPRECATED: 'make hybrid-eval' → 'make rag-eval-hybrid'" >&2
	@$(MAKE) --no-print-directory rag-eval-hybrid

ragas-eval:
	@echo "DEPRECATED: 'make ragas-eval' → 'make rag-eval-ragas'" >&2
	@$(MAKE) --no-print-directory rag-eval-ragas

observability-up:
	@echo "DEPRECATED: 'make observability-up' → 'make obs-up'" >&2
	@$(MAKE) --no-print-directory obs-up

observability-down:
	@echo "DEPRECATED: 'make observability-down' → 'make obs-down'" >&2
	@$(MAKE) --no-print-directory obs-down
