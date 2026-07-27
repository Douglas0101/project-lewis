# Comandos make — guia de uso (pós-refactor FASE 7)

Flags padronizadas: `DRY_RUN=1` `FORCE=1` `RUN_ID=...` `STAGE=e065|e07` `JSON=1`
Domínio-específicas: `FEATURES=1` (MLP), `TFLM=1` / `STUB=1` (firmware nativo).

```bash
make help                 # ajuda por seção (somente alvos públicos)
make setup                # ambiente + hooks
make doctor               # pré-requisitos do ambiente
make check                # lint + markers + integridade E07R

make data-download-all    # baixa todos os datasets
make data-process         # resample + pré-processamento
make data-features        # feature engineering

make e07r-check           # preflight E07R (9 checks)
make e07r-status          # painel: preflight + células + seleção
make e07r-e065            # E06.5-PD (resume write-once)
make e07r-e065 DRY_RUN=1  # apenas planejamento
make e07r-e065 FORCE=1    # arquiva evidência e re-treina com run-id novo
make e07r-e07             # E07-PD (BLOCKED pré-registrado sai 0)
make e07r-watch STAGE=e065        # dashboard TUI ao vivo
make e07r-watch EXTRA="--once"    # snapshot para logs/CI

make mlp-run              # pipeline MLP v2.3 completo
make mlp-run FEATURES=1   # idem, regenerando features
make mlp-qg5              # testes QG5' do MLP

make fw-build             # compila firmware STM32F4
make fw-test              # testes do firmware sob Renode
make fw-native TFLM=1     # simulador nativo com TFLM
make fw-native STUB=1     # simulador nativo com stub
make gates-ci             # hard gates de CI

make kb-index             # reindexa knowledge base
make kb-query             # consulta interativa

make clean                # limpa processados/features/modelos
```

## Aliases legados (DEPRECATED)

Todos os alvos antigos continuam funcionando — o make imprime um aviso
`DEPRECATED` e delega ao alvo canônico equivalente. Exemplos:

- `download-all` → `data-download-all` (idem para `download-chapman`, `download-mitbih`,
  `catalog`, `process`, `features`, `audit-training-data`, `qg0`, `provenance`,
  `mirror`, `mirror-restore`, `dlq-replay`)
- `mlp-pipeline` → `mlp-run`; `mlp-pipeline-with-features` → `mlp-run FEATURES=1`;
  `mlp-test-qg5` → `mlp-qg5`; `mlp-validate-quantized` → `mlp-validate-quant`
- `e07r-preflight` → `e07r-check`; `e07r-all` → `e07r`;
  `e07r-e065-dry` → `e07r-e065 DRY_RUN=1`; `e07r-e065-fresh` → `e07r-e065 FORCE=1`;
  `e07r-e07-dry` → `e07r-e07 DRY_RUN=1`
- `firmware-*`/`verify-renode`/`check-*`/`hard-gates*` → `fw-*`/`gates-*`
- `knowledge-*` → `kb-*`; `hybrid-eval` → `rag-eval-hybrid`;
  `ragas-eval` → `rag-eval-ragas`; `observability-*` → `obs-*`

## Alvos internos (fora do help, ainda funcionais)

`env`, `pre-commit-install`, `format`, `type-check`, `dev`, `docker-*`,
`mlp-logs-dir`, `mlp-prepare-features`, `firmware-deps`, `firmware-tflm`,
`firmware-tflm-lib`, `all`, `quality-report`, `stress-test*`,
`clean-raw`, `clean-mirrors`, `pretrain`, `finetune`, `quantize`, `export`.

> Aviso de integridade: `finetune`, `mlp-select-best` e `quantize`/`export`
> escrevem em `models/`. Desde o freeze E07R (101 pins), qualquer mudança em
> `models/` é detectada pelo preflight e bloqueia os workflows E07R.
