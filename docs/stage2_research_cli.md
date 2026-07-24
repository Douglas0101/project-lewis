# CLI canônico de pesquisa Stage 2

Entrada única:

```bash
uv run --locked python -m src.cli.stage2_research <comando>
```

## Invariantes congeladas

- outer folds e inner splits são patient-disjoint e independentes das seeds do modelo;
- outer test nunca participa de fitting, sampling, prior/class-count estimation ou seleção;
- baseline/H6/H11/H12 usam MLP-128, amostragem natural, sparse CE e argmax do softmax;
- E07 altera somente o sampler train-only;
- E08 altera somente o tratamento long-tail/classificador;
- modo audit é determinístico, serial (`max_parallel=1`) e CPU;
- cada célula é vinculada aos hashes de preflight, source tree, runtime, dataset, split e features;
- `DONE` só é criado após save/reload e verificação do conjunto exato de artefatos;
- `--force` nunca sobrescreve célula finalizada.

## Sequência operacional

### 1. Preflight

```bash
uv run --locked python -m src.cli.stage2_research \
  preflight --config config/stage2_research.yaml --deterministic --device cpu
```

### 2. Plano E06.5

```bash
uv run --locked python -m src.cli.stage2_research \
  plan --stage e06.5 --candidates baseline,H6,H11,H12 \
  --folds 1,2,3,4,5 --seeds 17,29,43,71,101
```

### 3. Smoke canônico

```bash
uv run --locked python -m src.cli.stage2_research \
  e065-run --candidates baseline,H6,H11,H12 --folds 1 --seeds 17 \
  --profile smoke --deterministic --device cpu --max-parallel 1
```

O audit de 100 células é bloqueado até existir um `E06_5_SMOKE_PASS` válido para exatamente as quatro células acima.

### 4. Audit E06.5 — somente após autorização

```bash
uv run --locked python -m src.cli.stage2_research \
  e065-run --candidates baseline,H6,H11,H12 --folds 1,2,3,4,5 \
  --seeds 17,29,43,71,101 --profile audit --deterministic \
  --device cpu --max-parallel 1 --resume
```

Depois: `fold-audit`, `representation-select` e `verify --stage e06.5`.
E07/E08 permanecem bloqueados até a release E06.5 ser validada.

## Exit codes

| Código | Significado |
| ---: | --- |
| 0 | PASS |
| 2 | argumentos inválidos |
| 3 | precondição bloqueada |
| 4 | regressão |
| 5 | experimento inválido |
| 6 | integridade de dados |
| 7 | leakage |
| 8 | artefato incompatível/stale |
| 9 | interrupção retomável |
| 10 | falha de treinamento |
| 11 | falha de avaliação |
| 12 | experimento válido, gate científico não atingido |

## Artefatos

Raiz: `experiments/stage2_v2.4_research/`.
Cada célula finalizada contém manifest de execução/ambiente/preprocessamento/sampling/método, métricas, previsões, histórico, checkpoint Keras, logs, hashes e `DONE`.
