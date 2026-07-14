# R00 — congelamento do estado atual

## Hipótese

A falha residual é reproduzível no estado atual sem alteração dos artefatos protegidos.

## Evidências

- Git HEAD: `7510fdcb9b2046625ed025aa511355cc3756be03`.
- Estado Git completo: `git_state.txt` (151 entradas no snapshot inicial).
- Ambiente: `environment.json`; `uv` usa Python 3.12, mas `which python` aponta para Conda Python 3.13.
- Teste isolado: `isolated_test_output.txt`, exit 1, recall `0.0661`.
- Suíte: `full_test_output.txt`, exit 2, `1 failed, 655 passed`.
- Baseline validado: `baseline_failure.json`.
- Amostras: 2.048 (128 N, 1.920 Anormal).
- Matriz manual TP/FP/TN/FN: 127/0/128/1.793.
- Threshold efetivo: `0.5800000000000001`.

## Verificações

- [PASS] falha reproduzida isoladamente;
- [PASS] falha reproduzida na suíte completa;
- [PASS] hashes de modelo, scaler e threshold registrados;
- [PASS] distribuição das amostras registrada;
- [PASS] threshold no ponto de decisão registrado;
- [PASS] nenhum modelo, scaler ou threshold escrito pela investigação;
- [NOT RUN] equivalência do loader (R03);
- [NOT RUN] contrato histórico da família (R05/R10).

## Checkpoint

`PASS_WITH_WARNINGS`: o ambiente `uv` aprovado está correto, porém o Python global ativo é 3.13 e deve permanecer fora do pipeline.

Próxima etapa autorizada: `R01 — rastreamento completo do fluxo de inferência`.
