# R03 checkpoint

## Estado

`HYPOTHESIS_REJECTED`

## Evidência mínima

- helper: `KERAS_3_STANDALONE` → `keras.saving.load_model`;
- `safe_mode=true`;
- fixture: `(11,500,1)` float32, SHA-256
  `59d08ed5ae4a15e55e45f2c09afd3cebcdf4921782c71e8febd649df81d48794`;
- estruturas equivalentes: `true`;
- 10/10 pesos `array_equal`, delta máximo `0.0`;
- previsões delta máximo/médio/p99: `0.0/0.0/0.0`;
- divergências argmax/threshold: `0/0`;
- compile true/false: delta `0.0`, Adam e loss restaurados;
- hashes protegidos antes/depois: idênticos;
- `make lint`: PASS;
- Pyright: `0/0/0`;
- testes focados: `8 passed`;
- QG5: baseline preservado, Recall `0.0661458333`.

## Hipótese

`H5 — regressão provocada pelo helper = REJECTED`.

## Próxima etapa autorizada

`R04 — modo de inferência e Dropout`.
