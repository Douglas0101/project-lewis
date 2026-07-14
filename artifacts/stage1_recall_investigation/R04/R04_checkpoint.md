# R04 checkpoint

## Estado

`HYPOTHESIS_REJECTED`

## Evidência mínima

- Dropout: rate `0.3`, `seed_generator` identificado, uma variável não treinável;
- predict repetível: 3 chamadas idênticas, delta `0.0`;
- `training=False` repetível: 3 chamadas idênticas, delta `0.0`;
- predict ≡ `training=False`: delta `0.0`, argmax/threshold disagreements `0`;
- `training=True`: pesos imutáveis, estado RNG avança, outputs divergem;
- recarregamento determinístico cross-process: `identical = true`;
- nenhum callsite produtivo com `training=True`;
- todos os caminhos produtivos usam `model.predict(...)`;
- hashes de modelo/scaler/threshold/datasets preservados;
- fixture R03 preservada;
- `make lint`: PASS;
- `uv run pyright src tests`: `0/0/0`;
- `git diff --check`: PASS;
- testes focados: `21 passed`;
- QG5: baseline reproduzido, Recall `0.0661458333`.

## Hipótese

`H11 — modo de inferência incorreto = REJECTED`.

## Próxima etapa autorizada

`R05 — contrato modelo–scaler–preprocessing`.
