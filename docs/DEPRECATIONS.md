# DEPRECATIONS — supressões escopadas de warnings third-party

Data: 2026-07-29 | Origem: SDD_Otimizacoes_Warnings (OPT-006, fallback documentado)

Este arquivo registra as supressões de warnings em `pyproject.toml
[tool.pytest.ini_options] filterwarnings` que **não** são corrigíveis no nosso
código — a fonte está em dependências de terceiros sem correção upstream
disponível na versão compatível com o projeto. Cada supressão é escopada por
mensagem **e** módulo (nunca global).

## 1. `tensorflow_model_optimization` — distutils.version

- **Warning**: `DeprecationWarning: distutils Version classes are deprecated. Use packaging.version instead.`
- **Origem**: `tensorflow_model_optimization/__init__.py:65-66` (tfmot 0.8.1).
- **Por que suprimir**: o uso de `distutils.version.LooseVersion` é interno do
  tfmot; a versão 0.8.1 é a compatível com TF 2.21 + QAT/pruning do projeto.
  Upgrade de tfmot não disponível no canal estável para esta combinação.
- **Revisão pendente**: remover a supressão quando tfmot publicar release com
  `packaging.version`.

## 2. `keras` × numpy 2.x — `__array__` copy keyword (OPT-001, reversão)

- **Warning**: `DeprecationWarning: __array__ implementation doesn't accept a copy keyword, so passing copy=False failed.`
- **Origem**: `keras/src/backend/tensorflow/core.py` (keras 3.14.1 interno,
  `return np.array(x)` sob numpy 2.x).
- **Decisão (2026-07-29)**: o SDD OPT-001 designou downgrade para `numpy<2.0`.
  Ele foi aplicado e funcionou localmente, **mas** divergiu `pyproject.toml` do
  `uv.lock` — que é **pinado no freeze E07R** (write-once, fail-closed) — e
  quebraria o `uv sync --frozen` do CI. A opção de re-pin do lock foi descartada
  por exigir cirurgia no manifesto de governança. **Reversão**: manter
  `numpy>=1.26.4` e suprimir o warning escopadamente (third-party, sem fix na
  versão do lock). Venv local mantém numpy 1.26.4 (diferença local×CI benigna).
- **Revisão pendente**: remover quando o lock do Keras/TF for atualizado com
  suporte nativo ao kwarg `copy` (e o freeze E07R for re-publicado).

## 2. `starlette` TestClient — httpx

- **Warning**: `StarletteDeprecationWarning: Using httpx with starlette.testclient is deprecated; install httpx2 instead.`
- **Origem**: `fastapi/testclient.py:1` (fastapi 0.138.2 + starlette 1.3.1 + httpx 0.28.1).
- **Por que suprimir**: já estamos nas versões mais recentes disponíveis; o
  pacote `httpx2` sugerido pela mensagem **não existe no PyPI** (a deprecação é
  futurista/antecipatória do upstream). Uso restrito a testes.
- **Revisão pendente**: reavaliar quando `httpx2` for publicado ou quando
  fastapi/starlette ajustarem o TestClient.

## 3. Supressões já existentes (mantidas, mesma política)

- `gast` AST kwarg (TF 2.21 + Python 3.13) — upstream TF.
- `Statistics for quantized inputs` (TFLite converter) — benigno no pipeline PTQ.
- `tf.lite.Interpreter is deprecated` — migração LiteRT rastreada separadamente.

## Regra

Nenhuma supressão nova entra sem: (1) origem identificada em código
third-party, (2) escopo por mensagem+módulo, (3) registro neste arquivo,
(4) condição de remoção.
