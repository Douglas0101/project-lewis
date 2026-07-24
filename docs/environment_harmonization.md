# Harmonização de Ambiente Python / Keras / TensorFlow

## Fonte de verdade

| Item | Valor |
|------|-------|
| Interpretador Python | `.venv/bin/python3` (Python 3.12.3) |
| Gerenciador de dependências | `uv` |
| Lockfile | `uv.lock` |
| Manifest | `pyproject.toml` |
| Comando de ativação implícito | `uv run` / `make` |

## Problema resolvido

O projeto possui simultaneamente:

- `keras` 3 standalone;
- `tf-keras` (legado, Keras 2);
- `tensorflow_model_optimization` (QAT/pruning), que depende de `tf_keras`;
- `sentence-transformers`, que importa `transformers`, que pode redirecionar `tf.keras` para `tf_keras`.

Isso causava falhas de desserialização em modelos `.keras` quando testes que importavam essas bibliotecas alteravam o backend de `tf.keras` no processo:

```text
ModuleNotFoundError: No module named 'tf_keras.src.models.functional'
TypeError: Could not locate class 'Functional'
```

## Solução aplicada

1. **`TF_USE_LEGACY_KERAS=0`** é definida antes de qualquer import de TensorFlow:
   - `Makefile` (todos os targets);
   - `Dockerfile`;
   - `docker-compose.yml`;
   - `.env.example`.

2. **Helper `src/models/keras_loader.py`** centraliza o carregamento de modelos `.keras`.
   - Inspeciona `config.json` dentro do ZIP do modelo.
   - Se o artefato for da família `tf_keras`, usa `tf.keras.models.load_model`.
   - Caso contrário, usa `keras.models.load_model` (Keras 3 standalone).
   - Funciona como defesa em profundidade caso a variável acima não esteja setada.

3. **Substituição de `tf.keras.models.load_model`** por `load_keras_model` em todo o código produtivo e nos testes que carregam modelos `.keras`.

## Comandos oficiais

```bash
# Sincronizar ambiente (sem upgrade)
make env

# Lint
make lint

# Type check
make type-check          # mypy
uv run pyright src tests # Pyright/Pylance

# Testes
make test
make test-e2e
```

## Comandos proibidos durante manutenção

- `uv lock --upgrade` sem revisão manual.
- `uv sync` sem confirmar o ambiente de destino.
- Instalação direta com `pip` no ambiente ativo.
- Edição manual de `config.json` dentro de `.keras`.
- Sobrescrição de modelos `.keras` originais.

## Ambientes Conda

O shell pode ter `CONDA_PREFIX` ativo (Python 3.13), mas os comandos oficiais do projeto usam exclusivamente `.venv` Python 3.12. Conda permanece disponível para ferramentas nativas, mas não é a fonte de verdade das dependências Python.

## Rollback

1. Remover `export TF_USE_LEGACY_KERAS := 0` do `Makefile`.
2. Restaurar arquivos originais de `src/models/keras_loader.py` (remover) e reverter substituições de `tf.keras.models.load_model`.
3. Rodar baseline anterior (`git checkout -- <arquivos>`).

## Estado atual

- `make lint`: PASS
- `uv run pyright src tests`: 0 erros, 2 warnings preexistentes no stub do Keras (`verbose=0`).
- `make test`: 655 passados, 1 falha residual em `test_two_stage_qg5.py::test_two_stage_qg5_end_to_end` (métrica de recall do Estágio 1 abaixo do threshold; não relacionada à desserialização).
