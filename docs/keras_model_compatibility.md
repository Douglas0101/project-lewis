# Compatibilidade de Modelos `.keras`

## Formato dos artefatos

Os modelos do projeto são salvos no formato `.keras` (ZIP com `config.json`).
Foram identificadas duas famílias de serialização:

| Família | Módulo no `config.json` | Loader compatível |
|---------|--------------------------|-------------------|
| Keras 3 standalone | `keras.src.models.functional` | `keras.models.load_model` |
| tf-keras legado | `tf_keras.src.models.functional` | `tf.keras.models.load_model` (com `TF_USE_LEGACY_KERAS=1`) |

## Inventário

- 247 arquivos `.keras` inspecionados.
- Modelos principais (`models/stage1_*.keras`, `models/stage2_*.keras`, `experiments/**/model.keras`) são majoritariamente da família **Keras 3 standalone**.
- Alguns artefatos gerados após importar `tf_keras` podem usar a família **tf_keras**.

## Loader oficial

Use sempre:

```python
from src.models.keras_loader import load_keras_model

model = load_keras_model("models/stage1_float32_v2.0.keras", compile=False)
```

O helper detecta a família do artefato e escolhe o loader correto.

## O que não fazer

- Não chamar `tf.keras.models.load_model` diretamente em código novo.
- Não presumir que todos os `.keras` têm a mesma origem.
- Não editar manualmente `config.json` dentro do ZIP como solução definitiva.
- Não sobrescrever modelos originais ao migrar.

## Migração (quando necessária)

Se for preciso converter um modelo de `tf_keras` para Keras 3:

1. Carregue no ambiente compatível com a origem.
2. Gere entradas de referência fixas.
3. Salve predições de referência.
4. Reconstrua a arquitetura no runtime Keras 3.
5. Transfira pesos via API suportada.
6. Valide shapes, nomes de camadas e predições.
7. Salve com novo nome (`model_keras3_migrated_v1.keras`).
8. Crie `migration_manifest.json` com hashes e tolerâncias.

## Referência

- `src/models/keras_loader.py`
- `docs/environment_harmonization.md`
