"""Smoke test: treina fold 2 do Stage1 com backbone descongelado.

Objetivo: testar hipótese de que congelar toda a torre conv + embedding
impossibilita adaptação ao domínio MIT-BIH, resultando em AUC ~0.56.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts._smoke_stage1 import run_smoke_test

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("smoke_test_stage1_unfrozen")


def _setup(model):
    pretrained = Path("experiments/20260621_034237_pretrain_chapman/backbone_pretrained.keras")
    if pretrained.exists():
        from src.models.backbone_1d import load_backbone_weights_from_pretrained
        model = load_backbone_weights_from_pretrained(pretrained, model)
        LOGGER.info("Pesos pré-treinados carregados")

    for layer in model.layers:
        layer.trainable = True
    LOGGER.info("Todas as camadas descongeladas")
    return model


def main():
    run_smoke_test(
        name="unfrozen",
        out_dir=Path("experiments/smoke_stage1_unfrozen"),
        model_setup=_setup,
        learning_rate=1e-4,
    )


if __name__ == "__main__":
    main()
