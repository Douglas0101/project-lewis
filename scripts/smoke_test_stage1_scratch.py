"""Smoke test: treina fold 2 do Stage1 do zero (sem pre-treino).

Objetivo: testar se o pre-treino Chapman esta atrapalhando o Estagio 1.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts._smoke_stage1 import run_smoke_test

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("smoke_test_stage1_scratch")


def _setup(model):
    LOGGER.info("Modelo criado do zero (sem pre-treino)")
    return model


def main():
    run_smoke_test(
        name="scratch",
        out_dir=Path("experiments/smoke_stage1_scratch"),
        model_setup=_setup,
        learning_rate=1e-3,
    )


if __name__ == "__main__":
    main()
