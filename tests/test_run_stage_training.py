"""Testes de regressão para scripts de treinamento two-stage.

Garantem que argumentos CLI críticos (ex.: ``--freeze-backbone``) sejam
propagados até ``train_group_kfold``.
"""

from __future__ import annotations

import pytest

from tests.helpers.stage_training import run_stage_script


@pytest.mark.qg5
class TestRunStageTrainingFreezeBackbone:
    """Garante que ``--freeze-backbone`` chegue a ``train_group_kfold``."""

    def test_stage1_propagates_freeze_backbone_true(self):
        kwargs = run_stage_script("stage1", ["--freeze-backbone"])
        assert kwargs["freeze_backbone"] is True

    def test_stage1_default_freeze_backbone_false(self):
        kwargs = run_stage_script("stage1", [])
        assert kwargs["freeze_backbone"] is False

    def test_stage2_propagates_freeze_backbone_true(self):
        kwargs = run_stage_script("stage2", ["--freeze-backbone"])
        assert kwargs["freeze_backbone"] is True

    def test_stage2_default_freeze_backbone_false(self):
        kwargs = run_stage_script("stage2", [])
        assert kwargs["freeze_backbone"] is False
