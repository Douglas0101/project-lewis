"""Pacote de callbacks Keras para monitoramento de treinamento.

Exporta:
    GradientMonitor: monitoramento de gradientes em camadas Dense.
    CalibrationMonitor: monitoramento de ECE, MCE, Brier e reliability diagram.
    F1MacroCheckpoint: seleção e restauração de melhores pesos por métrica AAMI.
"""

from __future__ import annotations

from src.callbacks.calibration_monitor import CalibrationMonitor
from src.callbacks.f1_macro_checkpoint import F1MacroCheckpoint
from src.callbacks.gradient_monitor import GradientMonitor

__all__ = ["CalibrationMonitor", "F1MacroCheckpoint", "GradientMonitor"]
