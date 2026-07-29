"""Shared backbone configuration dataclass (FASE 6)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BackboneSpec:
    """Configuration for a backbone variant."""

    arch: str
    input_len: int = 500
    num_classes: int = 5
    dropout_rate: float = 0.3
