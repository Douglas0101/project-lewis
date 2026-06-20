"""Exceções do SLHA."""


class SLHAError(Exception):
    """Base para erros do SLHA."""


class DiscoveryError(SLHAError):
    """Falha na fase de discovery."""


class WarmupError(SLHAError):
    """Falha na fase de warmup."""


class DecisionError(SLHAError):
    """Falha na fase de decision."""
