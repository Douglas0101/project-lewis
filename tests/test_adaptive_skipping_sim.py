"""Testes de simulação do Adaptive Inference Skipping.

Esta suíte replica em Python a lógica do firmware C para validar o
comportamento do algoritmo sem depender do build embarcado.  Os dados são
sintéticos e pequenos para execução rápida.
"""

from __future__ import annotations

import pytest


class AdaptiveSkippingSimulator:
    """Simulador Python do adaptive skipping embarcado.

    Mantém um histórico circular dos últimos intervalos RR e decide se a
    próxima inferência deve ser pulada quando o ritmo cardíaco é estável.
    """

    MAX_WINDOW = 5
    MIN_CYCLES = 3

    def __init__(self) -> None:
        """Inicializa o simulador com histórico vazio."""
        self.rr_history: list[int] = []
        self.last_class: int | None = None
        self.has_last_class = False

    def reset(self) -> None:
        """Limpa o histórico de RR e a última classe conhecida."""
        self.rr_history.clear()
        self.last_class = None
        self.has_last_class = False

    def feed_rr(self, rr_ms: int) -> None:
        """Adiciona um novo intervalo RR ao histórico circular."""
        self.rr_history.append(rr_ms)
        if len(self.rr_history) > self.MAX_WINDOW:
            self.rr_history.pop(0)

    def update_class(self, class_id: int) -> None:
        """Atualiza a última classe conhecida."""
        self.last_class = class_id
        self.has_last_class = True

    def should_skip(
        self,
        rr_ms: int,
        threshold_ms: int,
        threshold_ratio: float,
    ) -> bool:
        """Decide se a inferência atual deve ser pulada.

        O skipping ocorre quando há pelo menos ``MIN_CYCLES`` RR no histórico,
        incluindo o valor atual, e a variação absoluta e relativa do conjunto
        estiver abaixo dos limites configuráveis.

        Parameters
        ----------
        rr_ms : int
            Intervalo RR atual em milissegundos.
        threshold_ms : int
            Limiar de variação absoluta máxima permitida (ms).
        threshold_ratio : float
            Limiar de variação relativa máxima permitida (variação / média).

        Returns
        -------
        bool
            ``True`` se a inferência deve ser pulada.
        """
        self.feed_rr(rr_ms)
        if len(self.rr_history) < self.MIN_CYCLES:
            return False
        if not self.has_last_class:
            return False

        values = self.rr_history
        mean = sum(values) / len(values)
        max_abs = max(abs(v - mean) for v in values)
        ratio = max_abs / mean if mean > 0.0 else float("inf")

        return (max_abs <= threshold_ms) and (ratio <= threshold_ratio)


class TestAdaptiveSkippingSimulator:
    """Testes comportamentais do simulador de adaptive skipping."""

    def test_empty_history_never_skips(self) -> None:
        """Com histórico vazio, nenhum skipping deve ocorrer."""
        sim = AdaptiveSkippingSimulator()
        sim.update_class(0)
        assert not sim.should_skip(800, 50, 0.05)

    def test_less_than_three_cycles_never_skips(self) -> None:
        """Com menos de 3 ciclos no total, o algoritmo ainda não confia na estabilidade."""
        sim = AdaptiveSkippingSimulator()
        sim.update_class(0)
        sim.feed_rr(800)
        assert not sim.should_skip(801, 50, 0.05)

    def test_stable_rhythm_triggers_skip(self) -> None:
        """RR estáveis por 3+ ciclos devem ativar o skipping."""
        sim = AdaptiveSkippingSimulator()
        sim.update_class(2)
        sim.feed_rr(800)
        sim.feed_rr(800)
        assert sim.should_skip(800, 10, 0.05)

    def test_unstable_rhythm_does_not_skip(self) -> None:
        """RR com variação alta não deve ativar o skipping."""
        sim = AdaptiveSkippingSimulator()
        sim.update_class(2)
        sim.feed_rr(600)
        sim.feed_rr(800)
        assert not sim.should_skip(1000, 10, 0.05)

    def test_window_does_not_exceed_five(self) -> None:
        """A janela de histórico deve ser limitada a 5 ciclos."""
        sim = AdaptiveSkippingSimulator()
        sim.update_class(0)
        for rr in [700, 800, 900, 850, 825, 800, 790]:
            sim.feed_rr(rr)
        assert len(sim.rr_history) == 5

    def test_absolute_threshold_only_blocks_skip(self) -> None:
        """Variação relativa pequena, mas absoluta acima do limite, não pula."""
        sim = AdaptiveSkippingSimulator()
        sim.update_class(0)
        sim.feed_rr(1000)
        sim.feed_rr(1000)
        assert not sim.should_skip(1100, 50, 0.10)

    def test_relative_threshold_only_blocks_skip(self) -> None:
        """Variação absoluta pequena, mas relativa acima do limite, não pula."""
        sim = AdaptiveSkippingSimulator()
        sim.update_class(0)
        sim.feed_rr(200)
        sim.feed_rr(200)
        assert not sim.should_skip(220, 50, 0.05)

    def test_no_last_class_never_skips(self) -> None:
        """Sem classe anterior conhecida, não há o que repetir; não pula."""
        sim = AdaptiveSkippingSimulator()
        sim.feed_rr(800)
        sim.feed_rr(800)
        assert not sim.should_skip(800, 10, 0.05)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
