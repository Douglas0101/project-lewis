"""Energy reporter para simulação Renode do Project-Lewis.

Este módulo é um stub de especificação v1.4. A implementação concreta deve:
1. Parsear transições de estado de energia do log UART (prefixo PWR).
2. Combinar as durações com o modelo de consumo em
   `firmware/config/power_model_v1.4.yaml`.
3. Retornar métricas de energia para inclusão no relatório JSON.

Veja a especificação completa em `docs/Camada-09-Energia-v1.4.md`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def parse_pwr_transitions(uart_log_text: str) -> list[dict[str, Any]]:
    """Parseia linhas 'PWR <state> <ms>' e retorna transições ordenadas."""
    transitions = []
    for line in uart_log_text.splitlines():
        if line.startswith("PWR "):
            parts = line.split()
            if len(parts) == 3:
                transitions.append({"state": parts[1], "ms": int(parts[2])})
    return transitions


def compute_energy(
    transitions: list[dict[str, Any]],
    power_model: dict[str, Any],
    total_runtime_ms: int,
) -> dict[str, Any]:
    """Calcula durações, carga (mAh), energia (mJ), corrente média e autonomia.

    O `power_model` segue o esquema de `firmware/config/power_model_v1.4.yaml`.
    Retorno esperado:
    {
      "average_current_ma": float,
      "energy_mj_per_beat": float,
      "estimated_autonomy_hours": float,
      "state_durations_ms": dict[str, int],
      "total_charge_mah": float,
      "total_energy_mj": float,
    }

    Algoritmo:
    1. Para cada transição i, duração = transitions[i+1].ms - transitions[i].ms.
    2. A última transição estende até total_runtime_ms.
    3. Para cada estado, multiplica duração pela corrente do power_model.
    4. Soma carga total (mAh) e energia total (mJ).
    5. Deriva average_current_ma e estimated_autonomy_hours.
    """
    raise NotImplementedError("energy_reporter.compute_energy ainda não implementado")


def load_power_model(path: str | Path) -> dict[str, Any]:
    """Carrega o modelo de consumo a partir de um arquivo YAML."""
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
