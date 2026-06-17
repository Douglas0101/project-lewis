# Project-Lewis — Camada 9.1: Modelagem de Energia no Renode
## Responsável: Firmware / QA

**Versão:** 1.4 | **Data:** 2026-06-17

## 1. Objetivo
Estimar o consumo de energia do firmware Project-Lewis no Renode antes de
testes em hardware físico, produzindo métricas transferíveis para análise de
autonomia de bateria.

## 2. Arquitetura

```
+----------------+      logs UART      +------------------------+
| Firmware C     |  ---------------->  | run_renode_tests.py    |
| (estados PWR)  |                     | (já parseia latência)  |
+----------------+                     +-----------+------------+
                                                     |
                                                     v
                                          +------------------------+
                                          | energy_reporter.py     |
                                          | - lê log UART          |
                                          | - lê power_model.yaml  |
                                          | - calcula mAh/mJ       |
                                          +-----------+------------+
                                                     |
                                                     v
                                          +------------------------+
                                          | firmware_simulation_   |
                                          | report.json (com campo |
                                          | energy)                |
                                          +------------------------+
```

## 3. Modelo de Consumo

### 3.1 STM32F407VG (MCU)
Baseado no datasheet DS8626 (Rev 11), VDD = 3.3 V, TA = 25 °C.
A linha "Run @ 168 MHz, periféricos ON" assume UART4, TIM2 e Flash habilitados,
próximo do cenário de simulação atual:

| Estado | Corrente Típica | Fonte |
| :--- | :--- | :--- |
| Run @ 168 MHz, periféricos ON (UART4 + TIM2 + Flash) | ~90 mA | Figura 25 / Tabela 22 (extrapolado) |
| Sleep @ 168 MHz | ~60 mA | Tabela 22 |
| Stop (regulador LP) | ~350 µA | Tabela 23 |
| Standby | ~3 µA | Tabela 24 |
| UART4 habilitado | +3.3 µA/MHz | Tabela 28 |
| TIM2 habilitado | +16.5 µA/MHz | Tabela 28 |

### 3.2 ADS1292R (AFE)
Baseado no datasheet SBAS502C:

| Modo | Consumo | Nota |
| :--- | :--- | :--- |
| Normal mode, 1 canal @ 500 SPS | ~335 µW/canal | AVDD=3 V, DVDD=1.8 V |
| Standby | ~160 µW | |
| Power-down | ~1 µW | |

### 3.3 Tensão e Bateria de Referência
- VDD nominal: 3.3 V
- Capacidade de referência: 500 mAh (Li-Po 3.7 V -> LDO 3.3 V)

## 4. Instrumentação do Firmware

Adicionar ao [`../firmware/src/hal/hal.h`](../firmware/src/hal/hal.h):
```c
typedef enum {
    LEWIS_PWR_ACTIVE,
    LEWIS_PWR_INFERENCE,
    LEWIS_PWR_SLEEP
} lewis_pwr_state_t;

void lewis_hal_pwr_set_state(lewis_pwr_state_t state);
```

Implementar em [`../firmware/src/hal/simulator/hal_sim.c`](../firmware/src/hal/simulator/hal_sim.c):
```c
void lewis_hal_pwr_set_state(lewis_pwr_state_t state) {
    const char* name = (state == LEWIS_PWR_INFERENCE) ? "inference"
                     : (state == LEWIS_PWR_SLEEP)     ? "sleep"
                     :                                   "active";
    lewis_debug_print("PWR ");
    lewis_debug_print(name);
    lewis_debug_print(" ");
    lewis_debug_print_uint(lewis_hal_millis());
    lewis_debug_print("\n");
}
```

Usar os hooks no [`../firmware/src/app/main.c`](../firmware/src/app/main.c):
- `lewis_hal_pwr_set_state(LEWIS_PWR_INFERENCE)` antes de `lewis_inference_run()`.
- `lewis_hal_pwr_set_state(LEWIS_PWR_ACTIVE)` após inferência.
- `LEWIS_PWR_SLEEP` durante espera por próximo batimento (WFI), quando o loop
  principal for adaptado para duty-cycle entre batimentos.

## 5. Especificação da Extensão do Runner Renode

Criar [`../firmware/scripts/energy_reporter.py`](../firmware/scripts/energy_reporter.py)
com a seguinte interface (implementação concreta na próxima iteração de firmware):

```python
def parse_pwr_transitions(uart_log_text: str) -> list[dict]:
    """Parseia linhas 'PWR <state> <ms>' e retorna transições ordenadas."""
    transitions = []
    for line in uart_log_text.splitlines():
        if line.startswith("PWR "):
            parts = line.split()
            transitions.append({"state": parts[1], "ms": int(parts[2])})
    return transitions

def compute_energy(transitions: list[dict], power_model: dict) -> dict:
    """Calcula durações, carga (mAh), energia (mJ), corrente média e autonomia."""
    # Duração total e por estado; último estado estende até o tempo final
    # da simulação (obtido do log UART ou parâmetro run_time).
    ...
```

Integrar em [`../firmware/scripts/run_renode_tests.py`](../firmware/scripts/run_renode_tests.py):
- Importar `energy_reporter`.
- Após `parse_uart_log`, chamar `energy_reporter.compute_energy()`.
- Incluir no `report` JSON:
  ```json
  "energy": {
    "average_current_ma": 42.5,
    "energy_mj_per_beat": 2.24,
    "estimated_autonomy_hours": 11.7,
    "state_durations_ms": {"active": 984, "inference": 16, "sleep": 0}
  }
  ```

## 6. Quality Gate QG19 — Consumo Energético

| Critério | Valor | Como Validar |
| :--- | :--- | :--- |
| Corrente média @ 1 batimento/s | < 50 mA | `report["energy"]["average_current_ma"]` |
| Energia por batimento | < 5 mJ | `report["energy"]["energy_mj_per_beat"]` |
| Autonomia estimada (500 mAh) | > 8 h | `report["energy"]["estimated_autonomy_hours"]` |
| Estados instrumentados | ≥ 2 (active + inference) | presença de logs `PWR` |
