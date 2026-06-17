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

### 3.4 Mapeamento Estados do Firmware → Correntes do Modelo

| Estado Firmware (`PWR <state>`) | Estado MCU + AFE | Corrente Estimada | Nota |
| :--- | :--- | :--- | :--- |
| `active` | Run @ 168 MHz + UART4 + TIM2 + ADS1292R Normal | ~90 mA + AFE | Entre batimentos, processando UART/comandos |
| `inference` | Run @ 168 MHz + FPU/DSP + TFLM + ADS1292R Normal | ~95 mA + AFE | Durante `lewis_inference_run()` |
| `sleep` | Sleep @ 168 MHz + UART4 + TIM2 + ADS1292R Normal | ~60 mA + AFE | WFI entre amostras (ainda não implementado) |
| `stop` | Stop mode (regulador LP) + ADS1292R Standby | ~0.35 mA | Estado futuro para baixo duty-cycle |
| `standby` | Standby + ADS1292R Power-down | ~0.003 mA | Estado futuro para longa inatividade |

> A soma da corrente do AFE ao estado MCU assume AVDD=3 V e DVDD=1.8 V.
> O valor do AFE é convertido para mA dividindo `power_uw` por `voltage_v`.

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
com a seguinte interface:

```python
def parse_pwr_transitions(uart_log_text: str) -> list[dict]:
    """Parseia linhas 'PWR <state> <ms>' e retorna transições ordenadas."""
    transitions = []
    for line in uart_log_text.splitlines():
        if line.startswith("PWR "):
            parts = line.split()
            transitions.append({"state": parts[1], "ms": int(parts[2])})
    return transitions

def compute_energy(transitions: list[dict], power_model: dict,
                   total_runtime_ms: int) -> dict:
    """Calcula durações, carga (mAh), energia (mJ), corrente média e autonomia.

    O `power_model` segue o esquema de `../firmware/config/power_model_v1.4.yaml`:
    {
      "voltage_v": 3.3,
      "battery_capacity_mah": 500,
      "mcu": {"states": {"active": {"current_ma": 90.0}, ...}},
      "afe": {"states": {"normal": {"power_uw_per_channel": 335}, ...}}
    }

    Retorno esperado:
    {
      "average_current_ma": float,
      "energy_mj_per_beat": float,
      "estimated_autonomy_hours": float,
      "state_durations_ms": dict[str, int],
      "total_charge_mah": float,
      "total_energy_mj": float
    }

    Algoritmo:
    1. Para cada transição i, duração = transitions[i+1].ms - transitions[i].ms.
    2. A última transição estende até total_runtime_ms.
    3. Para cada estado, multiplica duração pela corrente do power_model.
    4. Soma carga total (mAh) e energia total (mJ).
    5. Deriva average_current_ma e estimated_autonomy_hours.
    """
```

A configuração de consumo é lida de
[`../firmware/config/power_model_v1.4.yaml`](../firmware/config/power_model_v1.4.yaml).

Integrar em [`../firmware/scripts/run_renode_tests.py`](../firmware/scripts/run_renode_tests.py):
- Importar `energy_reporter`.
- Carregar `../firmware/config/power_model_v1.4.yaml`.
- Após `parse_uart_log`, chamar `energy_reporter.compute_energy()`.
- Incluir no `report` JSON (exemplo ilustrativo para 1 batimento/s com sleep ativo):
  ```json
  "energy": {
    "average_current_ma": 61.0,
    "energy_mj_per_beat": 0.202,
    "estimated_autonomy_hours": 8.2,
    "state_durations_ms": {"active": 16, "inference": 16, "sleep": 968},
    "total_charge_mah": 0.0170,
    "total_energy_mj": 0.202
  }
  ```

  > Valores arredondados a partir das correntes da Tabela 3.4, incluindo o
  > consumo do AFE (≈0.1 mA por canal).

## 6. Quality Gate QG19 — Consumo Energético

| Critério | Valor | Como Validar |
| :--- | :--- | :--- |
| Corrente média @ 1 batimento/s | < 50 mA | `report["energy"]["average_current_ma"]` |
| Energia por batimento | < 5 mJ | `report["energy"]["energy_mj_per_beat"]` |
| Autonomia estimada (500 mAh) | > 8 h | `report["energy"]["estimated_autonomy_hours"]` |
| Estados instrumentados | ≥ 2 (active + inference) | presença de logs `PWR` |
