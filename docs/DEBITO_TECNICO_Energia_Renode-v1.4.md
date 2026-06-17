# Débito Técnico: Modelagem de Energia no Renode

**Versão:** 1.4 | **Data:** 2026-06-17 | **Responsável:** Firmware / QA

## 1. Contexto

A Fase 1 do Project-Lewis entrega um firmware de classificação de arritmias
executável no emulador Renode (STM32F4Discovery) com métricas de latência,
uso de RAM/Flash e bit-exatidão (QG7–QG18). No entanto, conforme registrado
em [`Camada-08-Firmware-v1.1.md`](Camada-08-Firmware-v1.1.md) (seção 8.10, item 6),
[`Camada-09-Simulacao-v1.1.md`](Camada-09-Simulacao-v1.1.md) (seção 9.10, item 6)
e [`SIMULATION_LIMITS.md`](SIMULATION_LIMITS.md) (seção 1), **não existe estimativa de consumo
de energia** no relatório de simulação.

## 2. Por que isso é um débito técnico

- O dispositivo final é portátil e alimentado por bateria; consumo é um
  requisito crítico de design.
- Decisões de otimização (clock, uso de WFI, desligamento de periféricos)
  não podem ser quantificadas sem um modelo energético.
- A comparação entre CMSIS-NN vs kernels de referência, e entre arenas de
  48 KB vs 64 KB, deve levar em conta energia além de latência.
- A v1.3 valida corretude funcional; a v1.4 deve validar viabilidade
  energética antes de hardware físico.

## 3. Gap Técnico Atual

| Aspecto | Estado na v1.3 | Necessário na v1.4 |
| :--- | :--- | :--- |
| Modelo de CPU | Cortex-M4 @ 168 MHz emulado, sem estado de energia | Estados Run / Sleep / Stop / Standby com correntes por datasheet |
| Periféricos | UART4, TIM2, SysTick habilitados; sem contabilização | Contabilizar corrente de UART, TIM2, ADC (ADS1292R) e Flash |
| AFE | Substituído por stub; ADS1292R não modelado | Modelar consumo do ADS1292R @ 500 SPS (335 µW/canal típico) |
| Instrumentação | Logs de latência por batimento | Logs de estados de energia + timestamp virtual |
| Relatório | `firmware_simulation_report.json` sem energia | Campo `energy_mj_per_beat`, `estimated_autonomy_hours` |
| Quality Gate | Nenhum | QG19: < 50 mA médios e < 165 mJ/batimento @ 3.3 V para 1 batimento/s |

## 4. Abordagens Consideradas

### 4.1 Modelagem externa via runner Python (Recomendada)
Manter o Renode inalterado (ele não emula consumo nativamente para STM32F4)
e calcular energia fora da simulação, combinando:
- tempo virtual por estado obtido do log UART;
- corrente por estado obtida de `firmware/config/power_model_v1.4.yaml`;
- tensão nominal (3.3 V) e carga da bateria.

**Prós:** não depende de features não-existentes no Renode 1.15.3;
reprodutível; fácil de calibrar com hardware real depois.
**Contras:** é uma estimativa, não uma medição; precisa de instrumentação
cuidadosa no firmware.

### 4.2 Plugin/EXT de energia no Renode
Criar um plugin C# ou usar `ExternalControlClient` para interceptar execução
e inferir estados pela PC.

**Prós:** integrado ao emulador.
**Contras:** aumenta complexidade de build/CI; exige manutenção de código
C#; ganho marginal vs abordagem externa.

### 4.3 Estimativa estática por datasheet
Calcular energia apenas a partir do tempo de execução total e corrente média
do datasheet, sem estados.

**Prós:** trivial.
**Contras:** ignora WFI, duty-cycle de inferência e consumo do AFE;
inútil para otimizações.

## 5. Recomendação
Adotar a **Abordagem 4.1**: modelagem externa via runner Python com
instrumentação mínima no firmware.

## 6. Critérios de Aceite para Resolver o Débito

- [ ] Documento de especificação [`Camada-09-Energia-v1.4.md`](Camada-09-Energia-v1.4.md) aprovado.
- [ ] Modelo de consumo YAML versionado em [`firmware/config/power_model_v1.4.yaml`](../firmware/config/power_model_v1.4.yaml).
- [ ] Firmware emite logs de transição de estado de energia no formato:
      `PWR <state> <ms>` (ex: `PWR inference 16`, `PWR active 984`).
- [ ] Runner Renode gera campo `energy` no relatório JSON, contendo
      `energy_mj_per_beat` e `estimated_autonomy_hours`.
- [ ] QG19 falha se `average_current_ma > 50 mA` para 1 batimento/s.
- [ ] Limites documentados: valores são estimativas (@25 °C, VDD=3.3 V) e
      devem ser validados em silício real.

## 7. Riscos e Mitigações

| Risco | Mitigação |
| :--- | :--- |
| Renode não modela periféricos reais de energia | Usar modelo baseado em datasheet + calibração futura |
| Logs de estado aumentam tráfego UART | Usar prefixo curto (`PWR`) e apenas transições |
| Datasheet não reflete condições reais | Documentar margem de erro e sensibilidade |
| Atraso na v1.4 se modelo ficar complexo | MVP com 3 estados (active/sleep/inference) |
