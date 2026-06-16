# Limites de Simulação do Project-Lewis v1.2

Este documento descreve os limites e premissas das simulações executadas no Project-Lewis v1.2. Toda a validação é realizada **puramente em software** com o emulador Renode 1.15.3, sem dependência de hardware físico.

---

## Semihosting

- O Renode 1.15.3 ativa o semihosting implicitamente quando encontra a instrução `BKPT 0xAB`.
- Não há uma chave explícita `semihosting-enabled` no REPL do Cortex-M utilizado nas plataformas do projeto.
- **Implicação prática:** o ambiente de CI precisa garantir a versão exata do Renode (1.15.3) para que o comportamento do semihosting permaneça determinístico. O pipeline de hard-gates inclui verificação de versão (`make verify-renode`).

---

## Latência

- A latência medida nas simulações é de aproximadamente **16.18 ms/beat**.
- Esse valor é **determinístico no Renode**, pois o emulador executa a arquitetura Cortex-M de forma funcional.
- **Implicação prática:** o Renode não modela cache misses, pipeline hazards, contenção de barramento nem jitter. Portanto, o valor de 16.18 ms/beat deve ser tratado como um **upper bound teórico**; a latência real em um MCU STM32F4 pode ser maior ou variável dependendo do estado do sistema.

---

## Fidelidade Numérica

- A build ARM utiliza a biblioteca CMSIS-NN otimizada, enquanto a execução nativa utiliza as implementações de referência do TensorFlow Lite for Microcontrollers.
- Divergências entre essas implementações são esperadas devido a arredondamento otimizado em acumuladores de 32 bits.
- O Quality Gate QG8 (`tests/test_tflm_bitexact.py`) utiliza `np.allclose(arm_output, ref_output, atol=1)` para tolerar diferenças de até **1 LSB**.
- **Implicação prática:** a bit-exatidão estrita entre a simulação Renode/CMSIS-NN e a execução nativa de referência **não é garantida**. Aplicações que dependam de saídas idênticas bit a bit devem incluir tolerância apropriada.

---

## Timer e Watchdog

- O firmware utiliza **TIM2** como fonte de tick de 1 ms para `delay_ms()`, `millis()` e para o watchdog software.
- A ISR do TIM2 coloca a CPU em espera de baixo consumo (`__WFI`) e a acorda a cada milissegundo, eliminando o busy-wait anterior.
- Se o ambiente de simulação não modelar TIM2 corretamente, o HAL detecta a falha na inicialização e recai para o delay baseado em SysTick (busy-wait sem `__WFI`). Nesse modo fallback o watchdog não está disponível.
- O watchdog é **software** (baseado em contador decrementado na ISR do TIM2) porque o IWDG do STM32 não é modelado de forma confiável no Renode 1.15.3.
- Ao expirar, o watchdog loga `WATCHDOG_TIMEOUT` via UART e executa `NVIC_SystemReset()` no alvo/emulador ARM. Em host nativo, o equivalente usa `SIGALRM` e encerra com `lewis_hal_shutdown()`.
- No Renode, o reset via `NVIC_SystemReset()` é observável como reinício da máquina; caso o modelo não o trate perfeitamente, o log `WATCHDOG_TIMEOUT` ainda é emitido e o firmware não entra em hard fault.

## Referências

- `Makefile`: target `verify-renode` para pinagem de versão.
- `tests/test_tflm_bitexact.py`: tolerância de 1 LSB no QG8.
- `tests/test_watchdog.py`: Quality Gate QG13 para validação do watchdog.
- Documentação do Renode: semihosting via `BKPT 0xAB`.
