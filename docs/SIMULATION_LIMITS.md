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

## Referências

- `Makefile`: target `verify-renode` para pinagem de versão.
- `tests/test_tflm_bitexact.py`: tolerância de 1 LSB no QG8.
- Documentação do Renode: semihosting via `BKPT 0xAB`.
