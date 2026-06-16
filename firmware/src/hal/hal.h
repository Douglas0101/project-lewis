#ifndef LEWIS_HAL_H
#define LEWIS_HAL_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* --------------------------------------------------------------------------
 * Abstração de Hardware (HAL) para Project-Lewis.
 *
 * Implementações:
 *   - hal/simulator/  -> Renode / host nativo
 *   - hal/target/     -> STM32F4 real (quando houver hardware)
 * -------------------------------------------------------------------------- */

/**
 * @brief Inicializa periféricos mínimos: clock, UART, SysTick.
 */
void lewis_hal_init(void);

/**
 * @brief Retorna número de milissegundos desde o boot (wrap-around 32 bits).
 */
uint32_t lewis_hal_millis(void);

/**
 * @brief Aguarda por ms milissegundos (bloqueante).
 *
 * No alvo/emulador ARM utiliza timer de proposito geral (TIM2) com
 * interrupcao, colocando a CPU em espera de baixo consumo (__WFI) ate o ISR
 * acordar. No host nativo usa usleep.
 */
void lewis_hal_delay_ms(uint32_t ms);

/**
 * @brief Inicia watchdog software com timeout em milissegundos.
 *
 * Se o watchdog nao for parado antes do timeout, o sistema loga
 * WATCHDOG_TIMEOUT e reinicia (alvo) ou encerra (host nativo).
 */
void lewis_hal_watchdog_start(uint32_t timeout_ms);

/**
 * @brief Para o watchdog software.
 */
void lewis_hal_watchdog_stop(void);

/**
 * @brief Retorna true se o watchdog software expirou.
 */
bool lewis_hal_watchdog_expired(void);

/**
 * @brief Timeout padrao para protecao da inferencia TFLM.
 */
#define LEWIS_WATCHDOG_TIMEOUT_MS 1000U

/**
 * @brief Envia um byte pela UART de debug.
 */
void lewis_hal_uart_tx(uint8_t byte);

/**
 * @brief Retorna true se há byte disponível na UART de debug.
 */
bool lewis_hal_uart_rx_ready(void);

/**
 * @brief Lê um byte da UART de debug. Chamador deve verificar rx_ready().
 */
uint8_t lewis_hal_uart_rx(void);

/**
 * @brief Entra em estado de erro fatal.
 *
 * Comportamento padrao: loop infinito, mantendo o sistema parado.
 * Em builds de debug o implementador pode inserir __BKPT(0) para facilitar
 * a captura pelo depurador, mas a funcao nao retorna.
 */
void lewis_hal_panic(void) __attribute__((noreturn));

/**
 * @brief Encerra a execucao de forma controlada.
 *
 * No host nativo chama exit(0); no alvo/emulador ARM usa semihosting
 * SYS_EXIT. Nunca retorna.
 */
void lewis_hal_shutdown(void) __attribute__((noreturn));

/**
 * @brief Retorna timestamp de ciclos (SysTick no ARM, ns no host nativo).
 */
uint32_t lewis_hal_benchmark_start(void);

/**
 * @brief Retorna timestamp de ciclos atual.
 */
uint32_t lewis_hal_benchmark_stop(void);

/**
 * @brief Converte delta de ciclos (ARM) ou ns (host nativo) para microssegundos.
 */
uint32_t lewis_hal_benchmark_delta_us(uint32_t delta);

#ifdef __cplusplus
}
#endif

#endif /* LEWIS_HAL_H */
