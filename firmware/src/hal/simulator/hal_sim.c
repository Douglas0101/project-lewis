/*
 * HAL para ambiente de simulacao (Renode + host nativo).
 *
 * No Renode/STM32F4 usa SysTick como fonte de tempo monotonico e para
 * benchmark de ciclos (DWT_CYCCNT nao e emulado pelo Renode na plataforma
 * padrao). No host nativo usa stdio e clock_gettime.
 */

#include "hal/hal.h"

#ifdef LEWIS_HOST_NATIVE
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#else
#include <stdio.h>
#include <time.h>
#endif

/* Registradores SysTick do Cortex-M. */
#define SYST_CSR   (*(volatile uint32_t*)0xE000E010UL)
#define SYST_RVR   (*(volatile uint32_t*)0xE000E014UL)
#define SYST_CVR   (*(volatile uint32_t*)0xE000E018UL)
#define SYST_CALIB (*(volatile uint32_t*)0xE000E01CUL)

#define SYST_CSR_ENABLE     (1U << 0)
#define SYST_CSR_CLKSOURCE  (1U << 2)
#define SYST_CSR_COUNTFLAG  (1U << 16)

#ifndef LEWIS_SYSTICK_HZ
#define LEWIS_SYSTICK_HZ 168000000UL
#endif
#define LEWIS_CYCLES_PER_MS (LEWIS_SYSTICK_HZ / 1000UL)
#define SYST_RELOAD_MASK    0x00FFFFFFUL
#define SYST_RELOAD_FULL    (SYST_RELOAD_MASK + 1UL) /* 2^24 ciclos */

#ifndef LEWIS_HOST_NATIVE
static uint32_t s_systick_wraps = 0;
#endif

void lewis_hal_init(void)
{
#ifdef LEWIS_HOST_NATIVE
    /* Desabilita buffering para que logs aparecam imediatamente no terminal. */
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);
#else
    /*
     * Configura SysTick como contador livre decrescente @ LEWIS_SYSTICK_HZ.
     * Usamos RVR = 24-bit maximo. Se ja estiver habilitado (warm reset ou
     * reentrancia), nao o reconfiguramos para preservar a base de tempo.
     */
    if ((SYST_CSR & SYST_CSR_ENABLE) == 0) {
        SYST_RVR = SYST_RELOAD_MASK;
        SYST_CVR = 0UL;
        SYST_CSR = SYST_CSR_ENABLE | SYST_CSR_CLKSOURCE;
    }
#endif
}

uint32_t lewis_hal_millis(void)
{
#ifdef LEWIS_HOST_NATIVE
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint32_t)((ts.tv_sec * 1000ULL) + (ts.tv_nsec / 1000000ULL));
#else
    /*
     * Conta wraps do SysTick (COUNTFLAG) e converte ciclos totais em ms.
     * Cada wrap representa 2^24 ciclos. A leitura de SYST_CSR limpa
     * COUNTFLAG, portanto cada leitura captura no maximo um wrap. Como o
     * periodo do SysTick e ~100 ms, isso e seguro para os usos esperados.
     *
     * Le CVR antes de CSR para minimizar a janela entre a leitura do valor
     * e a deteccao do underflow.
     */
    const uint32_t val = SYST_CVR & SYST_RELOAD_MASK;
    if (SYST_CSR & SYST_CSR_COUNTFLAG) {
        s_systick_wraps++;
    }
    const uint64_t total_cycles =
        ((uint64_t)s_systick_wraps * SYST_RELOAD_FULL) +
        (SYST_RELOAD_MASK - val);
    return (uint32_t)(total_cycles / LEWIS_CYCLES_PER_MS);
#endif
}

void lewis_hal_delay_ms(uint32_t ms)
{
#ifdef LEWIS_HOST_NATIVE
    usleep(ms * 1000U);
#else
    /*
     * Busy-wait baseado no contador monotonico de SysTick. Cada iteracao
     * le o SysTick, o que tambem atualiza s_systick_wraps via COUNTFLAG,
     * mantendo millis() coerente durante o delay.
     */
    const uint32_t start = lewis_hal_millis();
    while ((lewis_hal_millis() - start) < ms) {
        /* aguarda */
    }
#endif
}

/*
 * lewis_hal_uart_tx e implementado separadamente:
 *   - hal/target/uart_stm32f4.c  (build ARM)
 *   - hal/native/uart_host.c     (build host nativo)
 */

void lewis_hal_panic(void)
{
    printf("[PANIC]\n");
    while (1) {
        /* loop infinito; em debug pode colocar BKPT */
    }
}

void lewis_hal_shutdown(void)
{
#ifdef LEWIS_HOST_NATIVE
    exit(0);
#else
    register uint32_t r0 __asm__("r0") = 0x18U;
    register uint32_t r1 __asm__("r1") = 0x20026U;
    __asm__ volatile("bkpt 0xAB" : : "r"(r0), "r"(r1));
    while (1) {
    }
#endif
}

uint32_t lewis_hal_benchmark_start(void)
{
#ifdef LEWIS_HOST_NATIVE
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint32_t)(ts.tv_sec * 1000000000ULL + ts.tv_nsec);
#else
    /* SysTick VAL e decrescente; invertemos no calculo do delta. */
    return SYST_CVR & SYST_RELOAD_MASK;
#endif
}

uint32_t lewis_hal_benchmark_stop(void)
{
    return lewis_hal_benchmark_start();
}

uint32_t lewis_hal_benchmark_delta_us(uint32_t delta)
{
#ifdef LEWIS_HOST_NATIVE
    /* No host nativo o delta e em nanossegundos. */
    return delta / 1000U;
#else
    /* SysTick conta ciclos de processador @ LEWIS_SYSTICK_HZ.
     * delta ja e a diferenca absoluta (start - stop) & 0xFFFFFF. */
    return delta / (LEWIS_SYSTICK_HZ / 1000000UL);
#endif
}
