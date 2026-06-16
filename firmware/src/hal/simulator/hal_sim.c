/*
 * HAL para ambiente de simulacao (Renode + host nativo).
 *
 * No Renode/STM32F4 usa TIM2 como fonte de tick de 1 ms para delays,
 * millis() e watchdog software. SysTick continua sendo usado apenas para
 * benchmark de ciclos (DWT_CYCCNT nao e emulado pelo Renode na plataforma
 * padrao). No host nativo usa stdio, clock_gettime e SIGALRM para o watchdog.
 */

#include "hal/hal.h"
#include "utils/debug.h"

#ifdef LEWIS_HOST_NATIVE
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <signal.h>
#include <sys/time.h>
#else
#include <stdio.h>
#include <time.h>
#include <stdint.h>
#endif

#ifdef LEWIS_HOST_NATIVE

/* --------------------------------------------------------------------------
 * Host nativo
 * -------------------------------------------------------------------------- */

static volatile uint32_t s_watchdog_counter = 0;
static volatile bool s_watchdog_active = false;
static volatile bool s_watchdog_expired = false;
static volatile bool s_watchdog_timer_initialized = false;

static void watchdog_sigalrm_handler(int sig)
{
    (void)sig;
    if (!s_watchdog_active || s_watchdog_expired) {
        return;
    }
    if (s_watchdog_counter > 0) {
        s_watchdog_counter--;
    }
    if (s_watchdog_counter == 0) {
        s_watchdog_expired = true;
        s_watchdog_active = false;
        lewis_debug_print("WATCHDOG_TIMEOUT\n");
        lewis_hal_shutdown();
    }
}

static void watchdog_timer_init(void)
{
    if (s_watchdog_timer_initialized) {
        return;
    }
    struct sigaction sa;
    sa.sa_handler = watchdog_sigalrm_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_RESTART;
    sigaction(SIGALRM, &sa, NULL);

    struct itimerval itv;
    itv.it_interval.tv_sec = 0;
    itv.it_interval.tv_usec = 1000; /* tick de 1 ms */
    itv.it_value.tv_sec = 0;
    itv.it_value.tv_usec = 1000;
    setitimer(ITIMER_REAL, &itv, NULL);
    s_watchdog_timer_initialized = true;
}

void lewis_hal_init(void)
{
    /* Desabilita buffering para que logs aparecam imediatamente no terminal. */
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);
}

uint32_t lewis_hal_millis(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint32_t)((ts.tv_sec * 1000ULL) + (ts.tv_nsec / 1000000ULL));
}

void lewis_hal_delay_ms(uint32_t ms)
{
    usleep(ms * 1000U);
}

void lewis_hal_watchdog_start(uint32_t timeout_ms)
{
    watchdog_timer_init();
    s_watchdog_counter = timeout_ms;
    s_watchdog_expired = false;
    s_watchdog_active = true;
}

void lewis_hal_watchdog_stop(void)
{
    s_watchdog_active = false;
}

bool lewis_hal_watchdog_expired(void)
{
    return s_watchdog_expired;
}

#else /* LEWIS_HOST_NATIVE */

/* --------------------------------------------------------------------------
 * Alvo/emulador ARM (STM32F4)
 * -------------------------------------------------------------------------- */

/* Registradores SysTick do Cortex-M. */
#define SYST_CSR   (*(volatile uint32_t*)0xE000E010UL)
#define SYST_RVR   (*(volatile uint32_t*)0xE000E014UL)
#define SYST_CVR   (*(volatile uint32_t*)0xE000E018UL)
#define SYST_CALIB (*(volatile uint32_t*)0xE000E01CUL)

#define SYST_CSR_ENABLE     (1U << 0)
#define SYST_CSR_CLKSOURCE  (1U << 2)
#define SYST_CSR_COUNTFLAG  (1U << 16)

#define SYST_RELOAD_MASK    0x00FFFFFFUL
#define SYST_RELOAD_FULL    (SYST_RELOAD_MASK + 1UL) /* 2^24 ciclos */

/* TIM2: timer 32 bits no APB1. */
#define TIM2_BASE           0x40000000U
#define TIM2_CR1            (*(volatile uint32_t*)(TIM2_BASE + 0x00U))
#define TIM2_DIER           (*(volatile uint32_t*)(TIM2_BASE + 0x0CU))
#define TIM2_SR             (*(volatile uint32_t*)(TIM2_BASE + 0x10U))
#define TIM2_CNT            (*(volatile uint32_t*)(TIM2_BASE + 0x24U))
#define TIM2_PSC            (*(volatile uint32_t*)(TIM2_BASE + 0x28U))
#define TIM2_ARR            (*(volatile uint32_t*)(TIM2_BASE + 0x2CU))

#define TIM_CR1_CEN         (1U << 0)
#define TIM_DIER_UIE        (1U << 0)
#define TIM_SR_UIF          (1U << 0)

/* RCC: habilitacao de clock do TIM2 (APB1 bit 0). */
#define RCC_BASE            0x40023800U
#define RCC_APB1ENR         (*(volatile uint32_t*)(RCC_BASE + 0x40U))
#define RCC_APB1ENR_TIM2EN  (1U << 0)

/* NVIC: habilita interrupcao TIM2 (IRQ 28). */
#define NVIC_ISER0          (*(volatile uint32_t*)0xE000E100UL)
#define TIM2_IRQ_BIT        (1U << 28)

/* Registrador de reset do sistema (AIRCR). */
#define SCB_AIRCR           (*(volatile uint32_t*)0xE000ED0CUL)
#define AIRCR_VECTKEY       (0x05FAU << 16)
#define AIRCR_SYSRESETREQ   (1U << 2)

#ifndef LEWIS_SYSTICK_HZ
#define LEWIS_SYSTICK_HZ 168000000UL
#endif
#define LEWIS_CYCLES_PER_MS (LEWIS_SYSTICK_HZ / 1000UL)

/* TIM2CLK = 2 * APB1. Com SYSCLK=168 MHz e APB1=42 MHz, TIM2CLK=84 MHz. */
#ifndef LEWIS_TIM2CLK_HZ
#define LEWIS_TIM2CLK_HZ 84000000UL
#endif
#define TIM2_TICKS_PER_MS   (LEWIS_TIM2CLK_HZ / 1000UL)

static volatile uint32_t s_millis = 0;
static volatile uint32_t s_systick_wraps = 0;
static volatile uint32_t s_watchdog_counter = 0;
static volatile bool s_watchdog_active = false;
static volatile bool s_watchdog_expired = false;
static volatile bool s_tim2_initialized = false;

/* Trava de emergencia: se TIM2 nao contar (emulador incompleto), volta para
 * o delay baseado em SysTick sem WFI, garantindo que o sistema nao trave. */
static volatile bool s_tim2_working = false;

void TIM2_IRQHandler(void)
{
    if (TIM2_SR & TIM_SR_UIF) {
        TIM2_SR &= ~TIM_SR_UIF;
        s_millis++;

        if (s_watchdog_active && s_watchdog_counter > 0) {
            s_watchdog_counter--;
            if (s_watchdog_counter == 0) {
                s_watchdog_active = false;
                s_watchdog_expired = true;
                /* Log minimo via UART (polling). O watchdog e caminho de
                 * falha, portanto a seguranca do sistema prevalece sobre a
                 * reentrancia do driver UART. */
                lewis_debug_print("WATCHDOG_TIMEOUT\n");
                /* Reinicia o sistema. Em Renode isso e observavel como
                 * reset/reinicio da maquina; se nao for modelado, o log ja
                 * foi emitido. */
                SCB_AIRCR = AIRCR_VECTKEY | AIRCR_SYSRESETREQ;
                while (1) {
                    __asm__ volatile("nop");
                }
            }
        }
    }
}

static void tim2_init(void)
{
    if (s_tim2_initialized) {
        return;
    }

    /* Habilita clock do TIM2. */
    RCC_APB1ENR |= RCC_APB1ENR_TIM2EN;

    /* Para o timer antes de reconfigurar. */
    TIM2_CR1 &= ~TIM_CR1_CEN;
    TIM2_DIER = 0;
    TIM2_SR = 0;

    /* Periodo de 1 ms: PSC divide o clock, ARR conta os ticks restantes.
     * PSC deve caber em 16 bits (0..65535). */
    uint32_t psc = (TIM2_TICKS_PER_MS / 1000U) - 1U;
    uint32_t arr = 999U;
    if (psc > 65535U) {
        psc = 65535U;
        arr = (TIM2_TICKS_PER_MS / (psc + 1U)) - 1U;
    }
    TIM2_PSC = psc;
    TIM2_ARR = arr;
    TIM2_CNT = 0;

    /* Limpa pending e habilita interrupcao de update. */
    TIM2_SR &= ~TIM_SR_UIF;
    TIM2_DIER |= TIM_DIER_UIE;

    /* Habilita interrupcao no NVIC. */
    NVIC_ISER0 |= TIM2_IRQ_BIT;

    /* Inicia o timer. */
    TIM2_CR1 |= TIM_CR1_CEN;

    s_tim2_initialized = true;

    /* Verifica se o timer esta contando (protecao contra emuladores que nao
     * modelam TIM2). Executa um numero fixo de iteracoes lendo CNT; se o
     * valor mudar, consideramos TIM2 funcional. */
    uint32_t cnt0 = TIM2_CNT;
    for (volatile uint32_t i = 0; i < 1000000U; i++) {
        if (TIM2_CNT != cnt0) {
            s_tim2_working = true;
            break;
        }
    }
}

void lewis_hal_init(void)
{
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

    tim2_init();
}

uint32_t lewis_hal_millis(void)
{
    /* Se TIM2 esta funcionando, retorna o contador atualizado pela ISR.
     * Caso contrario, recai em contagem por SysTick (menos precisa, mas
     * funciona em emuladores limitados). */
    if (s_tim2_working) {
        return s_millis;
    }

    /* Fallback: conta wraps do SysTick (COUNTFLAG) e converte ciclos totais
     * em ms. */
    const uint32_t val = SYST_CVR & SYST_RELOAD_MASK;
    if (SYST_CSR & SYST_CSR_COUNTFLAG) {
        s_systick_wraps++;
    }
    const uint64_t total_cycles =
        ((uint64_t)s_systick_wraps * SYST_RELOAD_FULL) +
        (SYST_RELOAD_MASK - val);
    return (uint32_t)(total_cycles / LEWIS_CYCLES_PER_MS);
}

void lewis_hal_delay_ms(uint32_t ms)
{
    if (ms == 0) {
        return;
    }

    if (s_tim2_working) {
        uint32_t start = s_millis;
        while ((s_millis - start) < ms) {
            /* Polling com memory barrier. Em simulacao Renode o WFI pode
             * deixar a CPU adormecida se a entrega da interrupcao for
             * interrompida; polling garante progresso enquanto TIM2 conta. */
            __asm__ volatile("dsb" ::: "memory");
        }
    } else {
        /* Fallback para emuladores sem TIM2: busy-wait baseado em SysTick.
         * Nao usa WFI porque nao ha interrupcao habilitada para acordar. */
        uint32_t start = lewis_hal_millis();
        while ((lewis_hal_millis() - start) < ms) {
            /* aguarda */
        }
    }
}

void lewis_hal_watchdog_start(uint32_t timeout_ms)
{
    if (timeout_ms == 0) {
        s_watchdog_active = false;
        return;
    }
    s_watchdog_counter = timeout_ms;
    s_watchdog_expired = false;
    s_watchdog_active = true;
}

void lewis_hal_watchdog_stop(void)
{
    s_watchdog_active = false;
}

bool lewis_hal_watchdog_expired(void)
{
    return s_watchdog_expired;
}

#endif /* LEWIS_HOST_NATIVE */

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
