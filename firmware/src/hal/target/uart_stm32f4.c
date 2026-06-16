/*
 * Driver UART4 minimalista para STM32F407VG.
 *
 * A STM32F4Discovery conecta UART4 ao ST-LINK VCP, portanto usamos UART4
 * para que o Renode capture a saida no console/analyzer.
 *
 * Usa PC10 (TX) e PC11 (RX) com alternate function AF8.
 * Baud rate padrao: 115200 com PCLK1 = 4 MHz (reset default).
 *
 * Esta implementacao e propositalmente simples: sem interrupcoes, sem DMA,
 * apenas polling de TX para logs de simulacao no Renode.
 */

#include "hal/hal.h"

#include <stdint.h>

/* Registradores do RCC */
#define RCC_BASE        0x40023800U
#define RCC_AHB1ENR     (*(volatile uint32_t*)(RCC_BASE + 0x30U))
#define RCC_APB1ENR     (*(volatile uint32_t*)(RCC_BASE + 0x40U))

/* Registradores do GPIOC */
#define GPIOC_BASE      0x40020800U
#define GPIOC_MODER     (*(volatile uint32_t*)(GPIOC_BASE + 0x00U))
#define GPIOC_OTYPER    (*(volatile uint32_t*)(GPIOC_BASE + 0x04U))
#define GPIOC_OSPEEDR   (*(volatile uint32_t*)(GPIOC_BASE + 0x08U))
#define GPIOC_PUPDR     (*(volatile uint32_t*)(GPIOC_BASE + 0x0CU))
#define GPIOC_AFR1      (*(volatile uint32_t*)(GPIOC_BASE + 0x24U))

/* Registradores do UART4 */
#define UART4_BASE      0x40004C00U
#define UART4_SR        (*(volatile uint32_t*)(UART4_BASE + 0x00U))
#define UART4_DR        (*(volatile uint32_t*)(UART4_BASE + 0x04U))
#define UART4_BRR       (*(volatile uint32_t*)(UART4_BASE + 0x08U))
#define UART4_CR1       (*(volatile uint32_t*)(UART4_BASE + 0x0CU))

#define USART_SR_TXE    (1U << 7)
#define USART_SR_RXNE   (1U << 5)

#define USART_CR1_UE    (1U << 13)
#define USART_CR1_TE    (1U << 3)
#define USART_CR1_RE    (1U << 2)

static uint8_t s_uart_initialized = 0;

static void uart4_init(void)
{
    if (s_uart_initialized) {
        return;
    }

    /* Habilita clocks: GPIOC (AHB1) e UART4 (APB1). */
    RCC_AHB1ENR |= (1U << 2);
    RCC_APB1ENR |= (1U << 19);

    /* PC10 TX e PC11 RX em alternate function. */
    GPIOC_MODER &= ~((3U << (2 * 10)) | (3U << (2 * 11)));
    GPIOC_MODER |= ((2U << (2 * 10)) | (2U << (2 * 11)));

    /* AF8 para PC10 e PC11. */
    GPIOC_AFR1 &= ~((0xFU << (4 * 2)) | (0xFU << (4 * 3)));
    GPIOC_AFR1 |= ((8U << (4 * 2)) | (8U << (4 * 3)));

    /* Push-pull, high speed, pull-up. */
    GPIOC_OTYPER &= ~((1U << 10) | (1U << 11));
    GPIOC_OSPEEDR |= ((3U << (2 * 10)) | (3U << (2 * 11)));
    GPIOC_PUPDR &= ~((3U << (2 * 10)) | (3U << (2 * 11)));
    GPIOC_PUPDR |= ((1U << (2 * 10)) | (1U << (2 * 11)));

    /* Baud rate 115200 @ 4 MHz PCLK1: BRR = 4_000_000 / 115200 ~= 34.72.
     * Mantissa = 34, fractional = round(0.72 * 16) = 12 -> BRR = (34 << 4) | 12 = 0x22C.
     */
    UART4_BRR = 0x22C;

    /* Habilita TX e RX. */
    UART4_CR1 = USART_CR1_TE | USART_CR1_RE;

    /* Habilita UART. */
    UART4_CR1 |= USART_CR1_UE;

    s_uart_initialized = 1;
}

void lewis_hal_uart_tx(uint8_t byte)
{
    uart4_init();
    while (!(UART4_SR & USART_SR_TXE)) {
        /* espera buffer de transmissao vazio */
    }
    UART4_DR = byte;
}

bool lewis_hal_uart_rx_ready(void)
{
    uart4_init();
    return (UART4_SR & USART_SR_RXNE) != 0;
}

uint8_t lewis_hal_uart_rx(void)
{
    return (uint8_t)(UART4_DR & 0xFFU);
}
