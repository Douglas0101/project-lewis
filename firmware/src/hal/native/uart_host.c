/* Host-native UART stub: redireciona para stdout. */

#include "hal/hal.h"

#include <stdio.h>

void lewis_hal_uart_tx(uint8_t byte)
{
    (void)putchar((int)byte);
}

bool lewis_hal_uart_rx_ready(void)
{
    return false;
}

uint8_t lewis_hal_uart_rx(void)
{
    return 0;
}
