#include "utils/debug.h"
#include "hal/hal.h"

#include <stdint.h>
#include <stddef.h>

void lewis_debug_init(void)
{
    lewis_hal_init();
}

void lewis_debug_putc(char c)
{
    lewis_hal_uart_tx((uint8_t)c);
}

void lewis_debug_print(const char* msg)
{
    if (!msg) {
        return;
    }
    while (*msg) {
        lewis_hal_uart_tx((uint8_t)*msg);
        ++msg;
    }
}

void lewis_debug_print_uint(uint32_t value)
{
    char buf[11];
    int i = 0;
    if (value == 0) {
        lewis_hal_uart_tx('0');
        return;
    }
    while (value > 0 && i < 10) {
        buf[i++] = (char)('0' + (value % 10));
        value /= 10;
    }
    while (i > 0) {
        lewis_hal_uart_tx((uint8_t)buf[--i]);
    }
}

void lewis_debug_print_int(int32_t value)
{
    if (value < 0) {
        lewis_hal_uart_tx('-');
        value = -value;
    }
    lewis_debug_print_uint((uint32_t)value);
}

void lewis_debug_print_hex(uint32_t value)
{
    const char* digits = "0123456789abcdef";
    lewis_debug_print("0x");
    for (int i = 28; i >= 0; i -= 4) {
        lewis_hal_uart_tx((uint8_t)digits[(value >> i) & 0xF]);
    }
}

void lewis_debug_hex(const uint8_t* data, size_t len)
{
    const char* digits = "0123456789abcdef";
    for (size_t i = 0; i < len; ++i) {
        lewis_hal_uart_tx((uint8_t)digits[data[i] >> 4]);
        lewis_hal_uart_tx((uint8_t)digits[data[i] & 0x0F]);
        lewis_hal_uart_tx(' ');
    }
}
