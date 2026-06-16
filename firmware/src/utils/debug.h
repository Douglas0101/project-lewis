#ifndef LEWIS_DEBUG_H
#define LEWIS_DEBUG_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Inicializa canal de debug (UART no emulador/alvo).
 */
void lewis_debug_init(void);

/**
 * @brief Envia string terminada em '\0' pelo canal de debug.
 */
void lewis_debug_print(const char* msg);

/**
 * @brief Envia um caractere.
 */
void lewis_debug_putc(char c);

/**
 * @brief Envia inteiro com sinal em decimal.
 */
void lewis_debug_print_int(int32_t value);

/**
 * @brief Envia inteiro sem sinal em decimal.
 */
void lewis_debug_print_uint(uint32_t value);

/**
 * @brief Envia inteiro sem sinal em hexadecimal.
 */
void lewis_debug_print_hex(uint32_t value);

/**
 * @brief Envia buffer binario em hexadecimal.
 */
void lewis_debug_hex(const uint8_t* data, size_t len);

#ifdef __cplusplus
}
#endif

#endif /* LEWIS_DEBUG_H */
