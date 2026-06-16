#ifndef LEWIS_ADC_STUB_H
#define LEWIS_ADC_STUB_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Retorna um batimento de teste de 500 amostras @ 500 Hz.
 *
 * O sinal é gerado deterministicamente a partir de um seed interno.
 *
 * @param idx Indice do batimento (0, 1, ...). Cada indice gera um sinal diferente.
 * @param out_buffer Buffer de saida com 500 amostras int8.
 * @return Numero de amostras escritas (sempre 500 em caso de sucesso).
 */
size_t lewis_adc_stub_get_beat(uint32_t idx, int8_t out_buffer[500]);

/**
 * @brief Retorna um sinal senoidal puro para testes estruturais.
 */
size_t lewis_adc_stub_get_sine(float amplitude_mv, int8_t out_buffer[500]);

#ifdef __cplusplus
}
#endif

#endif /* LEWIS_ADC_STUB_H */
