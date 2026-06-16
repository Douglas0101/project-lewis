#ifndef LEWIS_FILTER_H
#define LEWIS_FILTER_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Estado de uma secao biquad (transposed direct-form II).
 */
typedef struct {
    float d0;
    float d1;
} lewis_biquad_state_t;

/**
 * @brief Descritor de uma cascata de biquads.
 *
 * Os coeficientes sao gerados em tempo de compilacao e apontados por @c coeffs.
 * O estado e alocado pela aplicacao (sem allocacao dinamica).
 */
typedef struct {
    const float* coeffs;          /**< [b0, b1, b2, a1, a2] por secao */
    uint32_t num_sections;        /**< Numero de secoes SOS */
    lewis_biquad_state_t* state;  /**< Estado por secao (num_sections elementos) */
} lewis_biquad_cascade_t;

/**
 * @brief Inicializa descritor de cascata.
 *
 * Nao modifica o estado; use lewis_biquad_reset() para zerar os delays.
 */
void lewis_biquad_init(lewis_biquad_cascade_t* cascade,
                       const float* coeffs,
                       uint32_t num_sections,
                       lewis_biquad_state_t* state);

/**
 * @brief Zera o estado (delays) da cascata.
 */
void lewis_biquad_reset(lewis_biquad_cascade_t* cascade);

/**
 * @brief Processa uma unica amostra.
 *
 * @param cascade Descritor da cascata.
 * @param x Amostra de entrada.
 * @return Amostra filtrada.
 */
float lewis_biquad_sample(lewis_biquad_cascade_t* cascade, float x);

/**
 * @brief Processa um bloco de amostras in-place ou out-of-place.
 *
 * input e output podem apontar para o mesmo buffer (processamento in-place).
 */
void lewis_biquad_process_block(lewis_biquad_cascade_t* cascade,
                                const float* input,
                                float* output,
                                size_t len);

/* ---------------------------------------------------------------------------
 * Filter chain (bandpass -> notch)
 * --------------------------------------------------------------------------- */

#define LEWIS_FILTER_MAX_SECTIONS 4u

/**
 * @brief Cadeia de filtros com estado interno.
 *
 * Memoria estatica para ate LEWIS_FILTER_MAX_SECTIONS secoes em cada filtro.
 */
typedef struct {
    lewis_biquad_cascade_t bandpass;
    lewis_biquad_state_t bandpass_state[LEWIS_FILTER_MAX_SECTIONS];
    lewis_biquad_cascade_t notch;
    lewis_biquad_state_t notch_state[LEWIS_FILTER_MAX_SECTIONS];
} lewis_filter_chain_t;

/**
 * @brief Inicializa a cadeia bandpass -> notch com coeficientes gerados.
 */
void lewis_filter_chain_init(lewis_filter_chain_t* chain);

/**
 * @brief Reseta o estado de ambos os filtros (preparo para novo batimento).
 */
void lewis_filter_chain_reset(lewis_filter_chain_t* chain);

/**
 * @brief Processa bloco atraves de bandpass -> notch.
 *
 * input e output podem ser o mesmo buffer.
 */
void lewis_filter_chain_process(lewis_filter_chain_t* chain,
                                const float* input,
                                float* output,
                                size_t len);

#ifdef __cplusplus
}
#endif

#endif /* LEWIS_FILTER_H */
