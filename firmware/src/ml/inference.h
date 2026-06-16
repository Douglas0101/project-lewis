#ifndef LEWIS_INFERENCE_H
#define LEWIS_INFERENCE_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Dimensões do modelo.
 */
#define LEWIS_INPUT_LEN   500
#define LEWIS_OUTPUT_LEN  5

/**
 * @brief Inicializa o interpretador TFLM e carrega o modelo.
 *
 * @return true se sucesso, false em caso de erro.
 */
bool lewis_inference_init(void);

/**
 * @brief Executa inferência em um batimento já quantizado em int8.
 *
 * @param input  Array de 500 amostras int8.
 * @param output Array de 5 logits/probabilidades int8 de saída.
 * @return true se sucesso, false em caso de erro.
 */
bool lewis_inference_run(const int8_t input[LEWIS_INPUT_LEN],
                         int8_t output[LEWIS_OUTPUT_LEN]);

/**
 * @brief Retorna o tamanho do FlatBuffer do modelo em bytes.
 */
size_t lewis_inference_model_size(void);

/**
 * @brief Retorna a arena TFLM utilizada em bytes (após init).
 */
size_t lewis_inference_arena_used(void);

#ifdef __cplusplus
}
#endif

#endif /* LEWIS_INFERENCE_H */
