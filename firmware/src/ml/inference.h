#ifndef LEWIS_INFERENCE_H
#define LEWIS_INFERENCE_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Dimensões comuns de entrada e dos dois estágios.
 */
#define LEWIS_INPUT_LEN            500
#define LEWIS_STAGE1_OUTPUT_LEN    2
#define LEWIS_STAGE2_OUTPUT_LEN    3

/* Alias de compatibilidade v1.1: o pipeline legacy é mapeado para o Estágio 1.
 * Será removido quando todos os callers migrarem para o pipeline two-stage. */
#define LEWIS_OUTPUT_LEN           LEWIS_STAGE1_OUTPUT_LEN

/**
 * @brief Inicializa o Estágio 1 (N vs Anormal).
 *
 * @return true se sucesso, false em caso de erro.
 */
bool lewis_stage1_init(void);

/**
 * @brief Executa inferência do Estágio 1 em um batimento quantizado em int8.
 *
 * @param input  Array de 500 amostras int8.
 * @param output Array de 2 logits/probabilidades int8 de saída.
 * @return true se sucesso, false em caso de erro.
 */
bool lewis_stage1_run(const int8_t input[LEWIS_INPUT_LEN],
                      int8_t output[LEWIS_STAGE1_OUTPUT_LEN]);

/**
 * @brief Inicializa o Estágio 2 (S vs V vs F).
 *
 * Destrói o interpretador do Estágio 1 caso esteja ativo, reutilizando a
 * mesma arena TFLM (pipeline sequencial).
 *
 * @return true se sucesso, false em caso de erro.
 */
bool lewis_stage2_init(void);

/**
 * @brief Executa inferência do Estágio 2 em um batimento quantizado em int8.
 *
 * @param input  Array de 500 amostras int8.
 * @param output Array de 3 logits/probabilidades int8 de saída.
 * @return true se sucesso, false em caso de erro.
 */
bool lewis_stage2_run(const int8_t input[LEWIS_INPUT_LEN],
                      int8_t output[LEWIS_STAGE2_OUTPUT_LEN]);

/**
 * @brief Inicializa ambos os estágios (compatibilidade com API legacy).
 *
 * Equivalente a lewis_stage1_init().
 */
bool lewis_inference_init(void);

/**
 * @brief Executa inferência legacy (Estágio 1).
 */
bool lewis_inference_run(const int8_t input[LEWIS_INPUT_LEN],
                         int8_t output[LEWIS_OUTPUT_LEN]);

/**
 * @brief Tamanho total dos FlatBuffers (stage1 + stage2) em bytes.
 */
size_t lewis_inference_model_size(void);

/**
 * @brief Tamanho do FlatBuffer do Estágio 1 em bytes.
 */
size_t lewis_stage1_model_size(void);

/**
 * @brief Tamanho do FlatBuffer do Estágio 2 em bytes.
 */
size_t lewis_stage2_model_size(void);

/**
 * @brief Dimensão de saída do modelo pré-treinado A2-full (5 superclasses SCP-ECG).
 */
#define LEWIS_PRETRAIN_OUTPUT_LEN  5

/**
 * @brief Inicializa o modelo pré-treinado A2-full (logits multi-label).
 *
 * Modelo adicional (T4/C08): não substitui os estágios 1/2. Compartilha a
 * mesma arena TFLM (pipeline sequencial — init destrói o estágio ativo).
 *
 * @return true se sucesso, false em caso de erro.
 */
bool lewis_pretrain_init(void);

/**
 * @brief Executa inferência do A2-full em uma janela de 500 amostras int8.
 *
 * @param input  Array de 500 amostras int8.
 * @param output Array de 5 logits int8 de saída (sigmoid removido na conversão;
 *               aplicar /T + sigmoid via lewis_pretrain_apply_temperature_sigmoid).
 * @return true se sucesso, false em caso de erro.
 */
bool lewis_pretrain_run(const int8_t input[LEWIS_INPUT_LEN],
                        int8_t output[LEWIS_PRETRAIN_OUTPUT_LEN]);

/**
 * @brief Tamanho do FlatBuffer do A2-full em bytes.
 */
size_t lewis_pretrain_model_size(void);

/**
 * @brief Retorna a arena TFLM utilizada em bytes (após init).
 */
size_t lewis_inference_arena_used(void);

#ifdef __cplusplus
}
#endif

#endif /* LEWIS_INFERENCE_H */
