/* Pos-processamento de calibracao do modelo pre-treinado A2-full.
 *
 * Ordem de inferencia: logits int8 -> dequant (scale/zero_point) -> /T -> sigmoid.
 * T = 0.3741 < 1 (modelo sub-confiante; Mukhoti 2020). A divisao por T ocorre
 * em float32 apos a dequantizacao — nao ha caminho de overflow int8.
 */
#ifndef LEWIS_PRETRAIN_CALIBRATE_H
#define LEWIS_PRETRAIN_CALIBRATE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Converte logits int8 em probabilidades calibradas (dequant -> /T -> sigmoid).
 *
 * @param logits      Vetor de logits int8 (saida do TFLM).
 * @param n           Numero de logits (PRETRAIN_A2_FULL_OUTPUT_LEN = 5).
 * @param scale       Escala de dequantizacao do tensor de saida.
 * @param zero_point  Zero-point do tensor de saida.
 * @param temperature Temperatura T (PRETRAIN_A2_FULL_TEMPERATURE).
 * @param probs_out   Vetor float32 de saida com n probabilidades em [0, 1].
 */
void lewis_pretrain_apply_temperature_sigmoid(const int8_t *logits,
                                              size_t n,
                                              float scale,
                                              int32_t zero_point,
                                              float temperature,
                                              float *probs_out);

#ifdef __cplusplus
}
#endif

#endif /* LEWIS_PRETRAIN_CALIBRATE_H */
