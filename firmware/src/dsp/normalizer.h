#ifndef LEWIS_NORMALIZER_H
#define LEWIS_NORMALIZER_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Aplica Z-score em um bloco de amostras float32 (in-place).
 *
 * Calcula media e desvio padrao da janela e normaliza:
 *   y[n] = (x[n] - mean) / std
 *
 * Se o desvio padrao for zero, a saida e zero.
 *
 * @param buffer Buffer de entrada/saida (float).
 * @param len Numero de amostras.
 */
void lewis_zscore_normalize(float* buffer, size_t len);

#ifdef __cplusplus
}
#endif

#endif /* LEWIS_NORMALIZER_H */
