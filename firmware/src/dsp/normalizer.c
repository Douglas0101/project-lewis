#include "dsp/normalizer.h"

#include <math.h>

void lewis_zscore_normalize(float* buffer, size_t len)
{
    if (!buffer || len == 0) {
        return;
    }

    /* Primeira passada: media. */
    float sum = 0.0f;
    for (size_t i = 0; i < len; ++i) {
        sum += buffer[i];
    }
    float mean = sum / (float)len;

    /* Segunda passada: desvio padrao (populacional). */
    float sq_sum = 0.0f;
    for (size_t i = 0; i < len; ++i) {
        float diff = buffer[i] - mean;
        sq_sum += diff * diff;
    }
    float std = sqrtf(sq_sum / (float)len);

    /* Normaliza in-place. */
    if (std > 0.0f) {
        for (size_t i = 0; i < len; ++i) {
            buffer[i] = (buffer[i] - mean) / std;
        }
    } else {
        for (size_t i = 0; i < len; ++i) {
            buffer[i] = 0.0f;
        }
    }
}
