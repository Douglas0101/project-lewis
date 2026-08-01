#include "ml/pretrain_calibrate.h"

#include <math.h>

void lewis_pretrain_apply_temperature_sigmoid(const int8_t *logits,
                                              size_t n,
                                              float scale,
                                              int32_t zero_point,
                                              float temperature,
                                              float *probs_out)
{
    if (logits == NULL || probs_out == NULL || temperature <= 0.0f) {
        return;
    }
    for (size_t i = 0U; i < n; i++) {
        /* dequant -> /T (float32) -> sigmoid, com clamp anti-overflow do expf */
        float z = (((float)logits[i] - (float)zero_point) * scale) / temperature;
        if (z > 30.0f) {
            z = 30.0f;
        } else if (z < -30.0f) {
            z = -30.0f;
        }
        probs_out[i] = 1.0f / (1.0f + expf(-z));
    }
}
