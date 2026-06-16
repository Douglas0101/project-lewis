#include "dsp/adc_stub.h"

/* LCG deterministico para geracao de sinais de teste sem dependencia da libm. */
static uint32_t lcg_next(uint32_t* state)
{
    *state = (*state * 1103515245U) + 12345U;
    return *state;
}

/* Seno aproximado por tabela lookup (0..255 -> -1..+1, em 1/256 de ciclo). */
static int32_t sin_lut(uint8_t phase_256)
{
    /* Tabela de seno * 256 para fases 0..63 (simetria nos outros quadrantes). */
    static const int16_t sine_table[64] = {
        0, 6, 13, 19, 25, 31, 38, 44, 50, 56, 62, 68, 74, 80, 86, 92,
        98, 104, 109, 115, 121, 126, 132, 137, 142, 147, 152, 157, 162, 167, 171, 176,
        180, 184, 188, 192, 196, 199, 203, 206, 209, 212, 215, 218, 220, 223, 225, 227,
        229, 231, 233, 234, 236, 237, 238, 239, 240, 241, 241, 242, 242, 242, 243, 243,
    };
    uint8_t quadrant = phase_256 >> 6;
    uint8_t idx = phase_256 & 0x3F;
    int32_t value;
    if (quadrant == 0) {
        value = sine_table[idx];
    } else if (quadrant == 1) {
        value = sine_table[63 - idx];
    } else if (quadrant == 2) {
        value = -sine_table[idx];
    } else {
        value = -sine_table[63 - idx];
    }
    return value;
}

/* Exponencial aproximada: exp(-x) para x >= 0, com precisao suficiente para teste. */
static float exp_approx(float x)
{
    if (x <= 0.0f) {
        return 1.0f;
    }
    if (x > 5.0f) {
        return 0.0f;
    }
    /* Aproximacao por serie de Taylor truncada: exp(-x) = 1 - x + x^2/2! - x^3/3! + ... */
    float term = 1.0f;
    float sum = 1.0f;
    for (int i = 1; i <= 8; ++i) {
        term *= -x / (float)i;
        sum += term;
    }
    return sum;
}

size_t lewis_adc_stub_get_beat(uint32_t idx, int8_t out_buffer[500])
{
    uint32_t state = 42U + idx * 7U;
    for (size_t i = 0; i < 500; ++i) {
        const float t = (float)i / 500.0f;
        const uint8_t phase = (uint8_t)((i * 256U) / 500U);
        const float qrs = 0.8f * exp_approx(-200.0f * (t - 0.5f) * (t - 0.5f));
        const float baseline = 0.05f * ((float)sin_lut((uint8_t)(phase / 3U)) / 256.0f);
        const uint32_t r = lcg_next(&state);
        const float noise = ((float)(r % 101U) - 50.0f) / 500.0f; /* ~ +/- 0.1 mV */
        float sample = qrs + baseline + noise;

        /* Clip para int8 aproximado em mV: assume 1 mV = 40 counts. */
        int32_t value = (int32_t)(sample * 40.0f);
        if (value > 127) {
            value = 127;
        } else if (value < -128) {
            value = -128;
        }
        out_buffer[i] = (int8_t)value;
    }
    return 500;
}

size_t lewis_adc_stub_get_sine(float amplitude_mv, int8_t out_buffer[500])
{
    for (size_t i = 0; i < 500; ++i) {
        const uint8_t phase = (uint8_t)((i * 256U) / 500U);
        float sample = amplitude_mv * ((float)sin_lut(phase) / 256.0f);
        int32_t value = (int32_t)(sample * 40.0f);
        if (value > 127) {
            value = 127;
        } else if (value < -128) {
            value = -128;
        }
        out_buffer[i] = (int8_t)value;
    }
    return 500;
}
