/*
 * Detector leve de R-peaks — implementacao em ponto flutuante simples.
 *
 * Algoritmo:
 *   1. Derivada de 5 pontos (Pan-Tompkins) no sinal filtrado.
 *   2. Quadrado ponto-a-ponto.
 *   3. Integracao por janela movel retangular de 150 ms.
 *   4. Threshold adaptativo (SPKF/NPKF) com periodo refratario de 360 ms.
 *   5. Search-back leve (fator 1.66 * RR medio) para batimentos perdidos.
 *   6. Discriminacao de onda T: se dois picos estiverem dentro do periodo
 *      refratario, mantem o de maior inclinacao no sinal original.
 */

#include "dsp/r_peak_detector.h"

#include <math.h>
#include <string.h>

/* Kernel da derivada de 5 pontos (normalizado para ganho unitario em DC). */
static const float DERIVATIVE_KERNEL[5] = {-0.125f, -0.25f, 0.0f, 0.25f, 0.125f};

static float clamped_sample(const float* sig, size_t len, int idx)
{
    if (idx < 0) {
        return sig[0];
    }
    if ((size_t)idx >= len) {
        return sig[len - 1U];
    }
    return sig[(size_t)idx];
}

static void derivative_5point(const float* in, float* out, size_t len)
{
    for (size_t i = 0; i < len; ++i) {
        float acc = 0.0f;
        for (int k = -2; k <= 2; ++k) {
            acc += DERIVATIVE_KERNEL[k + 2] * clamped_sample(in, len, (int)i + k);
        }
        out[i] = acc;
    }
}

static void square_signal(const float* in, float* out, size_t len)
{
    for (size_t i = 0; i < len; ++i) {
        out[i] = in[i] * in[i];
    }
}

static void moving_window_integration(
    const float* in,
    float* out,
    size_t len,
    size_t window
)
{
    if (window == 0U || len == 0U) {
        return;
    }

    float sum = 0.0f;
    for (size_t i = 0; i < len; ++i) {
        sum += in[i];
        if (i >= window) {
            sum -= in[i - window];
        }
        size_t denom = (i + 1U < window) ? (i + 1U) : window;
        out[i] = sum / (float)denom;
    }
}

static float array_max(const float* arr, size_t len)
{
    float m = arr[0];
    for (size_t i = 1; i < len; ++i) {
        if (arr[i] > m) {
            m = arr[i];
        }
    }
    return m;
}

static float local_slope(const float* sig, size_t len, size_t idx, size_t win)
{
    if (len < 2U || win == 0U) {
        return 0.0f;
    }
    float max_slope = 0.0f;
    size_t start = (idx > win) ? (idx - win) : 0U;
    size_t end = (idx + win + 1U < len) ? (idx + win + 1U) : len;
    for (size_t i = start + 1U; i < end; ++i) {
        float s = fabsf(sig[i] - sig[i - 1U]);
        if (s > max_slope) {
            max_slope = s;
        }
    }
    return max_slope;
}

static void t_wave_discrimination(
    const float* sig,
    size_t len,
    size_t* peaks,
    size_t* n_peaks,
    size_t refractory,
    float fs
)
{
    if (*n_peaks < 2U) {
        return;
    }

    size_t win = (size_t)roundf(0.040f * fs); /* 40 ms */
    if (win < 2U) {
        win = 2U;
    }

    size_t write = 1U;
    for (size_t i = 1U; i < *n_peaks; ++i) {
        size_t current = peaks[i];
        size_t previous = peaks[write - 1U];
        if ((current - previous) < refractory) {
            float slope_current = local_slope(sig, len, current, win);
            float slope_previous = local_slope(sig, len, previous, win);
            if (slope_current > slope_previous) {
                peaks[write - 1U] = current; /* substitui anterior */
            }
            /* caso contrario, descarta current */
        } else {
            peaks[write++] = current;
        }
    }
    *n_peaks = write;
}

int lewis_detect_r_peaks(
    const float* sig,
    size_t len,
    float fs,
    size_t* peaks,
    size_t* n_peaks
)
{
    if (sig == NULL || peaks == NULL || n_peaks == NULL || len == 0U || fs <= 0.0f) {
        return -1;
    }
    if (len > LEWIS_RPEAK_MAX_SAMPLES) {
        return -2;
    }

    size_t mwi_window = (size_t)roundf(0.150f * fs);
    if (mwi_window == 0U) {
        mwi_window = 1U;
    }
    size_t refractory = (size_t)roundf(0.360f * fs);
    if (refractory == 0U) {
        refractory = 1U;
    }

    /* VLA proporcional ao sinal de entrada: economiza pilha no embarcado
     * e permite sinais maiores no host nativo. */
    float deriv[len];
    float squared[len];
    float mwi[len];

    derivative_5point(sig, deriv, len);
    square_signal(deriv, squared, len);
    moving_window_integration(squared, mwi, len, mwi_window);

    /* Inicializa SPKF e NPKF com as primeiras 2 s de sinal transformado. */
    float spkf = 0.0f;
    float npkf = 0.0f;
    float rr_average1 = 0.8f * fs; /* ~800 ms padrao */
    size_t init_len = (size_t)(2.0f * fs);
    if (init_len > len / 4U) {
        init_len = len / 4U;
    }
    if (init_len > 10U) {
        float init_max = array_max(mwi, init_len);
        spkf = init_max * 0.25f;
        npkf = init_max * 0.125f;
    }

    size_t count = 0U;
    size_t last_peak = 0U;
    int has_last = 0;
    const float search_back_factor = 1.66f;

    size_t i = 0U;
    while (i < len && count < LEWIS_RPEAK_MAX_PEAKS) {
        size_t window_end = i + refractory;
        if (window_end > len) {
            window_end = len;
        }

        /* Encontra maximo local na proxima janela refrataria. */
        size_t local_max_idx = i;
        float local_max_val = mwi[i];
        for (size_t j = i + 1U; j < window_end; ++j) {
            if (mwi[j] > local_max_val) {
                local_max_val = mwi[j];
                local_max_idx = j;
            }
        }

        float thresh1 = npkf + 0.25f * (spkf - npkf);
        float thresh2 = 0.5f * thresh1;

        int accepted = 0;
        if (local_max_val > thresh1) {
            if (!has_last || (local_max_idx - last_peak) >= refractory) {
                peaks[count++] = local_max_idx;
                spkf = 0.125f * local_max_val + 0.875f * spkf;
                accepted = 1;
            } else {
                npkf = 0.125f * local_max_val + 0.875f * npkf;
            }
        } else if (local_max_val > thresh2 && has_last) {
            float expected_rr = search_back_factor * rr_average1;
            if ((float)(local_max_idx - last_peak) > expected_rr) {
                peaks[count++] = local_max_idx;
                spkf = 0.125f * local_max_val + 0.875f * spkf;
                accepted = 1;
            } else {
                npkf = 0.125f * local_max_val + 0.875f * npkf;
            }
        } else {
            if (local_max_val > 0.0f) {
                npkf = 0.125f * local_max_val + 0.875f * npkf;
            }
        }

        if (accepted) {
            if (count >= 2U) {
                /* Media movel dos ultimos 8 RR. */
                size_t n_rr = (count - 1U < 8U) ? (count - 1U) : 8U;
                float sum = 0.0f;
                for (size_t k = 0; k < n_rr; ++k) {
                    sum += (float)(peaks[count - 1U - k] - peaks[count - 2U - k]);
                }
                rr_average1 = sum / (float)n_rr;
            }
            last_peak = local_max_idx;
            has_last = 1;
            i = local_max_idx + 1U;
        } else {
            i = window_end;
        }
    }

    t_wave_discrimination(sig, len, peaks, &count, refractory, fs);

    *n_peaks = count;
    return 0;
}
