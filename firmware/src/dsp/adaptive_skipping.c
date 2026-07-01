#include "dsp/adaptive_skipping.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>

void lewis_adaptive_skipping_init(lewis_adaptive_skipping_t* ctx)
{
    if (ctx == NULL) {
        return;
    }

    for (size_t i = 0; i < LEWIS_ADAPTIVE_SKIPPING_MAX_WINDOW; ++i) {
        ctx->rr_history[i] = 0U;
    }
    ctx->count = 0U;
    ctx->head = 0U;
    ctx->last_class = 0U;
    ctx->has_last_class = false;
}

void lewis_adaptive_skipping_update_class(
    lewis_adaptive_skipping_t* ctx,
    uint32_t class_id
)
{
    if (ctx == NULL) {
        return;
    }
    ctx->last_class = class_id;
    ctx->has_last_class = true;
}

/**
 * @brief Calcula a media aritmetica de um subconjunto do historico.
 *
 * @param ctx Contexto com o historico circular.
 * @return Media em milissegundos como float.
 */
static float compute_mean(const lewis_adaptive_skipping_t* ctx)
{
    uint32_t sum = 0U;
    for (uint8_t i = 0U; i < ctx->count; ++i) {
        sum += ctx->rr_history[i];
    }
    return (float)sum / (float)ctx->count;
}

/**
 * @brief Calcula a maior variacao absoluta em relacao a media.
 *
 * @param ctx  Contexto com o historico circular.
 * @param mean Media dos valores do historico.
 * @return Maior desvio absoluto em milissegundos.
 */
static float compute_max_abs_deviation(
    const lewis_adaptive_skipping_t* ctx,
    float mean
)
{
    float max_dev = 0.0f;
    for (uint8_t i = 0U; i < ctx->count; ++i) {
        float dev = (float)ctx->rr_history[i] - mean;
        if (dev < 0.0f) {
            dev = -dev;
        }
        if (dev > max_dev) {
            max_dev = dev;
        }
    }
    return max_dev;
}

bool lewis_adaptive_skipping_should_skip(
    lewis_adaptive_skipping_t* ctx,
    uint32_t rr_ms,
    uint32_t threshold_ms,
    float threshold_ratio
)
{
    if (ctx == NULL) {
        return false;
    }

    /* Sempre insere o RR atual no historico circular. */
    if (ctx->count < LEWIS_ADAPTIVE_SKIPPING_MAX_WINDOW) {
        ctx->rr_history[ctx->count] = rr_ms;
        ctx->count++;
    } else {
        ctx->rr_history[ctx->head] = rr_ms;
        ctx->head++;
        if (ctx->head >= LEWIS_ADAPTIVE_SKIPPING_MAX_WINDOW) {
            ctx->head = 0U;
        }
    }

#if !ADAPTIVE_SKIPPING_ENABLED
    return false;
#endif

    if (ctx->count < LEWIS_ADAPTIVE_SKIPPING_MIN_CYCLES) {
        return false;
    }
    if (!ctx->has_last_class) {
        return false;
    }

    const float mean = compute_mean(ctx);
    if (mean <= 0.0f) {
        return false;
    }

    const float max_dev = compute_max_abs_deviation(ctx, mean);
    const float ratio = max_dev / mean;

    return (max_dev <= (float)threshold_ms) && (ratio <= threshold_ratio);
}
