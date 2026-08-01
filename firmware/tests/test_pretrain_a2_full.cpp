/* Suite PRETRAIN_A2 — validacao do modelo A2-full no harness (T5, C09).
 *
 * Cobre: QG6/QG7 (tamanho), QG8 (bit-exatidao int8 vs BUILTIN_REF, atol 1 LSB
 * para CMSIS-NN), QG9 (latencia < 200 ms), QG10 (cosseno C vs Python > 0.99
 * nas probabilidades calibradas) e arena TFLM (< 64 KB).
 */
#include "harness.h"
#include "hal/hal.h"
#include "ml/inference.h"
#include "ml/model_data.h"
#include "ml/pretrain_a2_full_config.h"
#include "ml/pretrain_a2_full_quant_params.h"
#include "ml/pretrain_calibrate.h"
#include "fixtures/generated/pretrain_a2_full_fixtures.h"

#include <math.h>
#include <stddef.h>

extern "C" {

static void test_pretrain_model_size(harness_result_t* r) {
    size_t sz = lewis_pretrain_model_size();
    harness_assert_int_eq(r,
                          (int64_t)pretrain_a2_full_int8_len,
                          (int64_t)sz,
                          "pretrain model_size matches header");
    harness_assert_true(r, sz > 0U && sz < 65536U, "pretrain FlatBuffer below 64KB (QG6)");
}

static void test_pretrain_init_arena(harness_result_t* r) {
    bool ok = lewis_pretrain_init();
    harness_assert_true(r, ok, "pretrain_init returns true");
    size_t arena = lewis_inference_arena_used();
    harness_print("[pretrain_a2] arena_used=%lu bytes\n", (unsigned long)arena);
    harness_assert_true(r, arena > 0U && arena <= (64U * 1024U),
                        "arena TFLM <= 64KB (QG9)");
}

static void test_pretrain_bitexact(harness_result_t* r) {
    bool ok = lewis_pretrain_init();
    harness_assert_true(r, ok, "pretrain_init");

    int8_t logits[LEWIS_PRETRAIN_OUTPUT_LEN];
    for (int idx = 0; idx < PRETRAIN_A2_FULL_FIXTURE_COUNT; idx++) {
        ok = lewis_pretrain_run(pretrain_a2_fx_inputs[idx], logits);
        harness_assert_true(r, ok, "pretrain_run");
        for (int i = 0; i < LEWIS_PRETRAIN_OUTPUT_LEN; i++) {
            /* atol 1 LSB: tolerancia documentada CMSIS-NN vs BUILTIN_REF (QG8) */
            harness_assert_int_close(r,
                                     (int64_t)pretrain_a2_fx_logits[idx][i],
                                     (int64_t)logits[i],
                                     1,
                                     "bitexact logit (atol 1 LSB)");
        }
    }
}

static void test_pretrain_calibration_fidelity(harness_result_t* r) {
    bool ok = lewis_pretrain_init();
    harness_assert_true(r, ok, "pretrain_init");

    float dot = 0.0f;
    float norm_exp = 0.0f;
    float norm_got = 0.0f;
    int8_t logits[LEWIS_PRETRAIN_OUTPUT_LEN];
    float probs[LEWIS_PRETRAIN_OUTPUT_LEN];

    for (int idx = 0; idx < PRETRAIN_A2_FULL_FIXTURE_COUNT; idx++) {
        ok = lewis_pretrain_run(pretrain_a2_fx_inputs[idx], logits);
        harness_assert_true(r, ok, "pretrain_run");
        lewis_pretrain_apply_temperature_sigmoid(
            logits,
            LEWIS_PRETRAIN_OUTPUT_LEN,
            PRETRAIN_A2_FULL_QUANT_PARAMS_OUTPUT_SCALE,
            PRETRAIN_A2_FULL_QUANT_PARAMS_OUTPUT_ZERO_POINT,
            PRETRAIN_A2_FULL_TEMPERATURE,
            probs);
        for (int i = 0; i < LEWIS_PRETRAIN_OUTPUT_LEN; i++) {
            float expected = pretrain_a2_fx_probs[idx][i];
            harness_assert_float_close(r, expected, probs[i], 2e-3f,
                                       "calibrated prob close");
            dot += expected * probs[i];
            norm_exp += expected * expected;
            norm_got += probs[i] * probs[i];
        }
    }
    float cosine = dot / (sqrtf(norm_exp) * sqrtf(norm_got));
    harness_print("[pretrain_a2] calibration cosine=%.6f\n", (double)cosine);
    harness_assert_true(r, cosine > 0.99f, "cosine C vs Python > 0.99 (QG10)");
}

static void test_pretrain_latency(harness_result_t* r) {
    bool ok = lewis_pretrain_init();
    harness_assert_true(r, ok, "pretrain_init");

    int8_t logits[LEWIS_PRETRAIN_OUTPUT_LEN];
    /* millis (TIM2) funciona em native e Renode; o benchmark SysTick nao e
     * confiavel no Renode (down-counter nao emulado — ver hal_sim.c). */
    uint32_t t0 = lewis_hal_millis();
    ok = lewis_pretrain_run(pretrain_a2_fx_inputs[0], logits);
    uint32_t t1 = lewis_hal_millis();
    harness_assert_true(r, ok, "pretrain_run");

    uint32_t ms = t1 - t0;
    harness_print("[pretrain_a2] latency=%lu ms\n", (unsigned long)ms);
    harness_assert_true(r, ms < 200U, "latency < 200 ms (QG9)");
}

void suite_pretrain_a2_full_register(void) {
    harness_register("PRETRAIN_A2", "model_size", test_pretrain_model_size);
    harness_register("PRETRAIN_A2", "init_arena", test_pretrain_init_arena);
    harness_register("PRETRAIN_A2", "bitexact", test_pretrain_bitexact);
    harness_register("PRETRAIN_A2", "calibration_fidelity", test_pretrain_calibration_fidelity);
    harness_register("PRETRAIN_A2", "latency", test_pretrain_latency);
}

} /* extern "C" */
