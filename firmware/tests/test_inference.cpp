#include "harness.h"
#include "ml/inference.h"
#include "ml/model_data.h"
#include "fixtures/generated/fixture_pipeline.h"
#include <string.h>

extern "C" {

static void test_model_size(harness_result_t* r) {
    size_t sz_total = lewis_inference_model_size();
    size_t sz_stage1 = lewis_stage1_model_size();
    size_t sz_stage2 = lewis_stage2_model_size();
    size_t expected_total = (size_t)stage1_int8_v2_0_len + (size_t)stage2_int8_v2_0_len;

    harness_assert_int_eq(r, (int64_t)expected_total, (int64_t)sz_total, "total model_size matches header");
    harness_assert_true(r, sz_stage1 > 0 && sz_stage1 < 65536, "stage1 FlatBuffer below 64KB (QG6)");
    harness_assert_true(r, sz_stage2 > 0 && sz_stage2 < 65536, "stage2 FlatBuffer below 64KB (QG6)");
    harness_assert_true(r, sz_total < (512U * 1024U), "combined model_size below 512KB Flash (QG9)");
}

static void test_inference_init(harness_result_t* r) {
    bool ok = lewis_inference_init();
    harness_assert_true(r, ok, "inference_init returns true");
}

#if 0 && LEWIS_USE_TFLM

static const int8_t* select_fixture_input(int idx) {
    switch (idx) {
        case 0: return fixture_pipeline_input_0_int8;
        case 1: return fixture_pipeline_input_1_int8;
        case 2: return fixture_pipeline_input_2_int8;
        case 3: return fixture_pipeline_input_3_int8;
        case 4: return fixture_pipeline_input_4_int8;
        default: return nullptr;
    }
}

static const int8_t* select_fixture_expected(int idx) {
    switch (idx) {
        case 0: return fixture_pipeline_expected_0;
        case 1: return fixture_pipeline_expected_1;
        case 2: return fixture_pipeline_expected_2;
        case 3: return fixture_pipeline_expected_3;
        case 4: return fixture_pipeline_expected_4;
        default: return nullptr;
    }
}

static void test_inference_bitexact(harness_result_t* r) {
    bool ok = lewis_inference_init();
    harness_assert_true(r, ok, "inference_init");

    int8_t output[LEWIS_OUTPUT_LEN];
    for (int idx = 0; idx < LEWIS_FIXTURE_PIPELINE_COUNT; idx++) {
        const int8_t* input = select_fixture_input(idx);
        const int8_t* expected = select_fixture_expected(idx);
        if (!input || !expected) {
            harness_assert_true(r, false, "invalid fixture index");
            return;
        }

        ok = lewis_inference_run(input, output);
        harness_assert_true(r, ok, "inference_run");

        for (int i = 0; i < LEWIS_OUTPUT_LEN; i++) {
            harness_assert_int8_eq(r, expected[i], output[i], "bitexact output");
        }
    }
}

static void test_inference_fidelity(harness_result_t* r) {
    /* Fidelidade numérica: verifica que a inferência TFLM embarcada produz
     * exatamente os mesmos logits int8 do interpretador Python BUILTIN_REF.
     * O teste de bit-exatidão acima já cobre isso; este teste documenta o
     * quality gate QG10 de forma explicita. */
    bool ok = lewis_inference_init();
    harness_assert_true(r, ok, "inference_init");

    int8_t output[LEWIS_OUTPUT_LEN];
    for (int idx = 0; idx < LEWIS_FIXTURE_PIPELINE_COUNT; idx++) {
        const int8_t* input = select_fixture_input(idx);
        const int8_t* expected = select_fixture_expected(idx);
        if (!input || !expected) {
            harness_assert_true(r, false, "invalid fixture index");
            return;
        }

        ok = lewis_inference_run(input, output);
        harness_assert_true(r, ok, "inference_run");

        for (int i = 0; i < LEWIS_OUTPUT_LEN; i++) {
            harness_assert_int8_eq(r, expected[i], output[i], "fidelity output");
        }
    }
}

#endif /* LEWIS_USE_TFLM */

void suite_inference_register(void) {
    harness_register("INFERENCE", "model_size", test_model_size);
    harness_register("INFERENCE", "init", test_inference_init);
    /* TODO: reativar bitexact/fidelity apos ajustar fixtures two-stage e
     * confirmar Invoke estavel no harness nativo. */
#if 0 && LEWIS_USE_TFLM
    harness_register("INFERENCE", "bitexact", test_inference_bitexact);
    harness_register("INFERENCE", "fidelity", test_inference_fidelity);
#endif
}

} /* extern "C" */
