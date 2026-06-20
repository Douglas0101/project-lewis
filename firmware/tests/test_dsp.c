#include "harness.h"
#include "dsp/filter.h"
#include "dsp/filter_coeffs.h"
#include "dsp/normalizer.h"
#include "fixtures/generated/fixture_dsp.h"
#include <string.h>

static void test_biquad_identity(harness_result_t* r) {
    lewis_biquad_state_t state[1];
    lewis_biquad_cascade_t cascade;
    /* Coeficientes pass-through: b0=1, b1=b2=a1=a2=0 */
    const float coeffs[5] = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    lewis_biquad_init(&cascade, coeffs, 1, state);
    lewis_biquad_reset(&cascade);

    float input[10] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 0.0f, -1.0f, -2.0f, -3.0f, -4.0f};
    float output[10];
    lewis_biquad_process_block(&cascade, input, output, 10);

    for (int i = 0; i < 10; i++) {
        harness_assert_float_close(r, input[i], output[i], 1e-6f, "biquad identity");
    }
}

static void test_filter_chain_init(harness_result_t* r) {
    lewis_filter_chain_t chain;
    lewis_filter_chain_init(&chain);
    lewis_filter_chain_reset(&chain);
    harness_assert_true(r, chain.bandpass.num_sections > 0, "bandpass sections > 0");
    harness_assert_true(r, chain.notch.num_sections > 0, "notch sections > 0");
}

static void test_filter_chain_silence(harness_result_t* r) {
    lewis_filter_chain_t chain;
    lewis_filter_chain_init(&chain);
    lewis_filter_chain_reset(&chain);

    float input[100];
    float output[100];
    memset(input, 0, sizeof(input));
    lewis_filter_chain_process(&chain, input, output, 100);

    for (int i = 0; i < 100; i++) {
        harness_assert_float_close(r, 0.0f, output[i], 1e-6f, "silence output");
    }
}

static void test_filter_chain_vs_python(harness_result_t* r) {
    lewis_filter_chain_t chain;
    float output[500];

    for (int idx = 0; idx < LEWIS_FIXTURE_DSP_COUNT; idx++) {
        const float* input = NULL;
        const float* expected = NULL;
        switch (idx) {
            case 0: input = fixture_dsp_input_0; expected = fixture_dsp_expected_0; break;
            case 1: input = fixture_dsp_input_1; expected = fixture_dsp_expected_1; break;
            case 2: input = fixture_dsp_input_2; expected = fixture_dsp_expected_2; break;
            case 3: input = fixture_dsp_input_3; expected = fixture_dsp_expected_3; break;
            case 4: input = fixture_dsp_input_4; expected = fixture_dsp_expected_4; break;
            default:
                harness_assert_true(r, false, "invalid fixture index");
                return;
        }

        lewis_filter_chain_init(&chain);
        lewis_filter_chain_reset(&chain);
        lewis_filter_chain_process(&chain, input, output, 500);
        lewis_zscore_normalize(output, 500);

        for (int i = 0; i < 500; i++) {
            harness_assert_float_close(r, expected[i], output[i], 1e-4f, "filter_chain_vs_python");
        }
    }
}

void suite_dsp_register(void) {
    harness_register("DSP", "biquad_identity", test_biquad_identity);
    harness_register("DSP", "filter_chain_init", test_filter_chain_init);
    harness_register("DSP", "filter_chain_silence", test_filter_chain_silence);
    harness_register("DSP", "filter_chain_vs_python", test_filter_chain_vs_python);
}
