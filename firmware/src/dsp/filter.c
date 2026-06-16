#include "dsp/filter.h"
#include "dsp/filter_coeffs.h"

#include <string.h>

void lewis_biquad_init(lewis_biquad_cascade_t* cascade,
                       const float* coeffs,
                       uint32_t num_sections,
                       lewis_biquad_state_t* state)
{
    cascade->coeffs = coeffs;
    cascade->num_sections = num_sections;
    cascade->state = state;
}

void lewis_biquad_reset(lewis_biquad_cascade_t* cascade)
{
    if (cascade && cascade->state) {
        memset(cascade->state, 0,
               sizeof(lewis_biquad_state_t) * cascade->num_sections);
    }
}

float lewis_biquad_sample(lewis_biquad_cascade_t* cascade, float x)
{
    const float* c = cascade->coeffs;
    lewis_biquad_state_t* s = cascade->state;
    float y = x;

    for (uint32_t i = 0; i < cascade->num_sections; ++i) {
        const float b0 = c[5 * i + 0];
        const float b1 = c[5 * i + 1];
        const float b2 = c[5 * i + 2];
        const float a1 = c[5 * i + 3];
        const float a2 = c[5 * i + 4];

        const float in = y;
        y = b0 * in + s[i].d0;
        s[i].d0 = b1 * in - a1 * y + s[i].d1;
        s[i].d1 = b2 * in - a2 * y;
    }

    return y;
}

void lewis_biquad_process_block(lewis_biquad_cascade_t* cascade,
                                const float* input,
                                float* output,
                                size_t len)
{
    for (size_t i = 0; i < len; ++i) {
        output[i] = lewis_biquad_sample(cascade, input[i]);
    }
}

void lewis_filter_chain_init(lewis_filter_chain_t* chain)
{
    lewis_biquad_init(&chain->bandpass,
                      LEWIS_BANDPASS_coeffs,
                      LEWIS_BANDPASS_SECTIONS,
                      chain->bandpass_state);
    lewis_biquad_init(&chain->notch,
                      LEWIS_NOTCH_coeffs,
                      LEWIS_NOTCH_SECTIONS,
                      chain->notch_state);
}

void lewis_filter_chain_reset(lewis_filter_chain_t* chain)
{
    lewis_biquad_reset(&chain->bandpass);
    lewis_biquad_reset(&chain->notch);
}

void lewis_filter_chain_process(lewis_filter_chain_t* chain,
                                const float* input,
                                float* output,
                                size_t len)
{
    /* bandpass -> notch; pode ser in-place. */
    lewis_biquad_process_block(&chain->bandpass, input, output, len);
    lewis_biquad_process_block(&chain->notch, output, output, len);
}
