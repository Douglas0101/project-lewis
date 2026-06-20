#include "harness.h"
#include "hal/hal.h"

int main(void) {
    lewis_hal_init();

    suite_dsp_register();
    suite_r_peak_register();
    suite_inference_register();
#if LEWIS_USE_TFLM
    suite_pipeline_register();
#endif

    harness_run_all();

    int failures = harness_summary();

#ifdef LEWIS_HOST_NATIVE
    return failures;
#else
    (void)failures;
    lewis_hal_shutdown();
#endif
}
