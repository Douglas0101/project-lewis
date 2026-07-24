#include "harness.h"
#include "../src/dsp/adaptive_skipping.h"
#include <string.h>

#define THRESHOLD_MS 10U
#define THRESHOLD_RATIO 0.05f

static void test_init_clears_state(harness_result_t* r) {
    lewis_adaptive_skipping_t ctx;
    lewis_adaptive_skipping_init(&ctx);

    harness_assert_true(r, ctx.count == 0, "count zero after init");
    harness_assert_true(r, !ctx.has_last_class, "no last class after init");
}

static void test_less_than_three_cycles_never_skips(harness_result_t* r) {
    lewis_adaptive_skipping_t ctx;
    lewis_adaptive_skipping_init(&ctx);
    lewis_adaptive_skipping_update_class(&ctx, 0U);

    bool skip = lewis_adaptive_skipping_should_skip(&ctx, 800U, THRESHOLD_MS, THRESHOLD_RATIO);
    harness_assert_true(r, !skip, "first cycle never skips");

    skip = lewis_adaptive_skipping_should_skip(&ctx, 800U, THRESHOLD_MS, THRESHOLD_RATIO);
    harness_assert_true(r, !skip, "second cycle never skips");
}

static void test_stable_rhythm_triggers_skip(harness_result_t* r) {
    lewis_adaptive_skipping_t ctx;
    lewis_adaptive_skipping_init(&ctx);
    lewis_adaptive_skipping_update_class(&ctx, 2U);

    (void)lewis_adaptive_skipping_should_skip(&ctx, 800U, THRESHOLD_MS, THRESHOLD_RATIO);
    (void)lewis_adaptive_skipping_should_skip(&ctx, 800U, THRESHOLD_MS, THRESHOLD_RATIO);
    bool skip = lewis_adaptive_skipping_should_skip(&ctx, 800U, THRESHOLD_MS, THRESHOLD_RATIO);

#if ADAPTIVE_SKIPPING_ENABLED
    harness_assert_true(r, skip, "stable rhythm should skip when enabled");
#else
    harness_assert_true(r, !skip, "stable rhythm should not skip when disabled");
#endif
}

static void test_unstable_rhythm_does_not_skip(harness_result_t* r) {
    lewis_adaptive_skipping_t ctx;
    lewis_adaptive_skipping_init(&ctx);
    lewis_adaptive_skipping_update_class(&ctx, 2U);

    (void)lewis_adaptive_skipping_should_skip(&ctx, 600U, THRESHOLD_MS, THRESHOLD_RATIO);
    (void)lewis_adaptive_skipping_should_skip(&ctx, 800U, THRESHOLD_MS, THRESHOLD_RATIO);
    bool skip = lewis_adaptive_skipping_should_skip(&ctx, 1000U, THRESHOLD_MS, THRESHOLD_RATIO);

    harness_assert_true(r, !skip, "unstable rhythm should not skip");
}

static void test_window_does_not_exceed_five(harness_result_t* r) {
    lewis_adaptive_skipping_t ctx;
    lewis_adaptive_skipping_init(&ctx);
    lewis_adaptive_skipping_update_class(&ctx, 0U);

    const uint32_t rr_values[] = {700U, 800U, 900U, 850U, 825U, 800U, 790U};
    const size_t n = sizeof(rr_values) / sizeof(rr_values[0]);
    for (size_t i = 0; i < n; ++i) {
        (void)lewis_adaptive_skipping_should_skip(&ctx, rr_values[i], THRESHOLD_MS, THRESHOLD_RATIO);
    }

    harness_assert_int_eq(r, (int64_t)LEWIS_ADAPTIVE_SKIPPING_MAX_WINDOW, (int64_t)ctx.count, "window capped at 5");
}

static void test_no_last_class_never_skips(harness_result_t* r) {
    lewis_adaptive_skipping_t ctx;
    lewis_adaptive_skipping_init(&ctx);

    (void)lewis_adaptive_skipping_should_skip(&ctx, 800U, THRESHOLD_MS, THRESHOLD_RATIO);
    (void)lewis_adaptive_skipping_should_skip(&ctx, 800U, THRESHOLD_MS, THRESHOLD_RATIO);
    bool skip = lewis_adaptive_skipping_should_skip(&ctx, 800U, THRESHOLD_MS, THRESHOLD_RATIO);

    harness_assert_true(r, !skip, "no last class means no skip");
}

static void test_disabled_flag_never_skips(harness_result_t* r) {
    lewis_adaptive_skipping_t ctx;
    lewis_adaptive_skipping_init(&ctx);
    lewis_adaptive_skipping_update_class(&ctx, 0U);

    (void)lewis_adaptive_skipping_should_skip(&ctx, 800U, THRESHOLD_MS, THRESHOLD_RATIO);
    (void)lewis_adaptive_skipping_should_skip(&ctx, 800U, THRESHOLD_MS, THRESHOLD_RATIO);

#if ADAPTIVE_SKIPPING_ENABLED
    bool skip = lewis_adaptive_skipping_should_skip(&ctx, 800U, THRESHOLD_MS, THRESHOLD_RATIO);
    harness_assert_true(r, skip, "enabled flag allows skip");
#else
    bool skip = lewis_adaptive_skipping_should_skip(&ctx, 800U, THRESHOLD_MS, THRESHOLD_RATIO);
    harness_assert_true(r, !skip, "disabled flag prevents skip");
#endif
}

void suite_adaptive_skipping_register(void) {
    harness_register("ADAPTIVE_SKIPPING", "init_clears_state", test_init_clears_state);
    harness_register("ADAPTIVE_SKIPPING", "less_than_three_cycles", test_less_than_three_cycles_never_skips);
    harness_register("ADAPTIVE_SKIPPING", "stable_rhythm", test_stable_rhythm_triggers_skip);
    harness_register("ADAPTIVE_SKIPPING", "unstable_rhythm", test_unstable_rhythm_does_not_skip);
    harness_register("ADAPTIVE_SKIPPING", "window_cap", test_window_does_not_exceed_five);
    harness_register("ADAPTIVE_SKIPPING", "no_last_class", test_no_last_class_never_skips);
    harness_register("ADAPTIVE_SKIPPING", "enabled_flag", test_disabled_flag_never_skips);
}
