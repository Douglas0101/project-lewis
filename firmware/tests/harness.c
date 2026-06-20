#include "harness.h"
#include "utils/debug.h"
#include <string.h>
#include <stdio.h>
#include <stdarg.h>

#define MAX_TESTS 32

static struct {
    const char* suite;
    const char* name;
    harness_test_fn_t fn;
} g_tests[MAX_TESTS];

static size_t g_num_tests = 0;

void harness_print(const char* fmt, ...) {
    char buf[256];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    lewis_debug_print(buf);
}

void harness_register(const char* suite, const char* name, harness_test_fn_t fn) {
    if (g_num_tests >= MAX_TESTS) {
        harness_print("HARNESS ERROR too many tests\r\n");
        return;
    }
    g_tests[g_num_tests].suite = suite;
    g_tests[g_num_tests].name = name;
    g_tests[g_num_tests].fn = fn;
    g_num_tests++;
}

void harness_assert_true(harness_result_t* r, bool cond, const char* msg) {
    if (!cond && r->passed) {
        r->passed = false;
        strncpy(r->detail, msg, sizeof(r->detail) - 1);
        r->detail[sizeof(r->detail) - 1] = '\0';
    }
}

void harness_assert_int_eq(harness_result_t* r, int64_t expected, int64_t actual, const char* msg) {
    if (expected != actual && r->passed) {
        r->passed = false;
        snprintf(r->detail, sizeof(r->detail), "%s: expected %lld got %lld", msg, (long long)expected, (long long)actual);
    }
}

void harness_assert_int_close(harness_result_t* r, int64_t expected, int64_t actual, int64_t tol, const char* msg) {
    int64_t diff = actual - expected;
    if (diff < 0) diff = -diff;
    if (diff > tol && r->passed) {
        r->passed = false;
        snprintf(r->detail, sizeof(r->detail), "%s: expected %lld+/-%lld got %lld", msg, (long long)expected, (long long)tol, (long long)actual);
    }
}

void harness_assert_float_close(harness_result_t* r, float expected, float actual, float tol, const char* msg) {
    float diff = actual - expected;
    if (diff < 0.0f) diff = -diff;
    if (diff > tol && r->passed) {
        r->passed = false;
        snprintf(r->detail, sizeof(r->detail), "%s: expected %.6f got %.6f", msg, (double)expected, (double)actual);
    }
}

void harness_assert_int8_eq(harness_result_t* r, int8_t expected, int8_t actual, const char* msg) {
    if (expected != actual && r->passed) {
        r->passed = false;
        snprintf(r->detail, sizeof(r->detail), "%s: expected %d got %d", msg, (int)expected, (int)actual);
    }
}

void harness_run_all(void) {
    harness_result_t result;
    size_t passed = 0;
    size_t failed = 0;

    harness_print("HARNESS START v1.0\r\n");
    for (size_t i = 0; i < g_num_tests; i++) {
        memset(&result, 0, sizeof(result));
        result.passed = true;
        strncpy(result.suite, g_tests[i].suite, sizeof(result.suite) - 1);
        strncpy(result.name, g_tests[i].name, sizeof(result.name) - 1);

        g_tests[i].fn(&result);

        if (result.passed) {
            passed++;
            harness_print("HARNESS %s %s PASS\r\n", result.suite, result.name);
        } else {
            failed++;
            harness_print("HARNESS %s %s FAIL %s\r\n", result.suite, result.name, result.detail);
        }
    }
    harness_print("HARNESS SUMMARY PASS %zu FAIL %zu TOTAL %zu\r\n", passed, failed, g_num_tests);
    harness_print("HARNESS END\r\n");
}

int harness_summary(void) {
    /* Recalcula rapidamente; usado pelo main apos run_all */
    int failures = 0;
    harness_result_t result;
    for (size_t i = 0; i < g_num_tests; i++) {
        memset(&result, 0, sizeof(result));
        result.passed = true;
        g_tests[i].fn(&result);
        if (!result.passed) failures++;
    }
    return failures;
}
