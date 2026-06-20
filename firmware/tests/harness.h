#ifndef LEWIS_TEST_HARNESS_H
#define LEWIS_TEST_HARNESS_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define HARNESS_MAX_NAME_LEN 64

typedef struct {
    char suite[HARNESS_MAX_NAME_LEN];
    char name[HARNESS_MAX_NAME_LEN];
    bool passed;
    char detail[128];
} harness_result_t;

typedef void (*harness_test_fn_t)(harness_result_t* result);

void harness_print(const char* fmt, ...);
void harness_register(const char* suite, const char* name, harness_test_fn_t fn);
void harness_run_all(void);
int  harness_summary(void); /* retorna numero de falhas */

void harness_assert_true(harness_result_t* r, bool cond, const char* msg);
void harness_assert_int_eq(harness_result_t* r, int64_t expected, int64_t actual, const char* msg);
void harness_assert_int_close(harness_result_t* r, int64_t expected, int64_t actual, int64_t tol, const char* msg);
void harness_assert_float_close(harness_result_t* r, float expected, float actual, float tol, const char* msg);
void harness_assert_int8_eq(harness_result_t* r, int8_t expected, int8_t actual, const char* msg);

/* Suites exportadas */
void suite_dsp_register(void);
void suite_r_peak_register(void);
void suite_inference_register(void);
#if LEWIS_USE_TFLM
void suite_pipeline_register(void);
#endif

#ifdef __cplusplus
}
#endif

#endif /* LEWIS_TEST_HARNESS_H */
