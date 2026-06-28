#include "ml/inference.h"
#include "ml/model_data.h"
#include "utils/debug.h"
#include "hal/hal.h"

#include <string.h>

#if LEWIS_USE_TFLM
#define TF_LITE_STRIP_ERROR_STRINGS
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#endif

#if LEWIS_USE_TFLM
/* Arena TFLM: tamanho configuravel via LEWIS_ARENA_SIZE (padrao 64 KB).
 * Como os dois modelos nunca rodam simultaneamente, uma unica arena e
 * compartilhada entre eles. */
#ifndef LEWIS_ARENA_SIZE
#define LEWIS_ARENA_SIZE (64 * 1024)
#endif
alignas(16) static uint8_t s_tensor_arena[LEWIS_ARENA_SIZE];

static tflite::MicroMutableOpResolver<15> s_resolver;
static bool s_resolver_initialized = false;
alignas(alignof(tflite::MicroInterpreter))
static uint8_t s_interpreter_storage[sizeof(tflite::MicroInterpreter)];
static tflite::MicroInterpreter* s_interpreter = nullptr;
static TfLiteTensor* s_input = nullptr;
static TfLiteTensor* s_output = nullptr;
static int s_current_stage = 0; /* 0=nenhum, 1=stage1, 2=stage2 */

static void ensure_resolver_initialized(void)
{
    if (s_resolver_initialized) {
        return;
    }
    s_resolver.AddConv2D();
    s_resolver.AddDepthwiseConv2D();
    s_resolver.AddFullyConnected();
    s_resolver.AddMaxPool2D();
    s_resolver.AddAveragePool2D();
    s_resolver.AddMean();
    s_resolver.AddSoftmax();
    s_resolver.AddReshape();
    s_resolver.AddExpandDims();
    s_resolver.AddQuantize();
    s_resolver.AddDequantize();
    s_resolver_initialized = true;
}

static void reset_arena(void)
{
    memset(s_tensor_arena, 0, sizeof(s_tensor_arena));
}

static void destroy_interpreter(void)
{
    if (s_interpreter != nullptr) {
        s_interpreter->~MicroInterpreter();
        s_interpreter = nullptr;
    }
    s_input = nullptr;
    s_output = nullptr;
    s_current_stage = 0;
}

static bool init_stage(const unsigned char* model_data,
                       unsigned int model_len,
                       int expected_input_bytes,
                       int expected_output_bytes,
                       int stage)
{
    if (s_current_stage == stage) {
        return true;
    }

    if (model_data == nullptr || model_len == 0U) {
        lewis_debug_print("[inference] ERRO: modelo nao disponivel\n");
        return false;
    }

    destroy_interpreter();
    reset_arena();

    const tflite::Model* model = tflite::GetModel(model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        lewis_debug_print("[inference] ERRO: schema version mismatch\n");
        return false;
    }

    ensure_resolver_initialized();

    s_interpreter = new (s_interpreter_storage)
        tflite::MicroInterpreter(model, s_resolver, s_tensor_arena, sizeof(s_tensor_arena));

    if (s_interpreter->AllocateTensors() != kTfLiteOk) {
        lewis_debug_print("[inference] ERRO: AllocateTensors falhou\n");
        destroy_interpreter();
        return false;
    }

    s_input = s_interpreter->input(0);
    s_output = s_interpreter->output(0);

    if (!s_input || !s_output ||
        s_input->type != kTfLiteInt8 || s_output->type != kTfLiteInt8 ||
        s_input->bytes != (size_t)expected_input_bytes ||
        s_output->bytes != (size_t)expected_output_bytes) {
        lewis_debug_print("[inference] ERRO: tensores de entrada/saida invalidos\n");
        destroy_interpreter();
        return false;
    }

    s_current_stage = stage;
    return true;
}

static bool run_stage(int stage,
                      const int8_t input[LEWIS_INPUT_LEN],
                      int8_t* output,
                      size_t output_len)
{
    if (!input || !output) {
        return false;
    }

    if (!s_interpreter || s_current_stage != stage || !s_input || !s_output) {
        lewis_debug_print("[inference] ERRO: interpretador nao inicializado\n");
        return false;
    }

    if (s_input->bytes != LEWIS_INPUT_LEN || s_output->bytes != output_len) {
        lewis_debug_print("[inference] ERRO: dimensoes dos tensores invalidas\n");
        return false;
    }

    memcpy(s_input->data.int8, input, LEWIS_INPUT_LEN);

    lewis_hal_watchdog_start(LEWIS_WATCHDOG_TIMEOUT_MS);
    TfLiteStatus status = s_interpreter->Invoke();
    lewis_hal_watchdog_stop();

    if (status != kTfLiteOk) {
        lewis_debug_print("[inference] ERRO: Invoke falhou\n");
        return false;
    }
    memcpy(output, s_output->data.int8, output_len);
    return true;
}
#endif /* LEWIS_USE_TFLM */

bool lewis_stage1_init(void)
{
#if LEWIS_USE_TFLM
    return init_stage(stage1_int8_v2_0_tflite,
                      stage1_int8_v2_0_len,
                      LEWIS_INPUT_LEN,
                      LEWIS_STAGE1_OUTPUT_LEN,
                      1);
#else
    static const char LEWIS_STUB_MARKER[] = "STUB_TFLM";
    (void)LEWIS_STUB_MARKER;
    lewis_debug_print("[inference] STUB: stage1 init ok (TFLM nao vinculado)\n");
    return true;
#endif
}

bool lewis_stage1_run(const int8_t input[LEWIS_INPUT_LEN],
                      int8_t output[LEWIS_STAGE1_OUTPUT_LEN])
{
#if LEWIS_USE_TFLM
    if (s_current_stage != 1 && !lewis_stage1_init()) {
        return false;
    }
    return run_stage(1, input, output, LEWIS_STAGE1_OUTPUT_LEN);
#else
    if (!input || !output) {
        return false;
    }
    memcpy(output, input, LEWIS_STAGE1_OUTPUT_LEN);
    return true;
#endif
}

bool lewis_stage2_init(void)
{
#if LEWIS_USE_TFLM
    return init_stage(stage2_int8_v2_0_tflite,
                      stage2_int8_v2_0_len,
                      LEWIS_INPUT_LEN,
                      LEWIS_STAGE2_OUTPUT_LEN,
                      2);
#else
    lewis_debug_print("[inference] STUB: stage2 init ok (TFLM nao vinculado)\n");
    return true;
#endif
}

bool lewis_stage2_run(const int8_t input[LEWIS_INPUT_LEN],
                      int8_t output[LEWIS_STAGE2_OUTPUT_LEN])
{
#if LEWIS_USE_TFLM
    if (s_current_stage != 2 && !lewis_stage2_init()) {
        return false;
    }
    return run_stage(2, input, output, LEWIS_STAGE2_OUTPUT_LEN);
#else
    if (!input || !output) {
        return false;
    }
    memcpy(output, input, LEWIS_STAGE2_OUTPUT_LEN);
    return true;
#endif
}

bool lewis_inference_init(void)
{
    return lewis_stage1_init();
}

bool lewis_inference_run(const int8_t input[LEWIS_INPUT_LEN],
                         int8_t output[LEWIS_OUTPUT_LEN])
{
    return lewis_stage1_run(input, output);
}

size_t lewis_inference_model_size(void)
{
#if LEWIS_USE_TFLM
    return (size_t)stage1_int8_v2_0_len + (size_t)stage2_int8_v2_0_len;
#else
    return 0;
#endif
}

size_t lewis_stage1_model_size(void)
{
#if LEWIS_USE_TFLM
    return (size_t)stage1_int8_v2_0_len;
#else
    return 0;
#endif
}

size_t lewis_stage2_model_size(void)
{
#if LEWIS_USE_TFLM
    return (size_t)stage2_int8_v2_0_len;
#else
    return 0;
#endif
}

size_t lewis_inference_arena_used(void)
{
#if LEWIS_USE_TFLM
    if (!s_interpreter) {
        return 0;
    }
    return s_interpreter->arena_used_bytes();
#else
    return 0;
#endif
}
