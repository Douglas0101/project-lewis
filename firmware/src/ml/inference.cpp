#include "ml/inference.h"
#include "ml/model_data.h"
#include "utils/debug.h"

#include <string.h>

/* Alias para os nomes gerados pelo export_tflite.py. */
#define g_model_data     model_int8_tflite
#define g_model_data_len model_int8_len

#if LEWIS_USE_TFLM
#define TF_LITE_STRIP_ERROR_STRINGS
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#endif

#if LEWIS_USE_TFLM
/* Arena TFLM: tamanho configuravel via LEWIS_ARENA_SIZE (padrao 64 KB). */
#ifndef LEWIS_ARENA_SIZE
#define LEWIS_ARENA_SIZE (64 * 1024)
#endif
alignas(16) static uint8_t s_tensor_arena[LEWIS_ARENA_SIZE];

static const tflite::Model* s_model = nullptr;
static tflite::MicroMutableOpResolver<15> s_resolver;
alignas(alignof(tflite::MicroInterpreter))
static uint8_t s_interpreter_storage[sizeof(tflite::MicroInterpreter)];
static tflite::MicroInterpreter* s_interpreter = nullptr;
static TfLiteTensor* s_input = nullptr;
static TfLiteTensor* s_output = nullptr;
#endif

bool lewis_inference_init(void)
{
#if LEWIS_USE_TFLM
    s_model = tflite::GetModel(g_model_data);
    if (s_model->version() != TFLITE_SCHEMA_VERSION) {
        lewis_debug_print("[inference] ERRO: schema version mismatch\n");
        return false;
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

    s_interpreter = new (s_interpreter_storage)
        tflite::MicroInterpreter(s_model, s_resolver, s_tensor_arena, sizeof(s_tensor_arena));

    if (s_interpreter->AllocateTensors() != kTfLiteOk) {
        lewis_debug_print("[inference] ERRO: AllocateTensors falhou\n");
        return false;
    }

    s_input = s_interpreter->input(0);
    s_output = s_interpreter->output(0);

    if (!s_input || !s_output ||
        s_input->type != kTfLiteInt8 || s_output->type != kTfLiteInt8 ||
        s_input->bytes != LEWIS_INPUT_LEN || s_output->bytes != LEWIS_OUTPUT_LEN) {
        lewis_debug_print("[inference] ERRO: tensores de entrada/saida invalidos\n");
        return false;
    }
    return true;
#else
    /* Marcador intencional para deteccao de builds acidentais com stub. */
    static const char LEWIS_STUB_MARKER[] = "STUB_TFLM";
    (void)LEWIS_STUB_MARKER;
    lewis_debug_print("[inference] STUB: init ok (TFLM nao vinculado)\n");
    return true;
#endif
}

bool lewis_inference_run(const int8_t input[LEWIS_INPUT_LEN],
                         int8_t output[LEWIS_OUTPUT_LEN])
{
    if (!input || !output) {
        return false;
    }

#if LEWIS_USE_TFLM
    if (!s_interpreter || !s_input || !s_output) {
        lewis_debug_print("[inference] ERRO: interpretador nao inicializado\n");
        return false;
    }

    /* Revalida dimensoes dos tensores em runtime (defesa contra corrupcao). */
    if (s_input->bytes != LEWIS_INPUT_LEN || s_output->bytes != LEWIS_OUTPUT_LEN) {
        lewis_debug_print("[inference] ERRO: dimensoes dos tensores invalidas\n");
        return false;
    }

    memcpy(s_input->data.int8, input, LEWIS_INPUT_LEN);
    if (s_interpreter->Invoke() != kTfLiteOk) {
        lewis_debug_print("[inference] ERRO: Invoke falhou\n");
        return false;
    }
    memcpy(output, s_output->data.int8, LEWIS_OUTPUT_LEN);
    return true;
#else
    /* Stub: copia entrada para saida para permitir testes estruturais. */
    memcpy(output, input, LEWIS_OUTPUT_LEN);
    return true;
#endif
}

size_t lewis_inference_model_size(void)
{
    return g_model_data_len;
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
