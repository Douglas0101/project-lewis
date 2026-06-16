#include "hal/hal.h"
#include "utils/debug.h"
#include "ml/inference.h"
#include "ml/quantization_params.h"
#include "dsp/adc_stub.h"
#include "dsp/filter.h"
#include "app/command_loop.h"

#include <stdint.h>
#include <stddef.h>

#define NUM_TEST_BEATS 3

/* Buffers de inferencia alocados estaticamente para evitar consumo de stack.
 * O modelo atual opera com 500 amostras de entrada e 5 saidas int8. */
static int8_t s_raw_input[LEWIS_INPUT_LEN];
static float s_float_input[LEWIS_INPUT_LEN];
static int8_t s_input[LEWIS_INPUT_LEN];
static int8_t s_output[LEWIS_OUTPUT_LEN];

static lewis_filter_chain_t s_filter_chain;

static void dequantize_beat(const int8_t* quantized, float* float_out, size_t len)
{
    const float scale = LEWIS_QUANTIZATION_PARAMS_INPUT_SCALE;
    const int32_t zp = LEWIS_QUANTIZATION_PARAMS_INPUT_ZERO_POINT;
    for (size_t i = 0; i < len; ++i) {
        float_out[i] = ((float)quantized[i] - (float)zp) * scale;
    }
}

static void quantize_beat(const float* float_in, int8_t* quantized, size_t len)
{
    const float scale = LEWIS_QUANTIZATION_PARAMS_INPUT_SCALE;
    const int32_t zp = LEWIS_QUANTIZATION_PARAMS_INPUT_ZERO_POINT;
    for (size_t i = 0; i < len; ++i) {
        float normalized = float_in[i] / scale;
        int32_t q = (int32_t)(normalized + (normalized >= 0.0f ? 0.5f : -0.5f)) + zp;
        if (q > 127) {
            q = 127;
        } else if (q < -128) {
            q = -128;
        }
        quantized[i] = (int8_t)q;
    }
}

static void apply_dsp_pipeline(const int8_t* raw_input, int8_t* quantized_out, size_t len)
{
    dequantize_beat(raw_input, s_float_input, len);
    lewis_filter_chain_reset(&s_filter_chain);
    lewis_filter_chain_process(&s_filter_chain, s_float_input, s_float_input, len);
    quantize_beat(s_float_input, quantized_out, len);
}

static void print_report(void)
{
    lewis_debug_print("Model size: ");
    lewis_debug_print_uint((uint32_t)lewis_inference_model_size());
    lewis_debug_print(" bytes\n");
    lewis_debug_print("Arena used: ");
    lewis_debug_print_uint((uint32_t)lewis_inference_arena_used());
    lewis_debug_print(" bytes\n");
}

static void run_demo_beats(void)
{
    for (uint32_t beat = 0; beat < NUM_TEST_BEATS; ++beat) {
        lewis_adc_stub_get_beat(beat, s_raw_input);
        apply_dsp_pipeline(s_raw_input, s_input, LEWIS_INPUT_LEN);

        uint32_t t0 = lewis_hal_benchmark_start();
        if (!lewis_inference_run(s_input, s_output)) {
            lewis_debug_print("[main] RUN FAIL\n");
            lewis_hal_panic();
        }
        uint32_t t1 = lewis_hal_benchmark_stop();
        uint32_t cycles = (t0 - t1) & 0x00FFFFFFUL;
        uint32_t us = lewis_hal_benchmark_delta_us(cycles);
        uint32_t ms = us / 1000U;

        lewis_debug_print("Beat ");
        lewis_debug_print_uint(beat);
        lewis_debug_print(": ");
        lewis_debug_print_uint(ms);
        lewis_debug_print(" ms (");
        lewis_debug_print_uint(us);
        lewis_debug_print(" us), output [");
        for (int i = 0; i < LEWIS_OUTPUT_LEN; ++i) {
            lewis_debug_print_int((int32_t)s_output[i]);
            if (i + 1 < LEWIS_OUTPUT_LEN) {
                lewis_debug_print(", ");
            }
        }
        lewis_debug_print("]\n");
    }
}

#if RENODE_SIMULATION

static uint8_t uart_read_byte(void)
{
    while (!lewis_hal_uart_rx_ready()) {
        /* aguarda byte */
    }
    return lewis_hal_uart_rx();
}

#define UART_BYTE_TIMEOUT_MS 100
#define UART_FRAME_TIMEOUT_MS 5000

static bool uart_read_byte_timeout(uint8_t *out, uint32_t timeout_ms)
{
    uint32_t start = lewis_hal_millis();
    while (!lewis_hal_uart_rx_ready()) {
        if ((lewis_hal_millis() - start) >= timeout_ms) {
            return false;
        }
    }
    *out = lewis_hal_uart_rx();
    return true;
}

static void infer_from_uart(uint8_t start_byte)
{
    static int8_t input_quantized[LEWIS_INPUT_LEN];
    static int8_t output[LEWIS_OUTPUT_LEN];
    static float frame[LEWIS_INPUT_LEN];
    uint8_t byte;

    if (start_byte != '<') {
        lewis_debug_print("[infer] FRAME ERR\n");
        return;
    }

    /* Le 500 floats little-endian (2000 bytes) apos '<'. */
    uint32_t start = lewis_hal_millis();
    for (int i = 0; i < LEWIS_INPUT_LEN; ++i) {
        uint32_t u = 0;
        for (int b = 0; b < 4; ++b) {
            uint32_t elapsed = lewis_hal_millis() - start;
            uint32_t remaining = (elapsed >= UART_FRAME_TIMEOUT_MS) ? 0
                                    : (UART_FRAME_TIMEOUT_MS - elapsed);
            uint32_t timeout = (remaining < UART_BYTE_TIMEOUT_MS) ? remaining
                                    : UART_BYTE_TIMEOUT_MS;
            if (!uart_read_byte_timeout(&byte, timeout)) {
                lewis_debug_print("[uart] ERRO: timeout\n");
                return;
            }
            u |= (uint32_t)byte << (8 * b);
        }
        /* Conversao bit-exata uint32_t -> float sem violar aliasing. */
        union {
            uint32_t u;
            float f;
        } conv;
        conv.u = u;
        frame[i] = conv.f;
    }

    /* Verifica terminador '>'. */
    if (!uart_read_byte_timeout(&byte, UART_BYTE_TIMEOUT_MS)) {
        lewis_debug_print("[uart] ERRO: timeout\n");
        return;
    }
    if (byte != '>') {
        lewis_debug_print("[infer] FRAME ERR\n");
        return;
    }

    /* Aplica filtros bandpass/notch no dominio float32. */
    lewis_filter_chain_reset(&s_filter_chain);
    lewis_filter_chain_process(&s_filter_chain, frame, frame, LEWIS_INPUT_LEN);

    /* Quantiza float32 -> int8 usando parametros do modelo. */
    quantize_beat(frame, input_quantized, LEWIS_INPUT_LEN);

    /* Executa inferencia. */
    if (!lewis_inference_run(input_quantized, output)) {
        lewis_debug_print("[infer] RUN FAIL\n");
        return;
    }

    /* Responde com '<' + 5 int8 + '>'. */
    lewis_hal_uart_tx('<');
    for (int i = 0; i < LEWIS_OUTPUT_LEN; ++i) {
        lewis_hal_uart_tx((uint8_t)output[i]);
    }
    lewis_hal_uart_tx('>');
}

static void command_loop(void)
{
    lewis_debug_print("Modo comando UART ativo\n");
    lewis_command_reset();

    for (;;) {
        uint8_t byte = uart_read_byte();

        if (byte == '<') {
            infer_from_uart(byte);
            lewis_command_reset();
            continue;
        }

        lewis_cmd_t cmd = lewis_command_feed(byte);
        switch (cmd) {
        case LEWIS_CMD_ECHO:
            lewis_debug_print("ECHO\n");
            lewis_command_reset();
            break;
        case LEWIS_CMD_RUN:
            lewis_debug_print("[RUN]\n");
            run_demo_beats();
            lewis_command_reset();
            break;
        case LEWIS_CMD_SHUTDOWN:
            lewis_debug_print("[SHUTDOWN]\n");
            print_report();
            lewis_debug_print("=== Fim ===\n");
            lewis_hal_shutdown();
            break;
        case LEWIS_CMD_WATCHDOG:
            lewis_debug_print("[WATCHDOG TEST]\n");
            lewis_hal_watchdog_start(LEWIS_WATCHDOG_TIMEOUT_MS);
            while (1) {
                /* Simula travamento: loop infinito ate o watchdog reiniciar. */
                __asm__ volatile("wfi");
            }
            break;
        default:
            break;
        }
    }
}

#endif /* RENODE_SIMULATION */

int main(void)
{
    lewis_hal_init();
    lewis_debug_init();

    lewis_debug_print("=== Project-Lewis Firmware v1.2 ===\n");
    print_report();

    if (!lewis_inference_init()) {
        lewis_debug_print("[main] INIT FAIL\n");
        lewis_hal_panic();
    }
    lewis_debug_print("Inference init OK\n");

    lewis_filter_chain_init(&s_filter_chain);
    lewis_debug_print("DSP filters init OK\n");

    run_demo_beats();

#if RENODE_SIMULATION
    command_loop();
#else
    lewis_debug_print("=== Fim ===\n");
    lewis_hal_shutdown();
#endif

    return 0;
}
