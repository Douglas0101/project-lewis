/* Runner nativo leve para exportar a saida da cadeia de filtros C.
 *
 * Usado por tests/test_firmware_filters_python.py para comparar a saida dos
 * filtros DSP em C com a referencia Python ponto a ponto (QG16).
 *
 * Uso: filter_c_runner <input.bin> <output.bin>
 *   input.bin  -> 500 amostras float32 (sinal bruto a 500 Hz).
 *   output.bin -> 500 amostras float32 apos bandpass -> notch.
 */

#include <stdio.h>
#include <stdlib.h>

#include "dsp/filter.h"

#define SAMPLES 500u

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input.bin> <output.bin>\n", argv[0]);
        return 1;
    }

    FILE* fin = fopen(argv[1], "rb");
    if (!fin) {
        perror("fopen input");
        return 1;
    }

    float input[SAMPLES];
    size_t read = fread(input, sizeof(float), SAMPLES, fin);
    fclose(fin);

    if (read != SAMPLES) {
        fprintf(stderr, "Expected %u samples, got %zu\n", SAMPLES, read);
        return 1;
    }

    lewis_filter_chain_t chain;
    lewis_filter_chain_init(&chain);
    lewis_filter_chain_reset(&chain);

    float output[SAMPLES];
    lewis_filter_chain_process(&chain, input, output, SAMPLES);

    FILE* fout = fopen(argv[2], "wb");
    if (!fout) {
        perror("fopen output");
        return 1;
    }

    size_t written = fwrite(output, sizeof(float), SAMPLES, fout);
    fclose(fout);

    if (written != SAMPLES) {
        fprintf(stderr, "Failed to write all samples (wrote %zu)\n", written);
        return 1;
    }

    return 0;
}
