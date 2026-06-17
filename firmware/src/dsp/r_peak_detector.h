/*
 * Detector leve de R-peaks para ECG @ 500 Hz (ou frequencia configuravel).
 *
 * Implementacao simplificada baseada em Pan-Tompkins / AMPT:
 *   - Derivada de 5 pontos
 *   - Quadrado ponto-a-ponto
 *   - Integracao por janela movel (150 ms)
 *   - Threshold adaptativo simples + periodo refratario (360 ms)
 *   - Discriminacao de onda T por inclinacao local
 *
 * Nao usa allocacao dinamica; trabalha com arrays na stack (VLA) ate
 * LEWIS_RPEAK_MAX_SAMPLES amostras.
 */

#ifndef LEWIS_R_PEAK_DETECTOR_H
#define LEWIS_R_PEAK_DETECTOR_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Limite de amostras para evitar estouro de pilha no embarcado. */
#define LEWIS_RPEAK_MAX_SAMPLES 16384U

/* Numero maximo de picos retornados. */
#define LEWIS_RPEAK_MAX_PEAKS 64U

/**
 * @brief Detecta picos R em um sinal ECG filtrado.
 *
 * @param sig   Sinal de entrada (1-D), ja filtrado (ex: bandpass 0.5-40 Hz).
 * @param len   Numero de amostras.
 * @param fs    Frequencia de amostragem em Hz.
 * @param peaks Array de saida com os indices dos picos detectados.
 * @param n_peaks Saida: numero de picos escritos em peaks.
 *
 * @return 0 em caso de sucesso; < 0 em erro (-1 parametros invalidos,
 *         -2 sinal muito longo).
 */
int lewis_detect_r_peaks(
    const float* sig,
    size_t len,
    float fs,
    size_t* peaks,
    size_t* n_peaks
);

#ifdef __cplusplus
}
#endif

#endif /* LEWIS_R_PEAK_DETECTOR_H */
