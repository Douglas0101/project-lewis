#ifndef LEWIS_ADAPTIVE_SKIPPING_H
#define LEWIS_ADAPTIVE_SKIPPING_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Habilita o adaptive inference skipping por padrao.
 *
 * Pode ser sobrescrito na linha de comando do compilador
 * (-DADAPTIVE_SKIPPING_ENABLED=0) para desabilitar o recurso sem remover o
 * codigo.
 */
#ifndef ADAPTIVE_SKIPPING_ENABLED
#define ADAPTIVE_SKIPPING_ENABLED 1
#endif

/**
 * @brief Tamanho maximo da janela de historico de intervalos RR.
 *
 * Limitado a 5 ciclos para evitar latencia excessiva na decisao de skipping.
 */
#define LEWIS_ADAPTIVE_SKIPPING_MAX_WINDOW 5U

/**
 * @brief Numero minimo de ciclos RR necessarios para confiar na estabilidade.
 */
#define LEWIS_ADAPTIVE_SKIPPING_MIN_CYCLES 3U

/**
 * @brief Estado do adaptive inference skipping.
 *
 * Todos os campos sao escalares ou arrays estaticos, sem alocacao dinamica,
 * para compatibilidade com bare-metal Cortex-M4F.
 */
typedef struct {
    uint32_t rr_history[LEWIS_ADAPTIVE_SKIPPING_MAX_WINDOW];
    uint8_t count;
    uint8_t head;
    uint32_t last_class;
    bool has_last_class;
} lewis_adaptive_skipping_t;

/**
 * @brief Inicializa o estado do adaptive skipping.
 *
 * @param ctx Ponteiro para o contexto. Deve ser valido.
 */
void lewis_adaptive_skipping_init(lewis_adaptive_skipping_t* ctx);

/**
 * @brief Alimenta um novo intervalo RR e decide se a inferencia deve ser
 *        pulada.
 *
 * O skipping ocorre quando ADAPTIVE_SKIPPING_ENABLED estiver ativo, houver
 * pelo menos LEWIS_ADAPTIVE_SKIPPING_MIN_CYCLES no historico, ja existir uma
 * classe anterior conhecida e a variacao absoluta e relativa dos ciclos
 * recentes estiver abaixo dos limites configurados.
 *
 * @param ctx             Contexto do adaptive skipping.
 * @param rr_ms           Intervalo RR atual em milissegundos.
 * @param threshold_ms    Limiar de variacao absoluta (ms).
 * @param threshold_ratio Limiar de variacao relativa (variacao / media).
 * @return true se a inferencia atual deve ser pulada; false caso contrario.
 */
bool lewis_adaptive_skipping_should_skip(
    lewis_adaptive_skipping_t* ctx,
    uint32_t rr_ms,
    uint32_t threshold_ms,
    float threshold_ratio
);

/**
 * @brief Atualiza a ultima classe conhecida.
 *
 * Deve ser chamada apos uma inferencia real, mesmo quando o skipping esta
 * ativo.
 *
 * @param ctx      Contexto do adaptive skipping.
 * @param class_id Identificador da classe predita.
 */
void lewis_adaptive_skipping_update_class(
    lewis_adaptive_skipping_t* ctx,
    uint32_t class_id
);

#ifdef __cplusplus
}
#endif

#endif /* LEWIS_ADAPTIVE_SKIPPING_H */
