#ifndef LEWIS_COMMAND_LOOP_H
#define LEWIS_COMMAND_LOOP_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define LEWIS_CMD_MAX_LEN 64

typedef enum {
    LEWIS_CMD_NONE,
    LEWIS_CMD_RUN,
    LEWIS_CMD_SHUTDOWN,
    LEWIS_CMD_ECHO,
    LEWIS_CMD_WATCHDOG,
    LEWIS_CMD_PEAK,
} lewis_cmd_t;

lewis_cmd_t lewis_command_feed(uint8_t byte);
void lewis_command_reset(void);

/**
 * @brief Reinicia o estado de adaptive skipping do loop de comandos.
 */
void lewis_command_adaptive_skipping_reset(void);

/**
 * @brief Decide, com base no RR fornecido, se a inferencia deve ser pulada.
 *
 * @param rr_ms          Intervalo RR atual em milissegundos.
 * @param out_last_class Saida: ultima classe conhecida (valida apenas se
 *                       retornar true).
 * @return true se a inferencia deve ser pulada; false caso contrario.
 */
bool lewis_command_adaptive_skipping_should_skip(
    uint32_t rr_ms,
    uint32_t* out_last_class
);

/**
 * @brief Atualiza a ultima classe conhecida pelo adaptive skipping.
 *
 * @param class_id Classe predita pela ultima inferencia real.
 */
void lewis_command_adaptive_skipping_update_class(uint32_t class_id);

#ifdef __cplusplus
}
#endif

#endif /* LEWIS_COMMAND_LOOP_H */
