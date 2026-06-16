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
} lewis_cmd_t;

lewis_cmd_t lewis_command_feed(uint8_t byte);
void lewis_command_reset(void);

#ifdef __cplusplus
}
#endif

#endif
