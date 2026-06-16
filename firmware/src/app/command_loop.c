#include "app/command_loop.h"

#include <stddef.h>
#include <string.h>

static uint8_t s_buffer[LEWIS_CMD_MAX_LEN];
static size_t s_len;

static bool cmd_is(const char* cmd)
{
    return s_len == strlen(cmd) && memcmp(s_buffer, cmd, s_len) == 0;
}

lewis_cmd_t lewis_command_feed(uint8_t byte)
{
    if (byte == '\r') {
        return LEWIS_CMD_NONE;
    }

    if (byte == '\n') {
        s_buffer[s_len] = '\0';
        lewis_cmd_t result = LEWIS_CMD_NONE;
        if (cmd_is("SHUTDOWN")) {
            result = LEWIS_CMD_SHUTDOWN;
        } else if (cmd_is("RUN")) {
            result = LEWIS_CMD_RUN;
        } else if (cmd_is("ECHO")) {
            result = LEWIS_CMD_ECHO;
        } else if (cmd_is("WATCHDOG")) {
            result = LEWIS_CMD_WATCHDOG;
        }
        s_len = 0;
        return result;
    }

    if (s_len < LEWIS_CMD_MAX_LEN - 1) {
        s_buffer[s_len++] = byte;
    }
    return LEWIS_CMD_NONE;
}

void lewis_command_reset(void)
{
    s_len = 0;
}
