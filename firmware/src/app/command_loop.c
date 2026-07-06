#include "app/command_loop.h"
#include "dsp/adaptive_skipping.h"
#include "hal/hal.h"

#include <stddef.h>
#include <string.h>

static uint8_t s_buffer[LEWIS_CMD_MAX_LEN];
static size_t s_len;

static lewis_adaptive_skipping_t s_adaptive_skipping;
static uint32_t s_last_infer_ms;
static bool s_has_last_infer_ms;

#define ADAPTIVE_SKIP_THRESHOLD_MS 10U
#define ADAPTIVE_SKIP_THRESHOLD_RATIO 0.05f

void lewis_command_reset(void)
{
    s_len = 0;
}

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
        } else if (cmd_is("PEAK")) {
            result = LEWIS_CMD_PEAK;
        }
        s_len = 0;
        return result;
    }

    if (s_len < LEWIS_CMD_MAX_LEN - 1) {
        s_buffer[s_len++] = byte;
    }
    return LEWIS_CMD_NONE;
}

void lewis_command_adaptive_skipping_reset(void)
{
    lewis_adaptive_skipping_init(&s_adaptive_skipping);
    s_last_infer_ms = 0U;
    s_has_last_infer_ms = false;
}

bool lewis_command_adaptive_skipping_should_skip(
    uint32_t rr_ms,
    uint32_t* out_last_class
)
{
    bool skip = lewis_adaptive_skipping_should_skip(
        &s_adaptive_skipping,
        rr_ms,
        ADAPTIVE_SKIP_THRESHOLD_MS,
        ADAPTIVE_SKIP_THRESHOLD_RATIO
    );
    if (skip && out_last_class != NULL) {
        *out_last_class = s_adaptive_skipping.last_class;
    }
    return skip;
}

void lewis_command_adaptive_skipping_update_class(uint32_t class_id)
{
    lewis_adaptive_skipping_update_class(&s_adaptive_skipping, class_id);
}
