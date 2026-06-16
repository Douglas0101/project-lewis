"""Testes unitarios para o runner de simulacao Renode.

Validam o parser de log UART e a funcao de check do relatorio independentemente
 da execucao do emulador.
"""

from __future__ import annotations

import pytest

from firmware.scripts.run_renode_tests import check_report, parse_uart_log


SAMPLE_LOG = """\
=== Project-Lewis Firmware v1.2 ===
Model size: 25240 bytes
Inference init OK
Arena used: 19972 bytes
Beat 0: 16 ms (16181 us), output [-79, -67, -79, -82, -77]
Beat 1: 16 ms (16181 us), output [-79, -67, -79, -82, -77]
Beat 2: 16 ms (16181 us), output [-79, -67, -79, -82, -77]
=== Fim ===
"""

LOG_NO_ARENA = """\
=== Project-Lewis Firmware v1.0 ===
Model size: 1024 bytes
Inference init OK
Beat 0: 10 ms, output [1, 2, 3]
Beat 1: 10 ms, output [4, 5, 6]
=== Fim ===
"""

LOG_EXTRA_SPACES = """\
===  Project-Lewis   Firmware  v2.0  ===
Model  size :   2048   bytes
Inference   init   OK
Arena  used :   8192   bytes
Beat   0 :   20   ms   ( 20001 us ) ,   output   [ -1 , -2 , -3 ]
Beat   1 :   20   ms ,   output   [ -4 , -5 , -6 ]
===  Fim  ===
"""

LOG_MISSING_END = """\
=== Project-Lewis Firmware v1.0 ===
Model size: 512 bytes
Inference init OK
Beat 0: 5 ms, output [0]
"""


@pytest.mark.qg9
class TestParseUartLog:
    def test_parses_complete_log(self):
        parsed = parse_uart_log(SAMPLE_LOG)
        assert parsed["header"] is True
        assert parsed["firmware_version"] == "1.2"
        assert parsed["model_size_bytes"] == 25240
        assert parsed["inference_init"] is True
        assert parsed["arena_used_bytes"] == 19972
        assert parsed["end"] is True
        assert len(parsed["beats"]) == 3
        assert parsed["beats"][0] == {
            "index": 0,
            "time_ms": 16,
            "output": [-79, -67, -79, -82, -77],
        }

    def test_parses_log_without_arena(self):
        parsed = parse_uart_log(LOG_NO_ARENA)
        assert parsed["model_size_bytes"] == 1024
        assert parsed["arena_used_bytes"] is None
        assert len(parsed["beats"]) == 2
        assert parsed["beats"][1]["output"] == [4, 5, 6]

    def test_tolerates_extra_spaces_and_missing_us(self):
        parsed = parse_uart_log(LOG_EXTRA_SPACES)
        assert parsed["firmware_version"] == "2.0"
        assert parsed["model_size_bytes"] == 2048
        assert parsed["arena_used_bytes"] == 8192
        assert len(parsed["beats"]) == 2
        assert parsed["beats"][0]["time_ms"] == 20
        assert parsed["beats"][1]["time_ms"] == 20

    def test_detects_missing_end(self):
        parsed = parse_uart_log(LOG_MISSING_END)
        assert parsed["header"] is True
        assert parsed["end"] is False
        assert len(parsed["beats"]) == 1


@pytest.mark.qg9
class TestCheckReport:
    def test_all_passed_for_complete_log(self):
        parsed = parse_uart_log(SAMPLE_LOG)
        checks = check_report(parsed, expected_beats=3)
        assert checks["all_passed"] is True
        assert checks["header"] is True
        assert checks["model_size"] is True
        assert checks["inference_init"] is True
        assert checks["beats"] is True
        assert checks["end"] is True

    def test_fails_when_expected_beats_not_met(self):
        parsed = parse_uart_log(LOG_NO_ARENA)
        checks = check_report(parsed, expected_beats=5)
        assert checks["all_passed"] is False
        assert checks["beats"] is False

    def test_model_size_over_limit_fails(self):
        log = (
            "=== Project-Lewis Firmware v1.0 ===\n"
            "Model size: 66000 bytes\n"
            "Inference init OK\n"
            "=== Fim ===\n"
        )
        parsed = parse_uart_log(log)
        checks = check_report(parsed)
        assert checks["model_size"] is False
        assert checks["all_passed"] is False
