"""Robot Framework library para envio de frames binarios pela UART do Renode.

Complementa ``fidelity.robot`` (QG10) e ``fault_injection.robot`` (QG11)
fornecendo keywords que transmitem frames byte a byte via
``sysbus.uart4 WriteChar``.
"""

from __future__ import annotations

import time
from pathlib import Path

from robot.libraries.BuiltIn import BuiltIn

INPUT_SAMPLES = 500
INPUT_BYTES = INPUT_SAMPLES * 4
START_BYTE = 0x3C  # '<'
END_BYTE = 0x3E  # '>'
BAD_END_BYTE = 0x21  # '!' (terminador invalido para injecao de falha)
RESPONSE_LEN = 3
START_DELAY_S = 0.005  # cede CPU sem consumir o timeout virtual por byte


class FidelityKeywords:
    """Keywords auxiliares para o teste de fidelidade QG10."""

    ROBOT_LIBRARY_SCOPE = "TEST"

    def __init__(self) -> None:
        self._builtin = BuiltIn()

    def send_binary_frame(self, path: str) -> None:
        """Envia '<' + conteudo binario de ``path`` + '>' pela UART4.

        Cada byte eh transmitido via comando monitor ``sysbus.uart4 WriteChar``,
        que eh a forma compativel com Renode 1.15.3 para injetar dados na UART
        do STM32F4 emulado.
        """
        file_path = Path(path)
        if not file_path.exists():
            raise RuntimeError(f"Arquivo de frame binario nao encontrado: {path}")

        data = file_path.read_bytes()
        if len(data) != INPUT_BYTES:
            raise RuntimeError(
                f"Tamanho invalido do frame binario: {len(data)} bytes "
                f"(esperado {INPUT_BYTES} bytes)"
            )

        # Inicio de frame: pausa para o firmware detectar '<' e entrar no handler
        self._builtin.run_keyword("Execute Command", f"sysbus.uart4 WriteChar {START_BYTE}")
        time.sleep(START_DELAY_S)

        # Execute Command e sincronizado pelo monitor do Renode. Sleeps reais
        # entre bytes deixam o tempo virtual avancar sem dados e tornam o gate
        # dependente da carga do runner, podendo disparar UART_BYTE_TIMEOUT_MS.
        for byte in data:
            self._builtin.run_keyword("Execute Command", f"sysbus.uart4 WriteChar {byte}")

        # Fim de frame
        self._builtin.run_keyword("Execute Command", f"sysbus.uart4 WriteChar {END_BYTE}")

    def send_corrupted_frame(self) -> None:
        """Envia '<' + 2000 bytes de preenchimento + terminador invalido.

        Usado pelo QG11 (fault_injection.robot): o firmware consome os 2000
        bytes esperados, rejeita o terminador e reporta '[infer] FRAME ERR'
        imediatamente. Por nao depender de timeout por byte, o teste fica
        imune a stalls do host no runner.
        """
        self._builtin.run_keyword("Execute Command", f"sysbus.uart4 WriteChar {START_BYTE}")
        time.sleep(START_DELAY_S)
        for _ in range(INPUT_BYTES):
            self._builtin.run_keyword("Execute Command", "sysbus.uart4 WriteChar 0")
        self._builtin.run_keyword("Execute Command", f"sysbus.uart4 WriteChar {BAD_END_BYTE}")

    def wait_for_response_in_log(self, log_path: str, timeout: float = 180.0) -> None:
        """Aguarda ate que ``log_path`` contenha uma resposta ``<5xint8>``.

        A saida do firmware e escrita no arquivo de backend da UART. Este
        metodo polling evita que o teste termine antes da resposta ser
        efetivamente gravada no disco.
        """
        log_file = Path(log_path)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not log_file.exists():
                time.sleep(0.05)
                continue
            data = log_file.read_bytes()
            for start in range(len(data) - 1, -1, -1):
                if data[start] == START_BYTE:
                    end = start + RESPONSE_LEN + 1
                    if end < len(data) and data[end] == END_BYTE:
                        return
            time.sleep(0.05)
        raise RuntimeError(
            f"Resposta '<{RESPONSE_LEN}xint8>' nao encontrada em {log_path} " f"apos {timeout}s"
        )
