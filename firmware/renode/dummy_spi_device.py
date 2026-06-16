"""Dummy SPI peripheral para injecao de falhas no Renode.

Responde a transacoes SPI com o padrao alternado ``[0x00, 0xFF, 0x00, 0xFF]``.
Implementa a interface ``Antmicro.Renode.Peripherals.SPI.ISPIPeripheral``.

Compatibilidade:
  * Renode 1.15.3 e superiores expoe ``ISPIPeripheral`` no namespace
    ``Antmicro.Renode.Peripherals.SPI``. Caso a importacao falhe (por exemplo,
    em versoes muito antigas do Renode), o script retorna ``None`` e o teste
    devera tratar a falha como ``BLOCKED``.
"""

try:
    from Antmicro.Renode.Peripherals.SPI import ISPIPeripheral
except Exception as exc:  # noqa: BLE001
    print(f"[dummy_spi_device] WARN: nao foi possivel importar ISPIPeripheral: {exc}")
    ISPIPeripheral = object


class DummySPIPeripheral(ISPIPeripheral):
    """Periferico SPI que retorna padrao fixo de bytes."""

    def __init__(self):
        self._pattern = [0x00, 0xFF, 0x00, 0xFF]
        self._index = 0

    def WriteByte(self, byte: int) -> int:
        """Retorna o proximo byte do padrao em cada transacao."""
        response = self._pattern[self._index % len(self._pattern)]
        self._index += 1
        return response

    def FinishTransmission(self):
        """Reseta o indice do padrao ao final da transmissao."""
        self._index = 0


def factory():
    """Factory function usada pelo Renode para instanciar o periferico."""
    return DummySPIPeripheral()
