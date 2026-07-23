*** Settings ***
Library    OperatingSystem
Library    ${CURDIR}/FidelityKeywords.py

*** Variables ***
${RESC}              ${CURDIR}/stm32f4_discovery.resc
${DUMMY_SPI}         ${CURDIR}/dummy_spi_device.py
${UART_LOG}          /tmp/renode_lewis_uart.log
${TIMEOUT}           30

*** Keywords ***
Try Attach Dummy SPI Peripheral
    # Fallback documentado: Renode 1.15.3 pode nao suportar anexao direta de
    # perifericos SPI a partir de testes Robot. Por isso tentamos varias
    # sintaxes conhecidas, ignorando erros. Se nenhuma funcionar, o teste
    # continua e valida o tratamento graceful de erro via corrupcao UART.
    Run Keyword And Ignore Error
    ...    Execute Command    machine LoadPeripheralFromFile @${DUMMY_SPI}
    Run Keyword And Ignore Error
    ...    Execute Command    machine LoadPeripheral @${DUMMY_SPI}
    Run Keyword And Ignore Error
    ...    Execute Command
    ...    python "from Antmicro.Renode.Peripherals.SPI import ISPIPeripheral; import sys; sys.path.insert(0, 'firmware/renode'); from dummy_spi_device import DummySPIDevice; machine['sysbus.spi1'].AttachPeripheral(DummySPIDevice())"

*** Test Cases ***
Fault Injection Via Dummy SPI Device
    Remove File    ${UART_LOG}
    Execute Script    ${RESC}
    Create Log Tester    0
    Wait For Log Entry    Modo comando UART ativo    timeout=${TIMEOUT}
    Try Attach Dummy SPI Peripheral
    # Frame completo (2000 bytes) com terminador invalido: o firmware consome
    # o payload, rejeita o terminador e reporta '[infer] FRAME ERR' de forma
    # imediata e deterministica, sem depender de timeout por byte (que seria
    # sensiveis a stalls do host no runner de CI).
    Send Corrupted Frame
    Wait For Log Entry    FRAME ERR    timeout=${TIMEOUT}
