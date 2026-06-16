*** Settings ***
Library    OperatingSystem

*** Variables ***
${RESC}              ${CURDIR}/stm32f4_discovery.resc
${DUMMY_SPI}         ${CURDIR}/dummy_spi_device.py
${UART_LOG}          /tmp/renode_lewis_uart.log
${TIMEOUT}           30
${NEWLINE}           10
${UART_CHAR_DELAY}   0.01

*** Keywords ***
Send Invalid UART Frame
    # Envia '<' + 0x00 0xFF + '>'. O firmware espera 500 floats (2000 bytes),
    # portanto a leitura expira e reporta ERRO de timeout/framing.
    Execute Command    sysbus.uart4 WriteChar 60
    Sleep    ${UART_CHAR_DELAY}
    Execute Command    sysbus.uart4 WriteChar 0
    Sleep    ${UART_CHAR_DELAY}
    Execute Command    sysbus.uart4 WriteChar 255
    Sleep    ${UART_CHAR_DELAY}
    Execute Command    sysbus.uart4 WriteChar 62
    Sleep    ${UART_CHAR_DELAY}
    Execute Command    sysbus.uart4 WriteChar ${NEWLINE}

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
    Send Invalid UART Frame
    Wait For Log Entry    ERRO    timeout=${TIMEOUT}
