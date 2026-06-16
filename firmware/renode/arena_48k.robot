*** Settings ***
Library    OperatingSystem

*** Variables ***
${RESC}      ${CURDIR}/arena_48k.resc
${UART_LOG}  /tmp/renode_lewis_uart.log
${TIMEOUT}   15

*** Keywords ***
Log Contains Init Fail
    ${content}=    Get File    ${UART_LOG}
    Should Contain    ${content}    INIT FAIL

*** Test Cases ***
Arena Limit 48 KB Causes Init Fail
    Remove File    ${UART_LOG}
    Execute Script    ${RESC}
    # O INIT FAIL e emitido durante o boot; usamos o arquivo de log da UART
    # em vez do log interno do Renode, que comeca a monitoracao apos o boot.
    Wait Until Keyword Succeeds    ${TIMEOUT}s    1s
    ...    Log Contains Init Fail
