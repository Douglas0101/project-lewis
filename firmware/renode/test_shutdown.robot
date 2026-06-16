*** Settings ***
Library    OperatingSystem

*** Variables ***
${RESC}      ${CURDIR}/stm32f4_discovery.resc
${UART_LOG}  /tmp/renode_lewis_uart.log
${TIMEOUT}   30

*** Keywords ***
Send Command
    [Arguments]    ${text}
    ${length}=    Get Length    ${text}
    FOR    ${idx}    IN RANGE    ${length}
        ${char}=    Get Substring    ${text}    ${idx}    ${idx+1}
        ${code}=    Evaluate    ord('${char}')
        Execute Command    sysbus.uart4 WriteChar ${code}
        Sleep    0.01
    END
    Execute Command    sysbus.uart4 WriteChar 10

*** Test Cases ***
Graceful Shutdown Via UART
    Remove File    ${UART_LOG}
    Execute Script    ${RESC}
    Create Log Tester    0
    Wait For Log Entry    Modo comando UART ativo    timeout=${TIMEOUT}
    Send Command    SHUTDOWN
    Wait For Log Entry    === Fim ===    timeout=${TIMEOUT}
