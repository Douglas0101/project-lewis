*** Settings ***
Library    OperatingSystem
Library    ${CURDIR}/FidelityKeywords.py

*** Variables ***
${RESC}             ${CURDIR}/stm32f4_discovery.resc
${INPUT_PATH}       ${CURDIR}/../../tests/ground_truth/ecg_input_00.bin
${UART_LOG}         /tmp/renode_lewis_uart.log
${TIMEOUT}          120
${NEWLINE}            10
${START_BYTE}         60
${END_BYTE}           62
${UART_CHAR_DELAY}    0.01
${POST_SEND_DELAY}    2

*** Keywords ***
Send Text Command
    [Arguments]    ${text}
    ${length}=    Get Length    ${text}
    FOR    ${idx}    IN RANGE    ${length}
        ${char}=    Get Substring    ${text}    ${idx}    ${idx+1}
        ${code}=    Evaluate    ord('${char}')
        Execute Command    sysbus.uart4 WriteChar ${code}
        Sleep    ${UART_CHAR_DELAY}
    END
    Execute Command    sysbus.uart4 WriteChar ${NEWLINE}

*** Test Cases ***
Fidelity Inference For Beat
    Run Keyword And Ignore Error    Remove File    ${UART_LOG}
    Execute Script    ${RESC}
    Create Log Tester    0
    Wait For Log Entry    Modo comando UART ativo    timeout=${TIMEOUT}
    Send Binary Frame    ${INPUT_PATH}
    Wait For Response In Log    ${UART_LOG}    timeout=180
