/* Linker script para STM32F407VG no Renode — executavel de test harness.
 *
 * Diferente do linker script principal (stm32f407vg.ld), este reserva uma
 * pilha maior para acomodar testes de firmware com sinais de entrada maiores
 * (ex.: detector de R-peaks com VLA).
 */

ENTRY(_start)

MEMORY
{
    FLASH (rx)  : ORIGIN = 0x08000000, LENGTH = 1024K
    RAM (rwx)   : ORIGIN = 0x20000000, LENGTH = 192K
}

SECTIONS
{
    .text :
    {
        KEEP(*(.isr_vector))
        *(.text*)
        *(.rodata*)
        *(.glue_7)
        *(.glue_7t)
        . = ALIGN(16);
        _etext = .;
    } > FLASH

    .rodata.model (READONLY) :
    {
        . = ALIGN(16);
        *(.rodata.model)
        . = ALIGN(16);
    } > FLASH

    __exidx_start = .;
    __exidx_end = .;

    _sidata = LOADADDR(.data);
    .data :
    {
        _sdata = .;
        *(.data*)
        . = ALIGN(4);
        _edata = .;
    } > RAM AT > FLASH

    .bss :
    {
        __bss_start__ = .;
        _sbss = .;
        *(.bss*)
        *(COMMON)
        . = ALIGN(4);
        _ebss = .;
        __bss_end__ = .;
    } > RAM

    .init_array :
    {
        __init_array_start = .;
        KEEP(*(SORT(.init_array.*)))
        KEEP(*(.init_array*))
        __init_array_end = .;
    } > FLASH

    __heap_bottom = ALIGN(8);
    /* Reserva 64 KB de pilha para o harness. */
    __heap_top = ORIGIN(RAM) + LENGTH(RAM) - 64K;

    __end__ = __heap_bottom;
    end = __end__;

    __stack_top = ORIGIN(RAM) + LENGTH(RAM);
}
