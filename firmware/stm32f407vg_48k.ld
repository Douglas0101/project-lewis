/* Linker script para STM32F407VG no Renode com 48 KB de RAM (QG12). */

ENTRY(_start)

MEMORY
{
    FLASH (rx)  : ORIGIN = 0x08000000, LENGTH = 1024K
    RAM (rwx)   : ORIGIN = 0x20000000, LENGTH = 48K
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
    __heap_top = ORIGIN(RAM) + LENGTH(RAM) - 4K;

    __end__ = __heap_bottom;
    end = __end__;

    __stack_top = ORIGIN(RAM) + LENGTH(RAM);
}
