/* Startup minimalista para STM32F407VG no Renode.
 *
 * Define o vetor de reset com stack pointer inicial e Reset_Handler.
 * O Reset_Handler inicializa .data, zera .bss e chama main().
 */

#include <stdint.h>
#include <string.h>

/* Fornecido pelo linker script. */
extern uint32_t _sidata;
extern uint32_t _sdata;
extern uint32_t _edata;
extern uint32_t _sbss;
extern uint32_t _ebss;
extern uint32_t __stack_top;

extern int main(void);

extern void (*__init_array_start[])(void);
extern void (*__init_array_end[])(void);

static void Reset_Handler(void)
{
    /* Habilita FPU (CP10/CP11 full access). */
    volatile uint32_t* cpacr = (volatile uint32_t*)0xE000ED88;
    *cpacr |= (0xFU << 20);

    /* Copia .data da Flash para RAM. */
    uint32_t* src = &_sidata;
    uint32_t* dst = &_sdata;
    while (dst < &_edata) {
        *dst++ = *src++;
    }

    /* Zera .bss. */
    dst = &_sbss;
    while (dst < &_ebss) {
        *dst++ = 0;
    }

    /* Chama construtores globais C++. */
    for (void (**p)(void) = __init_array_start; p < __init_array_end; ++p) {
        (*p)();
    }

    main();

    while (1) {
        /* Loop infinito apos retorno de main (nao deveria acontecer). */
    }
}

/* Handler padrao para excecoes nao tratadas. */
static void Default_Handler(void)
{
    while (1) {
    }
}

/* Atribui Default_Handler a todas as excecoes. */
void NMI_Handler(void) __attribute__((weak, alias("Default_Handler")));
void HardFault_Handler(void) __attribute__((weak, alias("Default_Handler")));
void MemManage_Handler(void) __attribute__((weak, alias("Default_Handler")));
void BusFault_Handler(void) __attribute__((weak, alias("Default_Handler")));
void UsageFault_Handler(void) __attribute__((weak, alias("Default_Handler")));
void SVC_Handler(void) __attribute__((weak, alias("Default_Handler")));
void DebugMon_Handler(void) __attribute__((weak, alias("Default_Handler")));
void PendSV_Handler(void) __attribute__((weak, alias("Default_Handler")));
void SysTick_Handler(void) __attribute__((weak, alias("Default_Handler")));

/* Vetor de interrupcoes. */
__attribute__((section(".isr_vector"))) const void* g_pfnVectors[] = {
    (void*)&__stack_top,
    (void*)Reset_Handler,
    NMI_Handler,
    HardFault_Handler,
    MemManage_Handler,
    BusFault_Handler,
    UsageFault_Handler,
    0,
    0,
    0,
    0,
    SVC_Handler,
    DebugMon_Handler,
    0,
    PendSV_Handler,
    SysTick_Handler,
};
