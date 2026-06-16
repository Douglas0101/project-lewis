/* Stubs minimos para C++ bare-metal sem libstdc++.
 *
 * O TFLM em C++ requer __cxa_guard_* para static locals nao-triviais
 * e operator delete quando headers possuem destrutores virtuais.
 * Aqui fornecemos implementacoes vazias suficientes para linkar.
 */

#include <stddef.h>

extern "C" {

int __cxa_guard_acquire(__attribute__((unused)) char* guard)
{
    return 1;
}

void __cxa_guard_release(__attribute__((unused)) char* guard)
{
}

void __cxa_guard_abort(__attribute__((unused)) char* guard)
{
}

}

void operator delete(void* ptr) noexcept
{
    (void)ptr;
}

void operator delete[](void* ptr) noexcept
{
    (void)ptr;
}

void operator delete(void* ptr, __attribute__((unused)) size_t size) noexcept
{
    (void)ptr;
}

void operator delete[](void* ptr, __attribute__((unused)) size_t size) noexcept
{
    (void)ptr;
}
