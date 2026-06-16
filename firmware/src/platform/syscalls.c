/* Stubs de syscalls newlib para build bare-metal sem warnings.
 *
 * Evita que o linker emita warnings dos stubs padrao do nosys.specs.
 * O firmware nao usa stdio real; a comunicacao e pela UART4.
 */

#include <sys/stat.h>
#include <errno.h>

int _close(int file)
{
    (void)file;
    errno = EBADF;
    return -1;
}

int _fstat(int file, struct stat* st)
{
    (void)file;
    st->st_mode = S_IFCHR;
    return 0;
}

int _isatty(int file)
{
    (void)file;
    return 1;
}

int _lseek(int file, int ptr, int dir)
{
    (void)file;
    (void)ptr;
    (void)dir;
    return 0;
}

int _read(int file, char* ptr, int len)
{
    (void)file;
    (void)ptr;
    (void)len;
    errno = EBADF;
    return -1;
}

int _write(int file, char* ptr, int len)
{
    (void)file;
    (void)ptr;
    (void)len;
    errno = EBADF;
    return -1;
}

int _getpid(void)
{
    return 1;
}

int _kill(int pid, int sig)
{
    (void)pid;
    (void)sig;
    while (1) {}
    return -1;
}
