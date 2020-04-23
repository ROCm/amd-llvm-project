#ifndef HOSTCALL_STUBS_H
#define HOSTCALL_STUBS_H

#define EXTERN extern "C" __attribute__((device))

#include <stdint.h>

/* These are the interfaces for the device stubs */
///
EXTERN int printf( const char * , ...);
EXTERN char *  printf_alloc(uint32_t bufsz);
EXTERN int     printf_execute(char * bufptr, uint32_t bufsz);
EXTERN uint32_t __strlen_max(char*instr, uint32_t maxstrlen);
EXTERN int     vector_product_zeros(int N, int*A, int*B, int*C);

typedef struct hostcall_result_s{
  uint64_t arg0,arg1,arg2,arg3,arg4,arg5,arg6,arg7;
} hostcall_result_t;

EXTERN hostcall_result_t hostcall_invoke(uint32_t id,
    uint64_t arg0, uint64_t arg1, uint64_t arg2, uint64_t arg3,
    uint64_t arg4, uint64_t arg5, uint64_t arg6, uint64_t arg7);

#endif
