/*  hostcall_printf.c

MIT License

Copyright © 2019 Advanced Micro Devices, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished
to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*/

#include <ctype.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "amd_hostcall.h"
#define NUMFPREGS 8
#define FPREGSZ 16


typedef enum gpusrv_status_t {
  GPUSRV_STATUS_SUCCESS = 0,
  GPUSRV_STATUS_UNKNOWN = 1,
  GPUSRV_STATUS_ERROR = 2,
  GPUSRV_STATUS_TERMINATE = 3,
} gpusrv_status_t;

enum gpusrv_dataclass {
  GPUSRV_DATACLASS_EMPTY,
  GPUSRV_DATACLASS_INTEGER,
  GPUSRV_DATACLASS_CHAR,
  GPUSRV_DATACLASS_FP,
  GPUSRV_DATACLASS_UNDEF
};
typedef enum gpusrv_dataclass gpusrv_dataclass_t;

typedef int uint128_t __attribute__((mode(TI)));
struct gpusrv_pfIntRegs {
  uint64_t rdi, rsi, rdx, rcx, r8, r9;
};
typedef struct gpusrv_pfIntRegs gpusrv_pfIntRegs_t; // size = 48 bytes

struct gpusrv_pfRegSaveArea {
  gpusrv_pfIntRegs_t iregs;
  uint128_t freg[NUMFPREGS];
};
typedef struct gpusrv_pfRegSaveArea gpusrv_pfRegSaveArea_t; // size = 304 bytes

struct gpusrv_ValistExt {
  uint32_t gp_offset;      /* offset to next available gpr in reg_save_area */
  uint32_t fp_offset;      /* offset to next available fpr in reg_save_area */
  void *overflow_arg_area; /* args that are passed on the stack */
  gpusrv_pfRegSaveArea_t *reg_save_area; /* int and fp registers */
  // need to track how much of overflow we are using for malloc
  size_t overflow_size;
} __attribute__((packed));
typedef struct gpusrv_ValistExt gpusrv_ValistExt_t;

// Macro to get string length rounded up to 4 byte alignment
// The typical rounding up formula is (((x-1)/4)+1)*4
// But strlen does not include termination character so we add 1
#define STRLENA4(x) ((strlen(x) / 4) + 1) * 4


// no debug for now 
#define DEBUG_PRINT(fmt, ...)

// Prototypes for static functions used in this file
static gpusrv_dataclass_t gpusrv_pfGetClassAndSize(char *numdata, char *sdata,
                                                   const char *fmt,
                                                   size_t *nbytes,
                                                   size_t *strbytes);
static gpusrv_status_t gpusrv_pfAddInteger(gpusrv_ValistExt_t *valist, void *val,
                                         size_t valsize, size_t *stacksize);
static gpusrv_status_t gpusrv_pfAddString(gpusrv_ValistExt_t *valist, char *val,
                                          size_t strsz, size_t *stacksize);
static gpusrv_status_t gpusrv_pfAddFloat(gpusrv_ValistExt_t *valist,
                                         char *numdata, size_t valsize,
                                         size_t *stacksize);
static gpusrv_status_t gpusrv_pfBuildValist(gpusrv_ValistExt_t *valist,
                                            const char *fmt, char *numdata,
                                            char *sdata,
                                            long long *data_not_used);
static gpusrv_status_t gpusrv_pfGetOverflow(gpusrv_ValistExt_t *valist,
                                          size_t needsize);


// hostcall_printf: HOSTCALL service for printf
//
amd_hostcall_error_t hostcall_printf(char *buf, size_t bufsz) {
  if (bufsz == 0) 
    return AMD_HOSTCALL_SUCCESS;
  char *fmtstr;
  char *numdata;
  char *sdata;

  int *datalen = (int *)buf;
  int *fmtstr_len = (int *)(buf + sizeof(int));
  long long remaining_data_size = (long long)(*datalen) - (2 * sizeof(int));

  fmtstr = (char *)(buf + ((size_t)*datalen));
  // Pointer to numeric data
  numdata = (char *)(buf + (2 * sizeof(int))); // move past buflen and fmsstrlen
  // Pointer to string data following the format string
  sdata = (char *)(buf + ((size_t)*datalen + (size_t)*fmtstr_len));

  if (fmtstr_len <= 0) {
    DEBUG_PRINT("hostcall_printf: Empty or missing format string.\n");
    return AMD_HOSTCALL_ERROR_INVALID_REQUEST;
  }

  gpusrv_ValistExt_t valist;

  va_list *real_va_list;
  real_va_list = (va_list *)&valist;
  long long data_not_used = remaining_data_size;
  if (gpusrv_pfBuildValist(&valist, fmtstr, numdata, sdata, &data_not_used) !=
      GPUSRV_STATUS_SUCCESS)
    return AMD_HOSTCALL_ERROR_INVALID_REQUEST;

  if (data_not_used < 0) {
    //  Terminate if you ran past end of buffer
    DEBUG_PRINT("format %s consumed more than %lld available bytes\n", fmtstr,
                remaining_data_size);
    return AMD_HOSTCALL_ERROR_INVALID_REQUEST;
  }

  // Roll back offsets and save stack pointer for vprintf to consume
  valist.gp_offset = 0;
  valist.fp_offset = sizeof(gpusrv_pfIntRegs_t);
  void *save_stack = valist.overflow_arg_area;

  vprintf(fmtstr, *real_va_list);

  // Cleanup allocated areas.
  if (valist.reg_save_area)
    free(valist.reg_save_area);
  if (save_stack)
    free(save_stack);

  return AMD_HOSTCALL_SUCCESS;
}

// gpusrv_pfBuildValist:
//   Called by gpusrv_printf to build the valist needed by vprintf.
//   Calls gpusrv_pfGetClassAndSize, gpusrv_pfAddInteger, gpusrv_pfAddFloat,
//   and gpusrv_pfAddString.
static gpusrv_status_t gpusrv_pfBuildValist(gpusrv_ValistExt_t *valist,
                                            const char *fmt, char *numdata,
                                            char *sdata,
                                            long long *data_not_used) {
  gpusrv_pfRegSaveArea_t *regs;
  size_t regs_size = sizeof(*regs);
  regs = (gpusrv_pfRegSaveArea_t *)malloc(regs_size);
  if (!regs) {
    DEBUG_PRINT("reg malloc failed \n");
    return GPUSRV_STATUS_ERROR;
  }
  memset(regs, 0, regs_size);
  *valist = (gpusrv_ValistExt_t){
      .gp_offset = 0,
      .fp_offset = 0,
      .overflow_arg_area = NULL,
      .reg_save_area = regs,
      .overflow_size = 0,
  };
  const char *f;
  size_t sz;
  size_t strsz;
  gpusrv_dataclass_t data_class;
  size_t stacksize = 0;

  int skip_star_fmt=0;
  for (f = fmt; *f != '\0'; ++f) {
    if ((*f != '%') && (*f != '*'))
      continue;
    if ((*f == '*') && (!skip_star_fmt))
      continue;
    data_class = gpusrv_pfGetClassAndSize(numdata, sdata, f, &sz, &strsz);
    if (*data_not_used < sz) {
      DEBUG_PRINT("not enough data totalsz:%lld sz:%ld format:%s\n",
                  *data_not_used, sz, fmt);
      return GPUSRV_STATUS_ERROR;
    }
    if (data_class == GPUSRV_DATACLASS_INTEGER) {
      if (gpusrv_pfAddInteger(valist, numdata, sz, &stacksize)) {
        DEBUG_PRINT("Could not add integer\n");
        return GPUSRV_STATUS_ERROR;
      }
    } else if (data_class == GPUSRV_DATACLASS_FP) {
      if (valist->fp_offset == 0)
        valist->fp_offset = sizeof(gpusrv_pfIntRegs_t);
      if (gpusrv_pfAddFloat(valist, numdata, sz, &stacksize)) {
        DEBUG_PRINT("Could not add float\n");
        return GPUSRV_STATUS_ERROR;
      }
    } else if (data_class == GPUSRV_DATACLASS_CHAR) {
      if (gpusrv_pfAddString(valist, (char *)&sdata, strsz, &stacksize)) {
        DEBUG_PRINT("Could not add character \n");
        return GPUSRV_STATUS_ERROR;
      }
    } else if (data_class == GPUSRV_DATACLASS_EMPTY) {
      // do nothing
    } else if (data_class == GPUSRV_DATACLASS_UNDEF) {
      DEBUG_PRINT("Not a valid specifier\n");
      return GPUSRV_STATUS_ERROR;
    }
    numdata += sz;
    sdata += strsz;
    *data_not_used -= sz;
    skip_star_fmt = 0;
    if ( f[1] == '*')
      skip_star_fmt = 1;
    else
      ++f;
  }
  return GPUSRV_STATUS_SUCCESS;
} // end gpusrv_pfBuildValist

// gpusrv_pfGetClassAndSize:
//   Called by gpusrv_pfBuildValist.  It parses the next specifier to
//   determine the clas and size in bytes.  It returns GPUSRV_DataClass
//   and provides the number of data bytes in the last argument.
//   numdata  - pointer to the current data buffer. This is only used to
//            find the data length in case next specifier is a %s string.
//            Neither the numdata pointer or data are updated here.
//   sdata  - pointer to string data buffer
//   fmt    - The next specifier. It must start with %.
//            Neither the fmt pointer or the format string are updated.
//   nbytes - The calculated number of bytes to consume from the data
//            buffer for the specifier discovered by this routine.
//   strbytes - The calculated number of bytes to consume from the string data
//            buffer for the specifier discovered by this routine.
// int isdigit(const char);
static gpusrv_dataclass_t gpusrv_pfGetClassAndSize(char *numdata, char *sdata,
                                                   const char *fmt,
                                                   size_t *nbytes,
                                                   size_t *strbytes) {
  gpusrv_dataclass_t data_class = GPUSRV_DATACLASS_UNDEF;
  const char *p;
  size_t intsz;
  *nbytes = -1;
  *strbytes = 0;
  if ((fmt[0] != '%') && (fmt[0] != '*'))
    return GPUSRV_DATACLASS_UNDEF;
  // Skip flags. accept duplicate flags
  for (p = fmt + 1; p[0]; ++p) {
    int dup = 0;
    switch (p[0]) {
    case '#':
    case '0':
    case '-':
    case ' ':
    case '+':
    case '\'':
      dup = 1;
      break;
    }
    if (!dup)
      break;
  }
  // Skip minimum field width, if any
  for (; isdigit(p[0]); ++p)
    if (p[0] == '.')
      ++p;
  // Skip precision, if any
  for (; isdigit(p[0]); ++p)
    ;
  intsz = 0;
  switch (p[0]) {
  case 'h':
    if (p[1] == 'h') {
      ++p;
      intsz = sizeof(char);
    } else {
      intsz = sizeof(short);
    }
    break;
  case 'l':
    if (p[1] == 'l') {
      ++p;
      intsz = sizeof(long long);
    } else {
      intsz = sizeof(long);
    }
    break;
  case 'j':
    intsz = sizeof(intmax_t);
    break;
  case 't':
    intsz = sizeof(size_t);
    break;
  case 'z':
    intsz = sizeof(size_t);
    break;
  default:
    p--;
    break;
  }

  if (intsz == 0)
    intsz = sizeof(int);
  ++p;

  switch (p[0]) {
  case 'c':
    *nbytes = sizeof(char);
    data_class = GPUSRV_DATACLASS_INTEGER;
    break;
  case 'd':
  case '*':
  case 'i':
  case 'o':
  case 'u':
  case 'x':
  case 'X':
    *nbytes = intsz;
    data_class = GPUSRV_DATACLASS_INTEGER;
    break;
  case 'p':
    *nbytes = sizeof(void *);
    data_class = GPUSRV_DATACLASS_INTEGER;
    break;
  case 'f':
  case 'F':
  case 'e':
  case 'E':
  case 'g':
  case 'G':
  case 'a':
  case 'A':
    // singles already converted to double on GPU
    *nbytes = sizeof(double);
    data_class = GPUSRV_DATACLASS_FP;
    break;
  case 's':
    *strbytes = (int)*numdata;
    *nbytes = 4;
    data_class = GPUSRV_DATACLASS_CHAR;
    break;
  case '%':
    *nbytes = 0;
    DEBUG_PRINT("Unknown specifier %c starting with %s", p[0], fmt - 1);
    data_class = GPUSRV_DATACLASS_EMPTY;
    break;
  default:
    *nbytes = 0;
    DEBUG_PRINT("Unknown specifier %c starting with %s", p[0], fmt - 1);
    return GPUSRV_DATACLASS_UNDEF;
  }
  return data_class;
} // end of gpusrv_pfGetClassAndSize

// gpusrv_pfAddInteger:
//   Called by gpusrv_pfBuildValist
static gpusrv_status_t gpusrv_pfAddInteger(gpusrv_ValistExt_t *valist, void *val,
                                         size_t valsize, size_t *stacksize) {
  uint64_t ival;
  switch (valsize) {
  case 1:
    ival = *(uint8_t *)val;
    break;
  case 2:
    ival = *(uint32_t *)val;
    break;
  case 4:
    ival = (*(uint32_t *)val);
    break;
  case 8:
    ival = *(uint64_t *)val;
    break;
  default: {
    DEBUG_PRINT("invalid valsize\n");
    return GPUSRV_STATUS_ERROR;
  }
  }
  //  Always copy 8 bytes sizeof(ival)
  if ((valist->gp_offset + sizeof(ival)) <= sizeof(gpusrv_pfIntRegs_t)) {
    memcpy(((char *)valist->reg_save_area + valist->gp_offset), &ival,
           sizeof(ival));
    valist->gp_offset += sizeof(ival);
    return GPUSRV_STATUS_SUCCESS;
  }
  // Ensure valist overflow area is big enough
  size_t needsize = (size_t)*stacksize + sizeof(ival);
  if (gpusrv_pfGetOverflow(valist, needsize) != GPUSRV_STATUS_SUCCESS)
    return GPUSRV_STATUS_ERROR;
  // Copy to overflow
  memcpy((char *)(valist->overflow_arg_area) + (size_t)*stacksize, &ival,
         sizeof(ival));
  *stacksize += sizeof(ival);
  return GPUSRV_STATUS_SUCCESS;
}

// gpusrv_pfAddString:
//   Called by gpusrv_pfBuildValist
static gpusrv_status_t gpusrv_pfAddString(gpusrv_ValistExt_t *valist, char *val,
                                          size_t strsz, size_t *stacksize) {
  // size_t valsize = strsz;
  size_t valsize =
      sizeof(char *); // ABI captures pointer to string,  not string
  if ((valist->gp_offset + valsize) <= sizeof(gpusrv_pfIntRegs_t)) {
    memcpy(((char *)valist->reg_save_area + valist->gp_offset), val, valsize);
    valist->gp_offset += valsize;
    return GPUSRV_STATUS_SUCCESS;
  }
  size_t needsize = (size_t)*stacksize + valsize;
  if (gpusrv_pfGetOverflow(valist, needsize) != GPUSRV_STATUS_SUCCESS)
    return GPUSRV_STATUS_ERROR;
  memcpy((char *)(valist->overflow_arg_area) + (size_t)*stacksize, val,
         valsize);
  *stacksize += valsize;
  return GPUSRV_STATUS_SUCCESS;
}

// gpusrv_pfAddFloat:
//   Called by gpusrv_pfBuildValist
static gpusrv_status_t gpusrv_pfAddFloat(gpusrv_ValistExt_t *valist,
                                         char *numdata, size_t valsize,
                                         size_t *stacksize) {
  double dval;
  if (valsize == 4) {
    float fval;
    memcpy(&fval, numdata, 4);
    dval = (double)fval; // Extend single to double per abi
  } else if (valsize == 8) {
    memcpy(&dval, numdata, 8);
  } else {
    DEBUG_PRINT("Floating point values must be 4 or 8 bytes\n");
    return GPUSRV_STATUS_ERROR;
  }
  if ((valist->fp_offset + FPREGSZ) <= sizeof(gpusrv_pfRegSaveArea_t)) {
    memcpy(((char *)valist->reg_save_area + (size_t)(valist->fp_offset)), &dval,
           sizeof(double));
    // move by full fpregsz (16 bytes) even though only 8 bytes copied.
    valist->fp_offset += FPREGSZ;
    return GPUSRV_STATUS_SUCCESS;
  }
  size_t needsize = (size_t)*stacksize + sizeof(double);
  if (gpusrv_pfGetOverflow(valist, needsize) != GPUSRV_STATUS_SUCCESS)
    return GPUSRV_STATUS_ERROR;
  memcpy((char *)(valist->overflow_arg_area) + (size_t)*stacksize, &dval,
         sizeof(double));
  // move only by the size of the double (8 bytes)
  *stacksize += sizeof(double);
  return GPUSRV_STATUS_SUCCESS;
}

static gpusrv_status_t gpusrv_pfGetOverflow(gpusrv_ValistExt_t *valist,
                                          size_t needsize) {

  if (needsize < valist->overflow_size)
    return GPUSRV_STATUS_SUCCESS;

  // Make the overflow area bigger
  size_t stacksize;
  void *newstack;
  if (valist->overflow_size == 0) {
    // Make initial save area big to reduce mallocs
    stacksize = (FPREGSZ * NUMFPREGS) * 2;
    if (needsize > stacksize)
      stacksize = needsize; // maybe a big string
  } else {
    // Initial save area not big enough, double it
    stacksize = valist->overflow_size * 2;
  }
  if (!(newstack = malloc(stacksize))) {
    DEBUG_PRINT(" malloc failed for Overflow area \n");
    return GPUSRV_STATUS_ERROR;
  }
  memset(newstack, 0, stacksize);
  if (valist->overflow_size) {
    memcpy(newstack, valist->overflow_arg_area, valist->overflow_size);
    free(valist->overflow_arg_area);
  }
  valist->overflow_arg_area = newstack;
  valist->overflow_size = stacksize;
  return GPUSRV_STATUS_SUCCESS;
}
