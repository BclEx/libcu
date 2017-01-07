/*
crtdefscu.h - xxx
The MIT License

Copyright (c) 2016 Sky Morey

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#pragma once

#include <crtdefs.h>
#if !defined(_CRTDEFS_H)
#define _CRTDEFS_H

/* Define NULL pointer value */
#ifndef NULL
#ifdef __cplusplus
#define NULL    0
#else
#define NULL    ((void *)0)
#endif
#endif
#define double64 double

#define MEMORY_ALIGNMENT 4096
#define _ROUNDT(t, x)		(((x)+sizeof(t)-1)&~(sizeof(t)-1))
#define _ROUND8(x)			(((x)+7)&~7)
#define _ROUNDN(x, size)	(((size_t)(x)+(size-1))&~(size-1))
#define _ROUNDDOWN8(x)		((x)&~7)
#define _ROUNDDOWNN(x, size) (((size_t)(x))&~(size-1))
#ifdef BYTEALIGNED4
#define HASALIGNMENT8(x) ((((char *)(x) - (char *)0)&3) == 0)
#else
#define HASALIGNMENT8(x) ((((char *)(x) - (char *)0)&7) == 0)
#endif

#define _LENGTHOF(symbol) (sizeof(symbol) / sizeof(symbol[0]))

#include <host_defines.h>

#ifdef  __cplusplus
extern "C" {
#endif

// BUILDIN
_CRTIMP _CRTNOALIAS void __cdecl free(_Pre_maybenull_ _Post_invalid_ void *_Memory);
_Check_return_ _Ret_maybenull_ _Post_writable_byte_size_(_Size) _CRTIMP _CRT_JIT_INTRINSIC _CRTNOALIAS _CRTRESTRICT void * __cdecl malloc(_In_ size_t _Size);
_CRTIMP __declspec(noreturn) void __cdecl exit(_In_ int _Code);
_Check_return_opt_ _CRTIMP int __cdecl printf(_In_z_ _Printf_format_string_ const char *_Format, ...);
#define panic(fmt, ...) printf(fmt, __VA_ARGS__); asm("trap;")

#ifdef  __cplusplus
}
#endif

__forceinline __device__ void *tagalloc(void *tag, size_t size) { return nullptr; } //return malloc(size); }
__forceinline __device__ void tagfree(void *tag, void *p) { }
__forceinline __device__ void *tagrealloc(void *tag, void *old, size_t size) { return nullptr; }

// ASSERT
//#ifndef NDEBUG
//#define ASSERTONLY(X) X
//__device__ __forceinline void Coverage(int line) { }
//#define ASSERTCOVERAGE(X) if (X) { Coverage(__LINE__); }
//#else
#define ASSERTONLY(X)
#define ASSERTCOVERAGE(X)
//#endif
#define _ALWAYS(X) (X)
#define _NEVER(X) (X)

#endif  /* _CRTDEFS_H */