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

//#pragma once
#ifndef _CRTDEFSCU_H
#define _CRTDEFSCU_H

#include <crtdefs.h>
//#if defined(__CUDA_ARCH__) || defined(LIBCUFORCE)
//#endif  /* __CUDA_ARCH__ */

//#ifdef  __cplusplus
//extern "C" {
//#endif
//
//	/* Built In */
//	_CRTIMP _CRTNOALIAS void __cdecl free(_Pre_maybenull_ _Post_invalid_ void *_Memory);
//	_Check_return_ _Ret_maybenull_ _Post_writable_byte_size_(_Size) _CRTIMP _CRT_JIT_INTRINSIC _CRTNOALIAS _CRTRESTRICT void * __cdecl malloc(_In_ size_t _Size);
//	_CRTIMP __declspec(noreturn) void __cdecl exit(_In_ int _Code);
//	_Check_return_opt_ _CRTIMP int __cdecl printf(_In_z_ _Printf_format_string_ const char *_Format, ...);
//	//void __cdecl free(void *memory);
//	//void * __cdecl malloc(size_t size);
//	//__declspec(noreturn) void __cdecl exit(int code);
//	//int __cdecl printf(const char *format, ...);
//
//#ifdef  __cplusplus
//}
//#endif

#define MEMORY_ALIGNMENT 4096
#define _ROUNDT(t, x)		(((x)+sizeof(t)-1)&~(sizeof(t)-1))
#define _ROUND8(x)			(((x)+7)&~7)
#define _ROUND64(x)			(((x)+63)&~63)
#define _ROUNDN(x, size)	(((size_t)(x)+(size-1))&~(size-1))
#define _ROUNDDOWN8(x)		((x)&~7)
#define _ROUNDDOWNN(x, size) (((size_t)(x))&~(size-1))
#ifdef BYTEALIGNED4
#define HASALIGNMENT8(x) ((((char *)(x) - (char *)0)&3) == 0)
#else
#define HASALIGNMENT8(x) ((((char *)(x) - (char *)0)&7) == 0)
#endif
#define _LENGTHOF(symbol) (sizeof(symbol) / sizeof(symbol[0]))
#define UNUSED_PARAMETER(x) (void)(x)
#define UNUSED_PARAMETER2(x,y) (void)(x),(void)(y)

//////////////////////
// WSD
#pragma region WSD

// When OMIT_WSD is defined, it means that the target platform does not support Writable Static Data (WSD) such as global and static variables.
// All variables must either be on the stack or dynamically allocated from the heap.  When WSD is unsupported, the variable declarations scattered
// throughout the SQLite code must become constants instead.  The _WSD macro is used for this purpose.  And instead of referencing the variable
// directly, we use its constant as a key to lookup the run-time allocated buffer that holds real variable.  The constant is also the initializer
// for the run-time allocated buffer.
//
// In the usual case where WSD is supported, the _WSD and _GLOBAL macros become no-ops and have zero performance impact.
#ifdef NEEDS_WSD
int __wsdinit(int n, int j);
void *__wsdfind(void *k, int l);
#define _WSD const
#define _GLOBAL(t, v) (*(t*)__wsdfind((void *)&(v), sizeof(v)))
#else
#define _WSD
#define _GLOBAL(t, v) v
#endif

#pragma endregion

#endif  /* _CRTDEFSCU_H */