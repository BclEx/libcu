/*
stddef.h - definitions/declarations for common constants, types, variables
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

/* CUDA double64 is double */
#ifndef double64
#define double64 double
#endif

#ifdef __CUDA_ARCH__
#ifndef _STDDEFCU_H
#define _STDDEFCU_H
#define _STDDEF_H
#define _INC_STDDEF
#include <crtdefscu.h>
#ifdef  __cplusplus
extern "C" {
#endif

#define panic(fmt, ...) { printf(fmt, __VA_ARGS__); asm("trap;"); }

	/* Define NULL pointer value */
#ifndef NULL
#ifdef __cplusplus
#define NULL 0
#else
#define NULL ((void *)0)
#endif
#endif


#ifdef __cplusplus
	namespace std { typedef decltype(__nullptr) nullptr_t; }
	using ::std::nullptr_t;
#endif

	/* Define offsetof macro */
#ifdef __cplusplus

#ifdef  _WIN64
#define offsetof(s,m) (size_t)((ptrdiff_t)&reinterpret_cast<const volatile char&>((((s *)0)->m)))
#else
#define offsetof(s,m) (size_t)&reinterpret_cast<const volatile char&>((((s *)0)->m))
#endif

#else

#ifdef  _WIN64
#define offsetof(s,m) (size_t)( (ptrdiff_t)&(((s *)0)->m) )
#else
#define offsetof(s,m) (size_t)&(((s *)0)->m)
#endif

#endif	/* __cplusplus */

	//	_CRTIMP extern unsigned long  __cdecl __threadid(void);
	//#define _threadid (__threadid())
	//	_CRTIMP extern uintptr_t __cdecl __threadhandle(void);

#ifdef  __cplusplus
}
#endif
#endif  /* _STDDEFCU_H */
#else
#define panic(fmt, ...) { printf(fmt, __VA_ARGS__); exit(1); }
#include <stddef.h>
#endif

/* Define tag allocs */
__forceinline __device__ void *tagalloc(void *tag, size_t size) { return nullptr; }
__forceinline __device__ void tagfree(void *tag, void *p) { }
__forceinline __device__ void *tagrealloc(void *tag, void *old, size_t size) { return nullptr; }

/* Define assert helpers */
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