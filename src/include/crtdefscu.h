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
#include <cuda_runtime.h>

#define HAS_STDIO_BUFSIZ_NONE__
//#define _LARGEFILE64_SOURCE

/* These are defined by the user (or the compiler) to specify the desired environment:
_LARGEFILE_SOURCE	Some more functions for correct standard I/O.
_LARGEFILE64_SOURCE	Additional functionality from LFS for large files.
_FILE_OFFSET_BITS=N	Select default filesystem interface.

All macros listed above as possibly being defined by this file are explicitly undefined if they are not explicitly defined. */

#ifdef _LARGEFILE_SOURCE
#define __USE_LARGEFILE		1
#endif

#ifdef _LARGEFILE64_SOURCE
#define __USE_LARGEFILE64	1
#endif

#if defined(_FILE_OFFSET_BITS) && _FILE_OFFSET_BITS == 64
#define __USE_FILE_OFFSET64	1
#endif

#ifndef CORE_MAXFILESTREAM
#define CORE_MAXFILESTREAM 10
#endif

#ifndef CORE_MAXHOSTPTR
#define CORE_MAXHOSTPTR 10
#endif

//////////////////////
// UTILITY
#pragma region UTILITY

/* For these things, GCC behaves the ANSI way normally,
and the non-ANSI way under -traditional.  */
#define __CONCAT(x,y) x ## y
#define __STRING(x) #x

/* This is not a typedef so `const __ptr_t' does the right thing.  */
#define __ptr_t void *
#define __long_double_t long double
/* CUDA double64 is double */
#ifndef double64
#define double64 double
#endif

#define MEMORY_ALIGNMENT 4096
/* Memory allocation - rounds to the type in T */
#define _ROUNDT(x, T)		(((x)+sizeof(T)-1)&~(sizeof(T)-1))
/* Memory allocation - rounds up to 8 */
#define _ROUND8(x)			(((x)+7)&~7)
/* Memory allocation - rounds up to 64 */
#define _ROUND64(x)			(((x)+63)&~63)
/* Memory allocation - rounds up to "size" */
#define _ROUNDN(x, size)	(((size_t)(x)+(size-1))&~(size-1))
/* Memory allocation - rounds down to 8 */
#define _ROUNDDOWN8(x)		((x)&~7)
/* Memory allocation - rounds down to "size" */
#define _ROUNDDOWNN(x, size) (((size_t)(x))&~(size-1))
/* Test to see if you are on aligned boundary, affected by BYTEALIGNED4 */
#ifdef BYTEALIGNED4
#define HASALIGNMENT8(x) ((((char *)(x) - (char *)0)&3) == 0)
#else
#define HASALIGNMENT8(x) ((((char *)(x) - (char *)0)&7) == 0)
#endif
/* Returns the length of an array at compile time (via math) */
#define _LENGTHOF(symbol) (sizeof(symbol) / sizeof(symbol[0]))
/* Removes compiler warning for unused parameter(s) */
#define UNUSED_SYMBOL(x) (void)(x)
#define UNUSED_SYMBOL2(x,y) (void)(x),(void)(y)

#pragma endregion

//////////////////////
// NAMESPACE
#pragma region NAMESPACE

/* C++ needs to know that types and declarations are C, not C++.  */
#ifdef	__cplusplus
#define __BEGIN_DECLS extern "C" {
#define __END_DECLS }
#else
#define __BEGIN_DECLS
#define __END_DECLS
#endif

/* The standard library needs the functions from the ISO C90 standard
in the std namespace.  At the same time we want to be safe for
future changes and we include the ISO C99 code in the non-standard
namespace __c99.  The C++ wrapper header take case of adding the
definitions to the global namespace.  */
#if defined(__cplusplus) && defined(_GLIBCPP_USE_NAMESPACES)
#define __BEGIN_NAMESPACE_STD namespace std {
#define __END_NAMESPACE_STD }
#define __USING_NAMESPACE_STD(name) using std::name;
#define __BEGIN_NAMESPACE_C99 namespace __c99 {
#define __END_NAMESPACE_C99 }
#define __USING_NAMESPACE_C99(name) using ext::name;
#define __BEGIN_NAMESPACE_EXT namespace ext {
#define __END_NAMESPACE_EXT }
#define __USING_NAMESPACE_EXT(name) using ext::name;
#else
/* For compatibility we do not add the declarations into any
namespace.  They will end up in the global namespace which is what
old code expects.  */
#define __BEGIN_NAMESPACE_STD
#define __END_NAMESPACE_STD
#define __USING_NAMESPACE_STD(name)
#define __BEGIN_NAMESPACE_C99
#define __END_NAMESPACE_C99
#define __USING_NAMESPACE_C99(name)
#define __BEGIN_NAMESPACE_EXT
#define __END_NAMESPACE_EXT
#define __USING_NAMESPACE_EXT(name)
#endif

#pragma endregion

//////////////////////
// DEVICE/HOST
#pragma region DEVICE/HOST

typedef struct hostptr_t {
	void *host;
} hostptr_t;

/* IsDevice support.  */
extern "C" __device__ char __cwd[];
#define ISDEVICEPATH(path) (((path)[1] != ':') && ((path)[0] == ':' || __cwd[0] != 0))
#define ISDEVICEHANDLE(handle) (handle >= INT_MAX-CORE_MAXFILESTREAM)
#define ISDEVICEPTR(ptr) ((hostptr_t *)(ptr) < __iob_hostptrs || (hostptr_t *)(ptr) > __iob_hostptrs + CORE_MAXHOSTPTR)
extern "C" __constant__ hostptr_t __iob_hostptrs[CORE_MAXHOSTPTR];

/* Host pointer support.  */
extern "C" __device__ hostptr_t *__hostptrGet(void *host);
extern "C" __device__ void __hostptrFree(hostptr_t *p);
template <typename T> __forceinline __device__ T *newhostptr(T *p) { return (T *)(p ? __hostptrGet(p) : nullptr); }
template <typename T> __forceinline __device__ void freehostptr(T *p) { if (p) __hostptrFree((hostptr_t *)p); }
template <typename T> __forceinline __device__ T *hostptr(T *p) { return (T *)(p ? ((hostptr_t *)p)->host : nullptr); }

#pragma endregion

//////////////////////
// ASSERT
#pragma region ASSERT

#ifndef NDEBUG
#define ASSERTONLY(X) X
#if defined(__CUDA_ARCH__)
__forceinline __device__ void Coverage(int line) { }
#else
__forceinline void Coverage(int line) { }
#endif
#define ASSERTCOVERAGE(X) if (X) { Coverage(__LINE__); }
#else
#define ASSERTONLY(X)
#define ASSERTCOVERAGE(X)
#endif
#define _ALWAYS(X) (X)
#define _NEVER(X) (X)

#pragma endregion

//////////////////////
// WSD
#pragma region WSD

// When NO_WSD is defined, it means that the target platform does not support Writable Static Data (WSD) such as global and static variables.
// All variables must either be on the stack or dynamically allocated from the heap.  When WSD is unsupported, the variable declarations scattered
// throughout the code must become constants instead.  The _WSD macro is used for this purpose.  And instead of referencing the variable
// directly, we use its constant as a key to lookup the run-time allocated buffer that holds real variable.  The constant is also the initializer
// for the run-time allocated buffer.
//
// In the usual case where WSD is supported, the _WSD and _GLOBAL macros become no-ops and have zero performance impact.
#ifdef NO_WSD
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