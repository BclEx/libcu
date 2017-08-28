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

#ifndef _CRTDEFSCU_H
#define _CRTDEFSCU_H

#include <crtdefs.h>
#include <cuda_runtime.h>
#include <stdint.h>
#define __LIBCU__

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

//_Check_return_opt_ _CRTIMP int __cdecl printf(_In_z_ _Printf_format_string_ const char *_Format, ...);
#if defined(__CUDA_ARCH__)
#define panic(fmt, ...) { printf(fmt"\n", __VA_ARGS__); asm("trap;"); }
#else
//__forceinline void Coverage(int line) { }
#define panic(fmt, ...) { printf(fmt"\n", __VA_ARGS__); exit(1); }
#endif  /* __CUDA_ARCH__ */

/* GCC does not define the offsetof() macro so we'll have to do it ourselves. */
#ifndef offsetof
#define offsetof(STRUCTURE,FIELD) ((int)((char*)&((STRUCTURE*)0)->FIELD))
#endif

//////////////////////
// UTILITY
#pragma region UTILITY

/* For these things, GCC behaves the ANSI way normally,
and the non-ANSI way under -traditional.  */
#define __CONCAT(x,y) x ## y
#define __STRING(x) #x

/* PTX conditionals */
#ifdef _WIN64
#define _UX ".u64"
#define _BX ".b64"
#define __R "l"
#else
#define _UX ".u32"
#define _BX ".b32"
#define __R "r"
#endif

/* This is not a typedef so `const __ptr_t' does the right thing.  */
#define __ptr_t void *
//#define __long_double_t long double
/* CUDA long_double is double */
#define long_double double

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
#define _HASALIGNMENT8(x) ((((char *)(x) - (char *)0)&3) == 0)
#else
#define _HASALIGNMENT8(x) ((((char *)(x) - (char *)0)&7) == 0)
#endif
/* Returns the length of an array at compile time (via math) */
#define _LENGTHOF(symbol) (sizeof(symbol) / sizeof(symbol[0]))
/* Removes compiler warning for unused parameter(s) */
#define UNUSED_SYMBOL(x) (void)(x)
#define UNUSED_SYMBOL2(x,y) (void)(x),(void)(y)

/* Macros to compute minimum and maximum of two numbers. */
#ifndef _MIN
#define _MIN(A,B) ((A)<(B)?(A):(B))
#endif
#ifndef _MAX
#define _MAX(A,B) ((A)>(B)?(A):(B))
#endif
/* Swap two objects of type TYPE. */
#define _SWAP(TYPE,A,B) { TYPE t=A; A=B; B=t; }

#pragma endregion

//////////////////////
// PTRSIZE
#pragma region PTRSIZE

/* Set the SQLITE_PTRSIZE macro to the number of bytes in a pointer */
#ifndef _PTRSIZE
#if defined(__SIZEOF_POINTER__)
#define _PTRSIZE __SIZEOF_POINTER__
#elif defined(i386) || defined(__i386__) || defined(_M_IX86) || defined(_M_ARM) || defined(__arm__) || defined(__x86)
#define _PTRSIZE 4
#else
#define _PTRSIZE 8
#endif
#endif

/* The uptr type is an unsigned integer large enough to hold a pointer */
#if defined(HAVE_STDINT_H)
//typedef uintptr_t uintptr_t;
#elif _PTRSIZE == 4
typedef uint32_t uintptr_t;
#else
typedef uint64_t uintptr_t;
#endif

/*
** The _WITHIN(P,S,E) macro checks to see if pointer P points to something between S (inclusive) and E (exclusive).
**
** In other words, S is a buffer and E is a pointer to the first byte after the end of buffer S.  This macro returns true if P points to something
** contained within the buffer S.
*/
#define _WITHIN(P,S,E) (((uintptr_t)(P)>=(uintptr_t)(S))&&((uintptr_t)(P)<(uintptr_t)(E)))

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

#ifndef __CUDA_ARCH__
#define __host_device__ __host__
#define __hostb_device__
#define __host_constant__
#else
#define __host_device__ __device__
#define __hostb_device__ __device__
#define __host_constant__ __constant__
#endif

typedef struct hostptr_t {
	void *host;
} hostptr_t;

/* IsDevice support.  */
extern "C" __device__ char __cwd[];
#define ISHOSTPATH(path) ((path)[1] == ':' || ((path)[0] != ':' && __cwd[0] == 0))
#define ISHOSTHANDLE(handle) (handle < INT_MAX-CORE_MAXFILESTREAM)
#define ISHOSTPTR(ptr) ((hostptr_t *)(ptr) >= __iob_hostptrs && (hostptr_t *)(ptr) <= __iob_hostptrs+CORE_MAXHOSTPTR)
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

/*
** NDEBUG and _DEBUG are opposites.  It should always be true that defined(NDEBUG) == !defined(_DEBUG).  If this is not currently true,
** make it true by defining or undefining NDEBUG.
**
** Setting NDEBUG makes the code smaller and faster by disabling the assert() statements in the code.  So we want the default action
** to be for NDEBUG to be set and NDEBUG to be undefined only if _DEBUG is set.  Thus NDEBUG becomes an opt-in rather than an opt-out feature.
*/
#if !defined(NDEBUG) && !defined(_DEBUG)
#define NDEBUG 1
#endif
#if defined(NDEBUG) && defined(_DEBUG)
#undef NDEBUG
#endif

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

//////////////////////
// EXT METHODS
#pragma region EXT-METHODS

struct strbld_t;
typedef struct ext_methods ext_methods;
struct ext_methods {
	void *(*tagallocRaw)(void *tag, uint64_t size);
	void *(*tagrealloc)(void *tag, void *old, uint64_t newSize);
	int *(*tagallocSize)(void *tag, void *p);
	void (*tagfree)(void *tag, void *p);
	void (*appendFormat[2])(strbld_t *b, void *va);
	int64_t (*getIntegerArg)(void *args);
	double (*getDoubleArg)(void *args);
	char *(*getStringArg)(void *args);
};
extern "C" __device__ ext_methods __extsystem;

#pragma endregion	

#endif  /* _CRTDEFSCU_H */