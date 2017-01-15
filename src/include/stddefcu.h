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

#ifdef __CUDA_ARCH__
#ifndef _INC_STDDEF
#define _INC_STDDEF
#include <crtdefscu.h>

//#define _CRTIMP
//#define _In_
//#define _Out_

#ifdef  __cplusplus
extern "C" {
#endif

	/* Built In */
	_CRTIMP _CRTNOALIAS void __cdecl free(_Pre_maybenull_ _Post_invalid_ void *_Memory);
	_Check_return_ _Ret_maybenull_ _Post_writable_byte_size_(_Size) _CRTIMP _CRT_JIT_INTRINSIC _CRTNOALIAS _CRTRESTRICT void * __cdecl malloc(_In_ size_t _Size);
	_CRTIMP __declspec(noreturn) void __cdecl exit(_In_ int _Code);
	_Check_return_opt_ _CRTIMP int __cdecl printf(_In_z_ _Printf_format_string_ const char *_Format, ...);
#define panic(fmt, ...) printf(fmt, __VA_ARGS__); asm("trap;")

	/* Define NULL pointer value */
#ifndef NULL
#ifdef __cplusplus
#define NULL 0
#else
#define NULL ((void *)0)
#endif
#endif

	/* CUDA double64 is double */
#define double64 double

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

	/* Define tag allocs */
	__forceinline __device__ void *tagalloc(void *tag, size_t size) { return nullptr; } //return malloc(size); }
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

#ifdef  __cplusplus
}
#endif

#endif  /* _INC_STDDEF */
#else
#include <stddef.h>
#endif