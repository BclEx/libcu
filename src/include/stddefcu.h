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

#if !__CUDACC__
#include <stddef.h>
#elif !defined(_INC_STDDEFCU)
#define _INC_STDDEFCU
#include <crtdefscu.h>

//#define _CRTIMP
//#define _In_
//#define _Out_

#ifdef  __cplusplus
extern "C" {
#endif

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

#endif  /* _INC_STDDEFCU */