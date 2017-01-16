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

#include <host_definescu.h>

#ifdef __CUDA_ARCH__
#ifndef _CRTDEFSCU_H
#define _CRTDEFS_H

#define _INC_VADEFS
#include <crtdefs.h>
#define _INC_CRTDEFS

#define _INC_SWPRINTF_INL_
#include <stddefcu.h>

#endif  /* _CRTDEFSCU_H */
#else
#include <crtdefs.h>
#endif

#ifndef _CRTDEFSCU_H
#ifdef  __cplusplus
extern "C" {
#endif

	/* Built In */
	_CRTIMP _CRTNOALIAS void __cdecl free(_Pre_maybenull_ _Post_invalid_ void *_Memory);
	_Check_return_ _Ret_maybenull_ _Post_writable_byte_size_(_Size) _CRTIMP _CRT_JIT_INTRINSIC _CRTNOALIAS _CRTRESTRICT void * __cdecl malloc(_In_ size_t _Size);
	_CRTIMP __declspec(noreturn) void __cdecl exit(_In_ int _Code);
	//_Check_return_opt_ _CRTIMP int __cdecl printf(_In_z_ _Printf_format_string_ const char *_Format, ...);
	//void __cdecl free(void *memory);
	//void * __cdecl malloc(size_t size);
	//__declspec(noreturn) void __cdecl exit(int code);
	//int __cdecl printf(const char *format, ...);

#ifdef  __cplusplus
}
#endif
#endif
#define _CRTDEFSCU_H