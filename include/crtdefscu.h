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

#define MEMORY_ALIGNMENT 4096
#define ROUNDT(t, x)		(((x)+sizeof(t)-1)&~(sizeof(t)-1))
#define ROUND8(x)			(((x)+7)&~7)
#define ROUNDN(x, size)	(((size_t)(x)+(size-1))&~(size-1))
#define ROUNDDOWN8(x)		((x)&~7)
#define ROUNDDOWNN(x, size) (((size_t)(x))&~(size-1))
#ifdef BYTEALIGNED4
#define HASALIGNMENT8(x) ((((char *)(x) - (char *)0)&3) == 0)
#else
#define HASALIGNMENT8(x) ((((char *)(x) - (char *)0)&7) == 0)
#endif

#include <host_defines.h>

#ifdef  __cplusplus
extern "C" {
#endif

_CRTIMP _CRTNOALIAS void __cdecl free(_Pre_maybenull_ _Post_invalid_ void *_Memory);
_CRTIMP __declspec(noreturn) void __cdecl exit(_In_ int _Code);

#ifdef  __cplusplus
}
#endif

#endif  /* _CRTDEFS_H */