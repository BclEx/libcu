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

//#pragma once
#ifndef _STDDEFCU_H
#define _STDDEFCU_H

#include <stddef.h>
#include <crtdefscu.h>

_Check_return_opt_ _CRTIMP int __cdecl printf(_In_z_ _Printf_format_string_ const char *_Format, ...);
#if defined(__CUDA_ARCH__)
#define panic(fmt, ...) { printf(fmt"\n", __VA_ARGS__); asm("trap;"); }
/* Define tag allocs */
__forceinline __device__ void *tagalloc(void *tag, size_t size) { return nullptr; }
__forceinline __device__ void tagfree(void *tag, void *p) { }
__forceinline __device__ void *tagrealloc(void *tag, void *old, size_t size) { return nullptr; }
#else
//__forceinline void Coverage(int line) { }
#define panic(fmt, ...) { printf(fmt"\n", __VA_ARGS__); exit(1); }
/* Define tag allocs */
__forceinline void *tagalloc(void *tag, size_t size) { return nullptr; }
__forceinline void tagfree(void *tag, void *p) { }
__forceinline void *tagrealloc(void *tag, void *old, size_t size) { return nullptr; }
#endif  /* __CUDA_ARCH__ */

#endif  /* _STDDEFCU_H */