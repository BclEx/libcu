/*
math.h - xxx
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
#ifndef _EXT_MATH_H
#define _EXT_MATH_H
#include <stdint.h>
#ifdef  __cplusplus
extern "C" {
#endif

	extern __device__ bool math_add(int64_t *aRef, int64_t b);
	extern __device__ bool math_sub(int64_t *aRef, int64_t b);
	extern __device__ bool math_mul(int64_t *aRef, int64_t b);
#ifndef OMIT_INLINEMATH
	__forceinline __device__ int math_abs(int x)
	{
		if (x >= 0) return x;
		if (x == (int)0x8000000) return 0x7fffffff;
		return -x;
	}
#else
	extern __device__ int math_abs(int x);
#endif

#ifdef  __cplusplus
}
#endif
#endif	/* _EXT_MATH_H */