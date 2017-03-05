/*
bitvec.h - xxx
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
#ifndef _EXT_BITVEC_H
#define _EXT_BITVEC_H
#include <stdint.h>
#ifdef  __cplusplus
extern "C" {
#endif

	typedef struct bitvec_t bitvec_t;

	__device__ bitvec_t *bitvecNew(uint32_t size);
	__device__ bool bitvecGet(bitvec_t *b, uint32_t index);
	__device__ bool bitvecSet(bitvec_t *b, uint32_t index);
	__device__ void bitvecClear(bitvec_t *b, uint32_t index, void *buffer);
	__device__ void bitvecDestroy(bitvec_t *b);
	__device__ uint32_t bitvecSize(bitvec_t *b);
#ifndef LIBCU_UNTESTABLE
	__device__ int bitvecBuiltinTest(int size, int *ops);
#endif

#ifdef  __cplusplus
}
#endif
#endif	/* _EXT_BITVEC_H */