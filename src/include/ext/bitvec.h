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

	/* A bitmap is an instance of the following structure. */
	typedef struct bitvec_t bitvec_t;

	/* Create a new bitmap object able to handle bits between 0 and iSize, inclusive.  Return a pointer to the new object.  Return NULL if malloc fails. */
	__device__ bitvec_t *bitvecNew(uint32_t size);
	/* Check to see if the i-th bit is set.  Return true or false. If p is NULL (if the bitmap has not been created) or if i is out of range, then return false. */
	__device__ bool bitvecGet(bitvec_t *b, uint32_t index);
	/* Set the i-th bit.  Return 0 on success and an error code if anything goes wrong. */
	__device__ bool bitvecSet(bitvec_t *b, uint32_t index);
	/* Clear the i-th bit. */
	__device__ void bitvecClear(bitvec_t *b, uint32_t index, void *buffer);
	/* Destroy a bitmap object.  Reclaim all memory used. */
	__device__ void bitvecDestroy(bitvec_t *b);
	/* Return the value of the iSize parameter specified when Bitvec *p was created. */
	__device__ uint32_t bitvecSize(bitvec_t *b);
#ifndef LIBCU_UNTESTABLE
	/* This routine runs an extensive test of the Bitvec code. */
	__device__ int bitvecBuiltinTest(int size, int *ops);
#endif

#ifdef  __cplusplus
}
#endif
#endif	/* _EXT_BITVEC_H */