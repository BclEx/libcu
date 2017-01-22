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

//#pragma once

#ifdef x__CUDA_ARCH__
#ifndef _SYS_BITVECCU_H
#define _SYS_BITVECCU_H

#define BITVEC_SZ 512
#define BITVEC_USIZE (((BITVEC_SZ - (3 * sizeof(uint32))) / sizeof(bitvec_t *)) * sizeof(bitvec_t *))
#define BITVEC_SZELEM 8
#define BITVEC_NELEM (BITVEC_USIZE / sizeof(uint8))
#define BITVEC_NBIT (BITVEC_NELEM * BITVEC_SZELEM)
#define BITVEC_NINT (BITVEC_USIZE / sizeof(uint))
#define BITVEC_MXHASH (BITVEC_NINT / 2)
#define BITVEC_HASH(X) (((X) * 1) % BITVEC_NINT)
#define BITVEC_NPTR (BITVEC_USIZE / sizeof(bitvec_t *))

struct bitvec_t
{
	uint32 _size;      // Maximum bit index.  Max iSize is 4,294,967,296.
	uint32 _set;       // Number of bits that are set - only valid for aHash element.  Max is BITVEC_NINT.  For BITVEC_SZ of 512, this would be 125.
	uint32 _divisor;   // Number of bits handled by each apSub[] entry.
	// Should >=0 for apSub element. */
	// Max iDivisor is max(uint32) / BITVEC_NPTR + 1.
	// For a BITVEC_SZ of 512, this would be 34,359,739.
	union
	{
		uint8 bitmap[BITVEC_NELEM]; // Bitmap representation
		uint32 hash[BITVEC_NINT];	// Hash table representation
		bitvec_t *sub[BITVEC_NPTR];	// Recursive representation
	} u;
};

extern __device__ bitvec_t *bitvecNew(uint32 size);
extern __device__ bool bitvecGet(bitvec_t *p, uint32 index);
extern __device__ bool bitvecSet(bitvec_t *p, uint32 index);
extern __device__ void bitvecClear(bitvec_t *p, uint32 index, void *buffer);
__forceinline __device__ void bitvecDestroy(bitvec_t *p, )
{
	if (!p)
		return;
	if (p->_divisor)
		for (unsigned int index = 0; index < BITVEC_NPTR; index++)
			bitvecDestroy(p->u.sub[index]);
	_free(p);
}
//__forceinline __device__ uint32 bitvecGetLength() { return _size; }

#endif // _SYS_BITVECCU_H
#endif