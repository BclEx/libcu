/*
memfile.h - xxx
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

#ifndef _EXT_MEMFILE_H
#define _EXT_MEMFILE_H
#include <stdint.h>
#ifdef  __cplusplus
extern "C" {
#endif

	typedef struct memfile_t memfile_t;

	extern __constant__ int __sizeofMemfile_t;
	extern __device__ void memfileOpen(memfile_t *f);
	extern __device__ void memfileRead(memfile_t *f, void *buffer, int amount, int64_t offset);
	extern __device__ bool memfileWrite(memfile_t *f, const void *buffer, int amount, int64_t offset);
	extern __device__ void memfileTruncate(memfile_t *f, int64_t size);
	extern __device__ void memfileClose(memfile_t *f);
	extern __device__ int64_t memfileGetFileSize(memfile_t *f);

#ifdef  __cplusplus
}
#endif
#endif  /* _EXT_MEMFILE_H */