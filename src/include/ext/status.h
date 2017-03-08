/*
status.h - xxx
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
#ifndef _EXT_STATUS_H
#define _EXT_STATUS_H
#include <stdint.h>
#ifdef  __cplusplus
extern "C" {
#endif

	enum STATUS : unsigned char
	{
		STATUS_MEMORY_USED = 0,
		STATUS_PAGECACHE_USED = 1,
		STATUS_PAGECACHE_OVERFLOW = 2,
		STATUS_SCRATCH_USED = 3,
		STATUS_SCRATCH_OVERFLOW = 4,
		STATUS_MALLOC_SIZE = 5,
		STATUS_PARSER_STACK = 6,
		STATUS_PAGECACHE_SIZE = 7,
		STATUS_SCRATCH_SIZE = 8,
		STATUS_MALLOC_COUNT = 9,
	};

	extern __device__ int status_value(STATUS op);
	extern __device__ void status_add(STATUS op, int n);
	extern __device__ void status_set(STATUS op, int x);
	extern __device__ bool status(STATUS op, int *current, int *highwater, bool resetFlag);

#ifdef  __cplusplus
}
#endif
#endif	/* _EXT_STATUS_H */