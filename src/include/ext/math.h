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
#include <crtdefscu.h>
#include <stdint.h>
__BEGIN_DECLS;

/* Estimated quantities used for query planning are stored as 16-bit logarithms.  For quantity X, the value stored is 10*log2(X).  This
** gives a possible range of values of approximately 1.0e986 to 1e-986. But the allowed values are "grainy".  Not every value is representable.
** For example, quantities 16 and 17 are both represented by a logest_t of 40.  However, since logest_t quantities are suppose to be estimates,
** not exact values, this imprecision is not a problem.
**
** "logest_t" is short for "Logarithmic Estimate".
**
** Examples:
**      1 -> 0              20 -> 43          10000 -> 132
**      2 -> 10             25 -> 46          25000 -> 146
**      3 -> 16            100 -> 66        1000000 -> 199
**      4 -> 20           1000 -> 99        1048576 -> 200
**     10 -> 33           1024 -> 100    4294967296 -> 320
**
** The logest_t can be negative to indicate fractional values.
** Examples:
**
**    0.5 -> -10           0.1 -> -33        0.0625 -> -40
*/
typedef int16_t logest_t;

extern __host_device__ int math_isnan(double x); //: sqlite3IsNaN
extern __host_device__ bool math_add64(int64_t *ao, int64_t b); //: sqlite3AddInt64
extern __host_device__ bool math_sub64(int64_t *ao, int64_t b); //: sqlite3SubInt64
extern __host_device__ bool math_mul64(int64_t *ao, int64_t b); //: sqlite3MulInt64
extern __host_device__ int math_abs32(int x); //: sqlite3AbsInt32
extern __host_device__ logest_t math_addLogest(logest_t a, logest_t b); //: sqlite3LogEstAdd
extern __host_device__ logest_t math_logest(uint64_t x); //: sqlite3LogEst
extern __host_device__ logest_t math_logestFromDouble(double x); //: sqlite3LogEstFromDouble

__END_DECLS;
#endif	/* _EXT_MATH_H */