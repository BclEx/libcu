/*
time.h - Date and time
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
#ifndef _TIMECU_H
#define _TIMECU_H
#include <featurescu.h>

#include <time.h>
#if defined(__CUDA_ARCH__) || defined(LIBCUFORCE)
__BEGIN_DECLS;

typedef long clock_t;
#define CLOCKS_PER_SEC 1000
struct timeval { long tv_sec; long tv_usec; };

__device__ time_t time_(time_t *timer);
#define time time_
__device__ int gettimeofday_(struct timeval *tp, void *tz);
#define gettimeofday gettimeofday_

__END_DECLS;
#else
#ifndef _WINSOCKAPI_
struct timeval { long tv_sec; long tv_usec; };
#endif
__BEGIN_DECLS;
int gettimeofday(struct timeval *tv, void *unused);
__END_DECLS;
//#define gettimeofday(tp, tz) 0
#endif  /* __CUDA_ARCH__ */

#endif  /* _TIMECU_H */
