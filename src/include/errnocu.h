/*
errno.h - Errors
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
#ifndef _ERRNOCU_H
#define _ERRNOCU_H
#include <crtdefscu.h>

#include <errno.h>
#if defined(__CUDA_ARCH__)
__BEGIN_DECLS;

extern __device__ int *_errno_(void);
#define _errno _errno_
#define errno (*_errno())
extern __device__ errno_t _set_errno_(int value);
#define _set_errno _set_errno_
extern __device__ errno_t _get_errno_(int *value);
#define _get_errno _get_errno_

__END_DECLS;
#endif  /* __CUDA_ARCH__ */

#endif  /* _ERRNOCU_H */
