/*
tagbase.h - xxx
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
#ifndef _EXT_TAGBASE_H
#define _EXT_TAGBASE_H
#include <featurescu.h>
#include <malloc.h>
__BEGIN_DECLS;

__forceinline __device__ void *allocaZero(size_t size) { void *p = alloca(size); if (p) memset(p, 0, size); return p; }

/*
** On systems with ample stack space and that support alloca(), make use of alloca() to obtain space for large automatic objects.  By default,
** obtain space from malloc().
**
** The alloca() routine never returns NULL.  This will cause code paths that deal with sqlite3StackAlloc() failures to be unreachable.
*/
#ifdef LIBCU_ALLOCA
#define tagalloca(tag, size) alloca(size)
#define tagallocaZero(tag, size) allocaZero(size)
#define tagfreea(tag, ptr)
#elif 0
#define tagalloca(tag, size) tagalloc(tag, size)
#define tagallocaZero(tag, size) tagallocZero(tag, size)
#define tagfreea(tag, ptr) tagfree(tag, ptr)
#else
#define tagalloca(tag, size) malloc(size)
#define tagallocaZero(tag, size) mallocZero(size)
#define tagfreea(tag, ptr) free(ptr)
#endif

__END_DECLS;
#endif  /* _EXT_TAGBASE_H */