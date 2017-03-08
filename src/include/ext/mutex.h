/*
mutex.h - xxx
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
#ifndef _EXT_MUTEX_H
#define _EXT_MUTEX_H
#ifdef  __cplusplus
extern "C" {
#endif

#if !defined(THREADSAFE)
#if defined(__THREADSAFE__)
#define THREADSAFE __THREADSAFE__
#else
#define THREADSAFE 0 // IMP: R-07272-22309
#endif
#endif

	// Figure out what version of the code to use.  The choices are
	//   MUTEX_OMIT         No mutex logic.  Not even stubs.  The mutexes implemention cannot be overridden at start-time.
	//   MUTEX_NOOP         For single-threaded applications.  No mutual exclusion is provided.  But this implementation can be overridden at start-time.
	//   MUTEX_PTHREADS     For multi-threaded applications on Unix.
	//   MUTEX_W32          For multi-threaded applications on Win32.
#if THREADSAFE == 0
#define MUTEX_OMIT
#else
#if OS_GPU
#define MUTEX_NOOP
#elif OS_UNIX
#define MUTEX_PTHREADS
#elif OS_WIN
#define MUTEX_WIN
#else
#define MUTEX_NOOP
#endif
#endif

	enum MUTEX : unsigned char
	{
		MUTEX_FAST = 0,
		MUTEX_RECURSIVE = 1,
		MUTEX_STATIC_MASTER = 2,
		MUTEX_STATIC_MEM = 3,  // sqlite3_malloc()
		MUTEX_STATIC_OPEN = 4,  // sqlite3BtreeOpen()
		MUTEX_STATIC_PRNG = 5,  // sqlite3_random()
		MUTEX_STATIC_LRU = 6,   // lru page list
		MUTEX_STATIC_PMEM = 7, // sqlite3PageMalloc()
	};

#ifdef MUTEX_OMIT
	typedef void *MutexEx;
#define mutex_held(X) ((void)(X), 1)
#define mutex_notheld(X) ((void)(X), 1)
#define mutex_init() 0
#define mutex_shutdown()
#define mutex_alloc(X) ((MutexEx)1)
#define mutex_free(X)
#define mutex_enter(X)
#define mutex_tryenter(X) 0
#define mutex_leave(X)
#define MUTEX_LOGIC(X)
#else
	struct mutex_obj;
	typedef mutex_obj *MutexEx;
#ifdef _DEBUG
	__device__ bool mutex_held(MutexEx p);
	__device__ bool mutex_notheld(MutexEx p);
#endif
	__device__ int mutex_init();
	__device__ void mutex_shutdown();
	__device__ MutexEx mutex_alloc(MUTEX id);
	__device__ void mutex_free(MutexEx p);
	__device__ void mutex_enter(MutexEx p);
	__device__ bool mutex_tryenter(MutexEx p);
	__device__ void mutex_leave(MutexEx p);
#define MUTEX_LOGIC(X) X
#endif

#ifdef  __cplusplus
}
#endif
#endif	/* _EXT_MUTEX_H */