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
#define THREADSAFE 0

	/*
	** Figure out what version of the code to use.  The choices are
	**
	**   MUTEX_OMIT         No mutex logic.  Not even stubs.  The mutexes implementation cannot be overridden at start-time.
	**
	**   MUTEX_NOOP         For single-threaded applications.  No mutual exclusion is provided.  But this
	**                             implementation can be overridden at start-time.
	**
	**   MUTEX_PTHREADS     For multi-threaded applications on Unix.
	**
	**   MUTEX_W32          For multi-threaded applications on Win32.
	*/
#if !THREADSAFE
#define MUTEX_OMIT
#endif
#if THREADSAFE && !defined(MUTEX_NOOP)
#if OS_UNIX
# define MUTEX_PTHREADS
#elif OS_WIN
# define MUTEX_W32
#else
# define MUTEX_NOOP
#endif
#endif

	// CAPI3REF: Mutex Handle
	typedef struct mutex mutex;

	// CAPI3REF: Mutex Methods Object
	typedef struct mutex_methods mutex_methods;
	struct mutex_methods {
		int (*MutexInitialize)();
		int (*MutexShutdown)();
		mutex *(*MutexAlloc)(int);
		void (*MutexFree)(mutex *);
		void (*MutexEnter)(mutex *);
		int (*MutexTryEnter)(mutex *);
		void (*MutexLeave)(mutex *);
		int (*MutexHeld)(mutex *);
		int (*MutexNotheld)(mutex *);
	};
	//?__device__ void __mutexsystem_setdefault(); // Default mutex interface
#define __mutexsystem g_runtimeStatics.MutexSystem

	// CAPI3REF: Mutex Types
#define MUTEX unsigned char
#define MUTEX_FAST             0
#define MUTEX_RECURSIVE        1
#define MUTEX_STATIC_MASTER    2
#define MUTEX_STATIC_MEM       3  // sqlite3_malloc()
#define MUTEX_STATIC_MEM2      4  // NOT USED
#define MUTEX_STATIC_OPEN      4  // sqlite3BtreeOpen()
#define MUTEX_STATIC_PRNG      5  // sqlite3_randomness()
#define MUTEX_STATIC_LRU       6  // lru page list
#define MUTEX_STATIC_LRU2      7  // NOT USED
#define MUTEX_STATIC_PMEM      7  // sqlite3PageMalloc()
#define MUTEX_STATIC_APP1      8  // For use by application
#define MUTEX_STATIC_APP2      9  // For use by application
#define MUTEX_STATIC_APP3     10  // For use by application
#define MUTEX_STATIC_VFS1     11  // For use by built-in VFS
#define MUTEX_STATIC_VFS2     12  // For use by extension VFS
#define MUTEX_STATIC_VFS3     13  // For use by application VFS

#ifdef MUTEX_OMIT
	/*
	** If this is a no-op implementation, implement everything as macros.
	*/
#define mutex_alloc(m) ((mutex *)8)
#define mutex_free(m)
#define mutex_enter(m)
#define mutex_tryenter(m) false
#define mutex_leave(m)
#ifdef DEBUG
#define mutex_held(m) ((void)(m), 1)
#define mutex_notheld(m) ((void)(m), 1)
#endif
#define mutexAlloc(m) ((mutex *)8)
#define mutexInitialize() 0
#define mutexShutdown() 0
#define MUTEX_LOGIC(X)
#else
#define MUTEX_LOGIC(X) X
#endif

#ifdef  __cplusplus
}
#endif
#endif	/* _EXT_MUTEX_H */