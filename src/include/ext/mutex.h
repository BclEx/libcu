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

#include <ext\global.h>
#ifndef _EXT_MUTEX_H
#define _EXT_MUTEX_H
#ifdef  __cplusplus
extern "C" {
#endif

	/*
	** The LIBCU_THREADSAFE macro must be defined as 0, 1, or 2. 0 means mutexes are permanently disable and the library is never
	** threadsafe.  1 means the library is serialized which is the highest level of threadsafety.  2 means the library is multithreaded - multiple
	** threads can use SQLite as long as no two threads try to use the same database connection at the same time.
	**
	** Older versions of SQLite used an optional THREADSAFE macro. We support that for legacy.
	*/
#if !defined(LIBCU_THREADSAFE)
#if defined(THREADSAFE)
#define LIBCU_THREADSAFE THREADSAFE
#else
#define LIBCU_THREADSAFE 1 // IMP: R-07272-22309
#endif
#endif

	/*
	** Figure out what version of the code to use.  The choices are
	**
	**   LIBCU_MUTEX_OMIT         No mutex logic.  Not even stubs.  The mutexes implementation cannot be overridden at start-time.
	**
	**   LIBCU_MUTEX_NOOP         For single-threaded applications.  No mutual exclusion is provided.  But this implementation can be overridden at start-time.
	**
	**   LIBCU_MUTEX_GPU          For single-threaded applications on Gpu.
	**
	**   LIBCU_MUTEX_PTHREADS     For multi-threaded applications on Unix.
	**
	**   LIBCU_MUTEX_W32          For multi-threaded applications on Win32.
	*/
#if !LIBCU_THREADSAFE
#define LIBCU_MUTEX_OMIT
#endif
#if LIBCU_THREADSAFE && !defined(LIBCU_MUTEX_NOOP)
# if OS_GPU
# define LIBCU_MUTEX_GPU
# elif OS_UNIX
# define LIBCU_MUTEX_PTHREADS
# elif OS_WIN
# define LIBCU_MUTEX_W32
# else
# define LIBCU_MUTEX_NOOP
# endif
#endif

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

	//?__device__ void __mutexsystem_setdefault(); // Default mutex interface
	// CAPI3REF: Mutex Handle
	typedef struct mutex mutex;

	// CAPI3REF: Mutexes
	__host_device__ mutex *mutex_alloc(MUTEX id);
	__host_device__ void mutex_free(mutex *m);
	__host_device__ void mutex_enter(mutex *m);
	__host_device__ bool mutex_tryenter(mutex *m);
	__host_device__ void mutex_leave(mutex *m);

	// CAPI3REF: Mutex Methods Object
	typedef struct mutex_methods mutex_methods;
	struct mutex_methods {
		RC (*initialize)();
		RC (*shutdown)();
		mutex *(*alloc)(int);
		void (*free)(mutex *);
		void (*enter)(mutex *);
		bool (*tryEnter)(mutex *);
		void (*leave)(mutex *);
		bool (*held)(mutex *);
		bool (*notheld)(mutex *);
	};
#define __mutexsystem _runtimeStatics.mutexSystem

	// CAPI3REF: Mutex Verification Routines
#ifndef NDEBUG
	__host_device__ bool mutex_held(mutex *m);
	__host_device__ bool mutex_notheld(mutex *m);
#endif

#ifndef LIBCU_MUTEX_OMIT
	__host_device__ mutex_methods const *__mutexsystemDefault();
	__host_device__ mutex_methods const *__mutexsystemNoop();
	__host_device__ mutex *mutexAlloc(MUTEX id);
	__host_device__ RC mutexInitialize();
	__host_device__ RC mutexShutdown();
#endif
#if !defined(LIBCU_MUTEX_OMIT) && !defined(LIBCU_MUTEX_NOOP)
	__host_device__ void systemMemoryBarrier();
#else
#define systemMemoryBarrier()
#endif

#ifdef LIBCU_MUTEX_OMIT
	/* If this is a no-op implementation, implement everything as macros. */
#define mutex_alloc(id) ((mutex *)8)
#define mutex_free(m)
#define mutex_enter(m)
#define mutex_tryenter(m) true
#define mutex_leave(m)
#define mutex_held(m) ((void)(m), true)
#define mutex_notheld(m) ((void)(m), true)
#define mutexAlloc(id) ((mutex *)8)
#define mutexInitialize() RC_OK
#define mutexShutdown() RC_OK
#define MUTEX_LOGIC(X)
#else
#define MUTEX_LOGIC(X) X
#endif

#ifdef  __cplusplus
}
#endif
#endif	/* _EXT_MUTEX_H */