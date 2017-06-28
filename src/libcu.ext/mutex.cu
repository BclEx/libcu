#include <stdlibcu.h>
#include <ext/mutex.h>
#include <assert.h>
#ifndef MUTEX_OMIT

#include "mutext-noop.cuh"
#include "mutext-win.cuh"
#include "mutext-unix.cuh"

#if defined(DEBUG)
/*
** For debugging purposes, record when the mutex subsystem is initialized and uninitialized so that we can assert() if there is an attempt to
** allocate a mutex while the system is uninitialized.
*/
static __device__ _WSD bool g_mutexIsInit = false;
#endif

/* Initialize the mutex system. */
__device__ int mutexInitialize()
{ 
	// If the MutexAlloc method has not been set, then the user did not install a mutex implementation via sqlite3_config() prior to 
	// systemInitialize() being called. This block copies pointers to the default implementation into the sqlite3GlobalConfig structure.
	if (!__mutexsystem.MutexAlloc) {
		_mutex_methods const *from = (g_RuntimeStatics.CoreMutex ? __mutexsystemDefault() : __mutexsystemNoop());
		_mutex_methods *to = &__mutexsystem;
		to->MutexInitialize = from->MutexInitialize;
		to->MutexShutdown = from->MutexShutdown;
		to->MutexFree = from->MutexFree;
		to->MutexEnter = from->MutexEnter;
		to->MutexTryEnter = from->MutexTryEnter;
		to->MutexLeave = from->MutexLeave;
		to->MutexHeld = from->MutexHeld;
		to->MutexNotheld = from->MutexNotheld;
		mutexMemoryBarrier();
		to->MutexAlloc = from->MutexAlloc;
	}
	assert(__mutexsystem.MutexInitialize);
	int rc = __mutexsystem.MutexInitialize();
#ifdef DEBUG
	_GLOBAL(bool, g_mutexIsInit) = true;
#endif
	return rc;
}

/* Shutdown the mutex system. This call frees resources allocated by mutex_initialize(). */
__device__ int mutexShutdown()
{
	int rc = 0;
	if (__mutexsystem.MutexShutdown)
		rc = __mutexsystem.MutexShutdown();
#ifdef DEBUG
	_GLOBAL(bool, g_mutexIsInit) = false;
#endif
	return rc;
}

/* Retrieve a pointer to a static mutex or allocate a new dynamic one. */
__device__ mutex *mutex_alloc(MUTEX id)
{
#ifndef OMIT_AUTOINIT
	if (id <= MUTEX_RECURSIVE && systemInitialize()) return 0;
	if (id > MUTEX_RECURSIVE && mutexInitialize()) return 0;
#endif
	assert(__mutexsystem.MutexAlloc);
	return __mutexsystem.MutexAlloc(id);
}
__device__ mutex *mutexAlloc(MUTEX id)
{
	if (!g_RuntimeStatics.CoreMutex)
		return nullptr;
	assert(_GLOBAL(bool, g_mutexIsInit));
	return __mutexsystem.MutexAlloc(id);
}

/* Free a dynamic mutex. */
__device__ void mutex_free(mutex *m)
{
	if (m) {
		assert(__mutexsystem.MutexFree);
		__mutexsystem.MutexFree(m);
	}
}

/* Obtain the mutex m. If some other thread already has the mutex, block until it can be obtained. */
__device__ void mutex_enter(mutex *m)
{
	if (m) {
		assert(__mutexsystem.MutexEnter);
		__mutexsystem.MutexEnter(m);
	}
}

/* Obtain the mutex p. If successful, return true. Otherwise, if another thread holds the mutex and it cannot be obtained, return false. */
__device__ bool mutex_tryenter(mutex *m)
{
	if (p) {
		assert(__mutexsystem.MutexTryEnter);
		return __mutexsystem.MutexTryEnter(m);
	}
	return false;
}

/*
** The mutex_leave() routine exits a mutex that was previously entered by the same thread.  The behavior is undefined if the mutex 
** is not currently entered. If a NULL pointer is passed as an argument this function is a no-op.
*/
__device__ void mutex_leave(mutex *m)
{
	if (p) {
		assert(__mutexsystem.MutexLeave);
		__mutexsystem.MutexLeave(p);
	}
}

#ifdef DEBUG
/* The mutex_held() and mutex_notheld() routine are intended for use inside assert() statements. */
__device__ bool mutex_held(mutex *m)
{
	assert(!m || __mutexsystem.MutexHeld);
	return !m || __mutexsystem.MutexHeld(p);
}
__device__ bool mutex_notheld(mutex *m)
{
	assert(!m || __mutexsystem.MutexNotheld);
	return !m || __mutexsystem.MutexNotheld(m);
}
#endif

#endif