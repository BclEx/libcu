#include <ext/mutex.h> //: mutex.c
#include <assert.h>
#ifndef LIBCU_MUTEX_OMIT

#if defined(_DEBUG)
/* For debugging purposes, record when the mutex subsystem is initialized and uninitialized so that we can assert() if there is an attempt to
** allocate a mutex while the system is uninitialized.
*/
static __hostb_device__ _WSD bool _mutexIsInit = false;
#endif

/* Initialize the mutex system. */
__host_device__ RC mutexInitialize() //: sqlite3MutexInit
{ 
	// If the _.alloc method has not been set, then the user did not install a mutex implementation via sqlite3_config() prior to 
	// systemInitialize() being called. This block copies pointers to the default implementation into the sqlite3GlobalConfig structure.
	if (!__mutexsystem.alloc) {
		mutex_methods const *from = (_runtimeConfig.coreMutex ? __mutexsystemDefault() : __mutexsystemNoop());
		mutex_methods *to = &__mutexsystem;
		to->initialize = from->initialize;
		to->shutdown = from->shutdown;
		to->free = from->free;
		to->enter = from->enter;
		to->tryEnter = from->tryEnter;
		to->leave = from->leave;
		to->held = from->held;
		to->notheld = from->notheld;
		systemMemoryBarrier();
		to->alloc = from->alloc;
	}
	assert(__mutexsystem.initialize);
	RC rc = __mutexsystem.initialize();
#ifdef _DEBUG
	_GLOBAL(bool, _mutexIsInit) = true;
#endif
	return rc;
}

/* Shutdown the mutex system. This call frees resources allocated by mutex_initialize(). */
__host_device__ RC mutexShutdown() //: sqlite3MutexEnd
{
	RC rc = 0;
	if (__mutexsystem.shutdown)
		rc = __mutexsystem.shutdown();
#ifdef _DEBUG
	_GLOBAL(bool, _mutexIsInit) = false;
#endif
	return rc;
}

/* Retrieve a pointer to a static mutex or allocate a new dynamic one. */
__host_device__ mutex *mutex_alloc(MUTEX id) //: sqlite3_mutex_alloc
{
#ifndef OMIT_AUTOINIT
	if (id <= MUTEX_RECURSIVE && runtimeInitialize()) return nullptr;
	if (id > MUTEX_RECURSIVE && mutexInitialize()) return nullptr;
#endif
	assert(__mutexsystem.alloc);
	return __mutexsystem.alloc(id);
}
__host_device__ mutex *mutexAlloc(MUTEX id) //: sqlite3MutexAlloc
{
	if (!_runtimeConfig.coreMutex)
		return nullptr;
	assert(_GLOBAL(bool, _mutexIsInit));
	return __mutexsystem.alloc(id);
}

/* Free a dynamic mutex. */
__host_device__ void mutex_free(mutex *m) //: sqlite3_mutex_free
{
	if (m) {
		assert(__mutexsystem.free);
		__mutexsystem.free(m);
	}
}

/* Obtain the mutex m. If some other thread already has the mutex, block until it can be obtained. */
__host_device__ void mutex_enter(mutex *m) //: sqlite3_mutex_enter
{
	if (m) {
		assert(__mutexsystem.enter);
		__mutexsystem.enter(m);
	}
}

/* Obtain the mutex p. If successful, return true. Otherwise, if another thread holds the mutex and it cannot be obtained, return false. */
__host_device__ bool mutex_tryenter(mutex *m) //: sqlite3_mutex_try
{
	if (m) {
		assert(__mutexsystem.tryEnter);
		return __mutexsystem.tryEnter(m);
	}
	return true;
}

/* The mutex_leave() routine exits a mutex that was previously entered by the same thread.  The behavior is undefined if the mutex 
** is not currently entered. If a NULL pointer is passed as an argument this function is a no-op.
*/
__host_device__ void mutex_leave(mutex *m) //: sqlite3_mutex_leave
{
	if (m) {
		assert(__mutexsystem.leave);
		__mutexsystem.leave(m);
	}
}

#ifdef _DEBUG
/* The mutex_held() and mutex_notheld() routine are intended for use inside assert() statements. */
__host_device__ bool mutex_held(mutex *m) //: sqlite3_mutex_held
{
	assert(!m || __mutexsystem.held);
	return !m || __mutexsystem.held(m);
}
__host_device__ bool mutex_notheld(mutex *m) //: sqlite3_mutex_notheld
{
	assert(!m || __mutexsystem.notheld);
	return !m || __mutexsystem.notheld(m);
}
#endif

#endif