#include <stdlibcu.h>
#include <ext/mutex.h>
#include <assert.h>
#ifndef MUTEX_OMIT

#if defined(DEBUG)
/*
** For debugging purposes, record when the mutex subsystem is initialized and uninitialized so that we can assert() if there is an attempt to
** allocate a mutex while the system is uninitialized.
*/
static __device__ _WSD bool g_mutexIsInit = false;
#endif

/* Initialize the mutex system. */
__device__ int mutex_initialize()
{ 
	if (!__mutexsystem.Alloc) {
		// If the xMutexAlloc method has not been set, then the user did not install a mutex implementation via sqlite3_config() prior to 
		// sqlite3_initialize() being called. This block copies pointers to the default implementation into the sqlite3GlobalConfig structure.
		_mutex_methods *to = &__mutexsystem;
		_mutex_methods const *from = (__mutexsystem_enabled ? sqlite3DefaultMutex() : sqlite3NoopMutex());
		to->MutexInit = from->MutexInit;
		to->MutexEnd = from->MutexEnd;
		to->MutexFree = from->MutexFree;
		to->MutexEnter = from->MutexEnter;
		to->MutexTry = from->MutexTry;
		to->MutexLeave = from->MutexLeave;
		to->MutexHeld = from->MutexHeld;
		to->MutexNotheld = from->MutexNotheld;
		sqlite3MemoryBarrier();
		to->MutexAlloc = from->MutexAlloc;
		memcpy(to, from, offsetof(_mutex_methods, Alloc));
		memcpy(&to->Free, &from->Free, sizeof(*to) - offsetof(_mutex_methods, Free));
		to->Alloc = from->Alloc;
	}
	int rc = __mutexsystem.Initialize();
#ifdef DEBUG
	_GLOBAL(bool, g_mutexIsInit) = true;
#endif
	return rc;
}

/* Shutdown the mutex system. This call frees resources allocated by mutex_initialize(). */
__device__ int mutex_shutdown()
{
	int rc = 0;
	if (__mutexsystem.Shutdown)
		rc = __mutexsystem.Shutdown();
#ifdef DEBUG
	_GLOBAL(bool, g_mutexIsInit) = false;
#endif
	return rc;
}

/* Retrieve a pointer to a static mutex or allocate a new dynamic one. */
__device__ mutex *mutex_alloc2(MUTEX id)
{
#ifndef OMIT_AUTOINIT
	if (id <= MUTEX_RECURSIVE && system_initialize()) return 0;
	if (id > MUTEX_RECURSIVE && mutex_initialize()) return 0;
#endif
	assert(__mutexsystem.Alloc);
	return __mutexsystem.Alloc(id);
}
__device__ mutex *mutex_alloc(MUTEX id)
{
	if (!__mutexsystem_enabled)
		return nullptr;
	assert(_GLOBAL(bool, g_mutexIsInit));
	return __mutexsystem.Alloc(id);
}




// Free a dynamic mutex.
__device__ void mutex_free(mutex *m)
{
	if (p) __mutexsystem.Free(p);
}

// Obtain the mutex p. If some other thread already has the mutex, block until it can be obtained.
__device__ void _mutex_enter(MutexEx p) { if (p) __mutexsystem.Enter(p); }

// Obtain the mutex p. If successful, return SQLITE_OK. Otherwise, if another thread holds the mutex and it cannot be obtained, return SQLITE_BUSY.
__device__ bool _mutex_tryenter(MutexEx p) { return !p || __mutexsystem.TryEnter(p); }

// The sqlite3_mutex_leave() routine exits a mutex that was previously entered by the same thread.  The behavior is undefined if the mutex 
// is not currently entered. If a NULL pointer is passed as an argument this function is a no-op.
__device__ void _mutex_leave(MutexEx p) { if (p) __mutexsystem.Leave(p); }

#ifdef _DEBUG
// The sqlite3_mutex_held() and sqlite3_mutex_notheld() routine are intended for use inside assert() statements.
__device__ bool _mutex_held(MutexEx p) { return !p || __mutexsystem.Held(p); }
__device__ bool _mutex_notheld(MutexEx p) { return !p || __mutexsystem.Notheld(p); }
#endif

RUNTIME_NAMEEND
#endif