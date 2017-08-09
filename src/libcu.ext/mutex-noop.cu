#include <ext/global.h>
#include <assert.h>

//////////////////////
// MUTEX NOOP
#pragma region MUTEX NOOP
#ifndef LIBCU_MUTEX_OMIT

#ifndef _DEBUG

/*
** Stub routines for all mutex methods.
**
** This routines provide no mutual exclusion or error checking.
*/
static __host_device__ RC noopMutexInitialize() { return 0; }
static __host_device__ RC noopMutexShutdown() { return 0; }

static __host_device__ mutex *noopMutexAlloc(MUTEX id) { UNUSED_SYMBOL(id); return (mutex *)8;  }
static __host_device__ void noopMutexFree(mutex *m) { UNUSED_SYMBOL(m); }
static __host_device__ void noopMutexEnter(mutex *m) { UNUSED_SYMBOL(m); }
static __host_device__ bool noopMutexTryEnter(mutex *m) { UNUSED_SYMBOL(m); return true; }
static __host_device__ void noopMutexLeave(mutex *m) { UNUSED_SYMBOL(m); }

static __host__ __constant__ const mutex_methods noopDefaultMethods = {
	noopMutexInitialize,
	noopMutexShutdown,
	noopMutexAlloc,
	noopMutexFree,
	noopMutexEnter,
	noopMutexTryEnter,
	noopMutexLeave,
	nullptr,
	nullptr,
};
__host_device__ mutex_methods const *__mutexsystemNoop() { return &noopDefaultMethods; }

#else
/* In this implementation, error checking is provided for testing and debugging purposes.  The mutexes still do not provide any mutual exclusion. */

/* The mutex object */
struct mutex {
	MUTEX id;	// The mutex type
	int refs;	// Number of entries without a matching leave
};

/* The mutex_held() and mutex_notheld() routine are intended for use inside assert() statements. */
static __host_device__ bool noopMutexHeld(mutex *m) { return (!m || m->refs); }
static __host_device__ bool noopMutexNotHeld(mutex *m) { return (!m || !m->refs); }

/* Initialize and deinitialize the mutex subsystem. */
static __hostb_device__ mutex noopMutexStatics[MUTEX_STATIC_VFS3 - 1];

static __host_device__ int noopMutexInitialize() { return 0; }
static __host_device__ int noopMutexShutdown() { return 0; }

/* The mutex_alloc() routine allocates a new mutex and returns a pointer to it.  If it returns NULL that means that a mutex could not be allocated. */
static __host_device__ mutex *noopMutexAlloc(MUTEX id)
{
	mutex *m = nullptr;
	switch (id) {
	case MUTEX_FAST:
	case MUTEX_RECURSIVE: {
		m = (mutex *)alloc(sizeof(*m));
		if (m) {
			m->id = id;
			m->refs = 0;
		}
		break; }
	default: {
#ifdef ENABLE_API_ARMOR
		if (id-2 < 0 || id-2 >= _LENGTHOF(noopMutexStatics)) {
			(void)RC_MISUSE_BKPT;
			return 0;
		}
#endif
		m = &noopMutexStatics[id-2];
		m->id = id;
		break; }
	}
	return m;
}

/* This routine deallocates a previously allocated mutex. */
static __host_device__ void noopMutexFree(mutex *m)
{
	assert(!m->refs);
	if (m->id == MUTEX_FAST || m->id == MUTEX_RECURSIVE)
		mfree(m);
	else {
#ifdef ENABLE_API_ARMOR
		(void)RC_MISUSE_BKPT;
#endif
	}
}

/*
** The mutex_enter() and mutex_tryenter() routines attempt to enter a mutex.  If another thread is already within the mutex,
** mutex_enter() will block and mutex_tryenter() will return true.  The mutex_tryenter() interface returns false
** upon successful entry.  Mutexes created using MUTEX_RECURSIVE can be entered multiple times by the same thread.  In such cases the,
** mutex must be exited an equal number of times before another thread can enter.  If the same thread tries to enter any other kind of mutex
** more than once, the behavior is undefined.
*/
static __host_device__ void noopMutexEnter(mutex *m)
{
	assert(m->id == MUTEX_RECURSIVE || mutex_notheld(m));
	m->refs++;
}

static __host_device__ bool noopMutexTryEnter(mutex *m)
{
	assert(m->id == MUTEX_RECURSIVE || mutex_notheld(m));
	m->refs++;
	return true;
}

/*
** The mutex_leave() routine exits a mutex that was previously entered by the same thread.  The behavior
** is undefined if the mutex is not currently entered or is not currently allocated.  Libcu will never do either.
*/
static __host_device__ void noopMutexLeave(mutex *m)
{
	assert(mutex_held(m));
	m->refs--;
	assert(m->id == MUTEX_RECURSIVE || mutex_notheld(m));
}

static __host_constant__ const mutex_methods noopDefaultMethods = {
	noopMutexInitialize,
	noopMutexShutdown,
	noopMutexAlloc,
	noopMutexFree,
	noopMutexEnter,
	noopMutexTryEnter,
	noopMutexLeave,
	noopMutexHeld,
	noopMutexNotHeld
};

__host_device__ mutex_methods const *__mutexsystemNoop() { return &noopDefaultMethods; }

#endif

/* If compiled with MUTEX_NOOP, then the no-op mutex implementation is used regardless of the run-time threadsafety setting. */
#ifdef LIBCU_MUTEX_NOOP
__host_device__ mutex_methods const *__mutexsystemDefault() { return &noopDefaultMethods; }
#endif

#endif
#pragma endregion