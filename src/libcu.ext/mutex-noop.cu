//////////////////////
// MUTEX NOOP
#pragma region MUTEX_NOOP
#ifdef MUTEX_NOOP

#ifndef DEBUG

/*
** Stub routines for all mutex methods.
**
** This routines provide no mutual exclusion or error checking.
*/
static __host__ __device__ int noopMutexInitialize() { return 0; }
static __host__ __device__ int noopMutexShutdown() { return 0; }

static __host__ __device__ mutex *noopMutexAlloc(MUTEX id) { UNUSED_SYMBOL(id); return (mutex *)8;  }
static __host__ __device__ void noopMutexFree(mutex *m) { UNUSED_SYMBOL(m); }
static __host__ __device__ void noopMutexEnter(mutex *m) { UNUSED_SYMBOL(m); }
static __host__ __device__ bool noopMutexTryEnter(mutex *m) { UNUSED_SYMBOL(m); return true; }
static __host__ __device__ void noopMutexLeave(mutex *m) { UNUSED_SYMBOL(m); }

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
__host__ __device__ mutex_methods const *__mutexsystemNoop() { return &noopDefaultMethods; }

#else
/* In this implementation, error checking is provided for testing and debugging purposes.  The mutexes still do not provide any mutual exclusion. */

/* The mutex object */
struct mutex {
	MUTEX Id;	// The mutex type
	int Refs;	// Number of entries without a matching leave
};

/* The mutex_held() and mutex_notheld() routine are intended for use inside assert() statements. */
static __host__ __device__ bool noopMutexHeld(mutex *m) { return (!p || p->Refs); }
static __host__ __device__ bool noopMutexNotheld(mutex *m) { return (!p || !p->Refs); }

/* Initialize and deinitialize the mutex subsystem. */
static __host__ __device__ mutex *noopMutexStatics[MUTEX_STATIC_VFS3 - 1];

static __host__ __device__ int noopMutexInitialize() { return 0; }
static __host__ __device__ int noopMutexEnd() { return 0; }

/* The mutex_alloc() routine allocates a new mutex and returns a pointer to it.  If it returns NULL that means that a mutex could not be allocated. */
static __host__ __device__ mutex *noopMutexAlloc(MUTEX id)
{
	mutex *m;
	switch (id) {
	case MUTEX_FAST:
	case MUTEX_RECURSIVE: {
		m = (mutex *)malloc(sizeof(*m));
		if (m) {
			m->Id = id;
			m->Refs = 0;
		}
		break; }
	default: {
#ifdef ENABLE_API_ARMOR
		if (id-2 < 0 || id-2 >= lengthof(noopMutexStatics)) {
			(void)MISUSE_BKPT;
			return 0;
		}
#endif
		m = &noopMutexStatics[id-2];
		m->Id = id;
		break; }
	}
	return m;
}

/* This routine deallocates a previously allocated mutex. */
static __host__ __device__ void noopMutexFree(mutex *m)
{
	assert(!p->Refs);
	if (p->Id == MUTEX_FAST || p->Id == MUTEX_RECURSIVE) {
		free(m);
	}
	else {
#ifdef ENABLE_API_ARMOR
		(void)MISUSE_BKPT;
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
static __host__ __device__ void noopMutexEnter(mutex *m)
{
	assert(m->Id == MUTEX_RECURSIVE || mutex_notheld(m));
	m->Refs++;
}

static __host__ __device__ bool noopMutexTryEnter(mutex *m)
{
	assert(m->Id == MUTEX_RECURSIVE || mutex_notheld(m));
	m->Refs++;
	return true;
}

/*
** The mutex_leave() routine exits a mutex that was previously entered by the same thread.  The behavior
** is undefined if the mutex is not currently entered or is not currently allocated.  Libcu will never do either.
*/
static __host__ __device__ void noopMutexLeave(mutex *m)
{
	assert(mutex_held(m));
	m->Refs--;
	assert(m->Id == MUTEX_RECURSIVE || mutex_notheld(m));
}

static __host__ __constant__ const mutex_methods noopDefaultMethods = {
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

__host__ __device__ mutex_methods const *__mutexsystemNoop() { return &noopDefaultMethods; }

#endif

/* If compiled with MUTEX_NOOP, then the no-op mutex implementation is used regardless of the run-time threadsafety setting. */
#ifdef MUTEX_NOOP
__host__ __device__ mutex_methods const *__mutexsystemDefault() { return &noopDefaultMethods; }
#endif

#endif
#pragma endregion