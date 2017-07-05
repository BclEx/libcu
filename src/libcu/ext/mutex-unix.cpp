#include <ext/global.h>

//////////////////////
// MUTEX PTHREADS
#pragma region MUTEX PTHREADS
#ifdef LIBCU_MUTEX_PTHREADS

#include <pthread.h>

/*
** The mutex.Id, mutex.Refs, and mutex.Owner fields are necessary under two condidtions:  (1) Debug builds and (2) using
** home-grown mutexes.  Encapsulate these conditions into a single #define.
*/
#if defined(_DEBUG) || defined(HOMEGROWN_RECURSIVE_MUTEX)
#define MUTEX_NREF 1
#else
#define MUTEX_NREF 0
#endif

/*
** Each recursive mutex is an instance of the following structure.
*/
struct mutex {
	pthread_mutex_t Mutex;		// Mutex controlling the lock
#if MUTEX_NREF || defined(ENABLE_API_ARMOR)
	int Id;						// Mutex type
#endif
#if MUTEX_NREF
	volatile int Refs;			// Number of entrances
	volatile pthread_t Owner;	// Thread that is within this mutex
	bool Trace;					// True to trace changes
#endif
};

#if MUTEX_NREF
#define MUTEX_INITIALIZER { PTHREAD_MUTEX_INITIALIZER, 0, 0, (pthread_t)nullptr, 0 }
#elif defined(ENABLE_API_ARMOR)
#define MUTEX_INITIALIZER { PTHREAD_MUTEX_INITIALIZER, 0 }
#else
#define MUTEX_INITIALIZER { PTHREAD_MUTEX_INITIALIZER }
#endif
static mutex MutexStatics[] = {
	MUTEX_INITIALIZER,
	MUTEX_INITIALIZER,
	MUTEX_INITIALIZER,
	MUTEX_INITIALIZER,
	MUTEX_INITIALIZER,
	MUTEX_INITIALIZER,
	MUTEX_INITIALIZER,
	MUTEX_INITIALIZER,
	MUTEX_INITIALIZER,
	MUTEX_INITIALIZER,
	MUTEX_INITIALIZER,
	MUTEX_INITIALIZER
};
#undef MUTEX_INITIALIZER

/*
** The mutex_held() and mutex_notheld() routine are intended for use only inside assert() statements.  On some platforms,
** there might be race conditions that can cause these routines to deliver incorrect results.  In particular, if pthread_equal() is
** not an atomic operation, then these routines might delivery incorrect results.  On most platforms, pthread_equal() is a 
** comparison of two integers and is therefore atomic.  But we are told that HPUX is not such a platform.  If so, then these routines
** will not always work correctly on HPUX.
**
** On those platforms where pthread_equal() is not atomic, Libcu should be compiled without -DDEBUG and with -DNDEBUG to
** make sure no assert() statements are evaluated and hence these routines are never called.
*/
#if !defined(NDEBUG) || defined(_DEBUG)
static bool MutexHeld(mutex *m) { return m->Refs && pthread_equal(m->Owner, pthread_self()); }
static bool MutexNotHeld(mutex *m) { return !m->Refs || !pthread_equal(m->Owner, pthread_self()); }
#endif

/*
** Try to provide a memory barrier operation, needed for initialization and also for the implementation of xShmBarrier in the VFS in cases
** where Libcu is compiled without mutexes.
*/
void mutexMemoryBarrier()
{
#if defined(MEMORY_BARRIER)
	MEMORY_BARRIER;
#elif defined(__GNUC__) && GCC_VERSION>=4001000
	__sync_synchronize();
#endif
}

/*
** Initialize and deinitialize the mutex subsystem.
*/
static int MutexInitialize() { return 0; }
static int MutexShutdown() { return 0; }

/*
** The mutex_alloc() routine allocates a new mutex and returns a pointer to it.  If it returns NULL
** that means that a mutex could not be allocated.  Libcu will unwind its stack and return an error.  The argument
** to mutex_alloc() is one of these integer constants:
**
** <ul>
** <li>  MUTEX_FAST
** <li>  MUTEX_RECURSIVE
** <li>  MUTEX_STATIC_MASTER
** <li>  MUTEX_STATIC_MEM
** <li>  MUTEX_STATIC_OPEN
** <li>  MUTEX_STATIC_PRNG
** <li>  MUTEX_STATIC_LRU
** <li>  MUTEX_STATIC_PMEM
** <li>  MUTEX_STATIC_APP1
** <li>  MUTEX_STATIC_APP2
** <li>  MUTEX_STATIC_APP3
** <li>  MUTEX_STATIC_VFS1
** <li>  MUTEX_STATIC_VFS2
** <li>  MUTEX_STATIC_VFS3
** </ul>
**
** The first two constants cause mutex_alloc() to create a new mutex.  The new mutex is recursive when MUTEX_RECURSIVE
** is used but not necessarily so when MUTEX_FAST is used. The mutex implementation does not need to make a distinction
** between MUTEX_RECURSIVE and MUTEX_FAST if it does not want to.  But Libcu will only request a recursive mutex in
** cases where it really needs one.  If a faster non-recursive mutex implementation is available on the host platform, the mutex subsystem
** might return such a mutex in response to MUTEX_FAST.
**
** The other allowed parameters to mutex_alloc() each return a pointer to a static preexisting mutex.  Six static mutexes are
** used by the current version of Libcu.  Future versions of Libcu may add additional static mutexes.  Static mutexes are for internal
** use by Libcu only.  Applications that use Libcu mutexes should use only the dynamic mutexes returned by MUTEX_FAST or
** MUTEX_RECURSIVE.
**
** Note that if one of the dynamic mutex parameters (MUTEX_FAST or MUTEX_RECURSIVE) is used then mutex_alloc()
** returns a different mutex on every call.  But for the static mutex types, the same mutex is returned on every call that has
** the same type number.
*/
static mutex *MutexAlloc(MUTEX id)
{
	mutex *m;
	switch (id) {
	case MUTEX_RECURSIVE: {
		m = (mutex *)allocZero(sizeof(*m));
		if (m) {
#ifdef HOMEGROWN_RECURSIVE_MUTEX
			// If recursive mutexes are not available, we will have to build our own.  See below.
			pthread_mutex_init(&m->Mutex, 0);
#else
			// Use a recursive mutex if it is available
			pthread_mutexattr_t recursiveAttr;
			pthread_mutexattr_init(&recursiveAttr);
			pthread_mutexattr_settype(&recursiveAttr, PTHREAD_MUTEX_RECURSIVE);
			pthread_mutex_init(&m->mutex, &recursiveAttr);
			pthread_mutexattr_destroy(&recursiveAttr);
#endif
		}
		break; }
	case MUTEX_FAST: {
		m = (mutex *)allocZero(sizeof(*m));
		if (m)
			pthread_mutex_init(&m->Mutex, 0);
		break; }
	default: {
#ifdef ENABLE_API_ARMOR
		if (id-2 < 0 || id-2 >= lengthof(MutexStatics)) {
			(void)MISUSE_BKPT;
			return 0;
		}
#endif
		m = &MutexStatics[id-2];
		break; }
	}
#if MUTEX_NREF || defined(ENABLE_API_ARMOR)
	if (m) m->Id = id;
#endif
	return m;
}

/*
** This routine deallocates a previously allocated mutex.  Libcu is careful to deallocate every mutex that it allocates.
*/
static void MutexFree(mutex *m)
{
	assert(!m->Refs);
#if ENABLE_API_ARMOR
	if (m->Id == MUTEX_FAST || m->Id == MUTEX_RECURSIVE)
#endif
	{
		pthread_mutex_destroy(&m->Mutex);
		free(p);
	}
#ifdef ENABLE_API_ARMOR
	else {
		(void)MISUSE_BKPT;
	}
#endif

}

/*
** The mutex_enter() and mutex_tryenter() routines attempt to enter a mutex.  If another thread is already within the mutex,
** mutex_enter() will block and mutex_tryenter() will return true.  The mutex_tryenter() interface returns false
** upon successful entry.  Mutexes created using MUTEX_RECURSIVE can be entered multiple times by the same thread.  In such cases the,
** mutex must be exited an equal number of times before another thread can enter.  If the same thread tries to enter any other kind of mutex
** more than once, the behavior is undefined.
*/
static void MutexEnter(mutex *m)
{
	assert(m->Id == MUTEX_RECURSIVE || MutexNotheld(m));
#ifdef HOMEGROWN_RECURSIVE_MUTEX
	// If recursive mutexes are not available, then we have to grow our own.  This implementation assumes that pthread_equal()
	// is atomic - that it cannot be deceived into thinking self and m->Owner are equal if m->Owner changes between two values
	// that are not equal to self while the comparison is taking place. This implementation also assumes a coherent cache - that 
	// separate processes cannot read different values from the same address at the same time.  If either of these two conditions
	// are not met, then the mutexes will fail and problems will result.
	{
		pthread_t self = pthread_self();
		if (p->Refs && pthread_equal(p->Owner, self))
			p->Refs++;
		else {
			pthread_mutex_lock(&m->Mutex);
			assert(!m->Refs);
			m->Owner = self;
			m->Refs = 1;
		}
	}
#else
	// Use the built-in recursive mutexes if they are available.
	pthread_mutex_lock(&m->Mutex);
#if MUTEX_NREF
	assert(m->Refs || !m->Owner);
	m->Owner = pthread_self();
	m->Refs++;
#endif
#endif
#ifdef _DEBUG
	if (m->Trace)
		printf("enter mutex %p (%d) with Refs=%d\n", m, m->Trace, m->Refs);
#endif
}

static bool MutexTryEnter(mutex *m)
{
	assert(m->Id == MUTEX_RECURSIVE || MutexNotheld(m));
	bool rc;
#ifdef HOMEGROWN_RECURSIVE_MUTEX
	// If recursive mutexes are not available, then we have to grow our own.  This implementation assumes that pthread_equal()
	// is atomic - that it cannot be deceived into thinking self and m->Owner are equal if m->Owner changes between two values
	// that are not equal to self while the comparison is taking place. This implementation also assumes a coherent cache - that 
	// separate processes cannot read different values from the same address at the same time.  If either of these two conditions
	// are not met, then the mutexes will fail and problems will result.
	{
		pthread_t self = pthread_self();
		if (m->Refs && pthread_equal(m->Owner, self)) {
			m->Refs++;
			rc = false;
		}
		else if (!pthread_mutex_trylock(&m->Mutex)) {
			assert(!m->Refs);
			m->Owner = self;
			m->Refs = 1;
			rc = false;
		}
		else rc = true;
	}
#else
	// Use the built-in recursive mutexes if they are available.
	if (!pthread_mutex_trylock(&m->Mutex)) {
#if MUTEX_NREF
		m->Owner = pthread_self();
		m->Refs++;
#endif
		rc = false;
	}
	else rc = true;
#endif

#ifdef _DEBUG
	if (!rc && m->Trace)
		printf("enter mutex %p (%d) with Refs=%d\n", m, m->Trace, m->Refs);
#endif
	return rc;
}

/*
** The mutex_leave() routine exits a mutex that was previously entered by the same thread.  The behavior
** is undefined if the mutex is not currently entered or is not currently allocated.  Libcu will never do either.
*/
static void MutexLeave(mutex *m)
{
	assert(MutexHeld(p));
#if MUTEX_NREF
	m->Refs--;
	if (!m->Refs) m->Owner = 0;
#endif
	assert(!m->Refs || m->Id == MUTEX_RECURSIVE);
#ifdef HOMEGROWN_RECURSIVE_MUTEX
	if (!m->Refs)
		pthread_mutex_unlock(&m->Mutex);
#else
	pthread_mutex_unlock(&m->Mutex);
#endif
#ifdef _DEBUG
	if (m->Trace)
		printf("leave mutex %p (%d) with Refs=%d\n", m, m->Trace, m->Refs);
#endif
}

static const mutex_methods DefaultMethods = {
	MutexInit,
	MutexEnd,
	MutexAlloc,
	MutexFree,
	MutexEnter,
	MutexTry,
	MutexLeave,
#ifdef _DEBUG
	MutexHeld,
	MutexNotheld
#else
	nullptr,
	nullptr
#endif
};

__host_device__ mutex_methods const *__mutexsystemDefault() { return &DefaultMethods; }

#endif
#pragma endregion
