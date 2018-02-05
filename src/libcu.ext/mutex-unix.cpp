#include <ext/global.h> //: mutex_unix.c

//////////////////////
// MUTEX PTHREADS
#pragma region MUTEX PTHREADS
#ifdef LIBCU_MUTEX_PTHREADS
#include <pthread.h>

/* The mutex.Id, mutex.Refs, and mutex.Owner fields are necessary under two condidtions:  (1) Debug builds and (2) using
** home-grown mutexes.  Encapsulate these conditions into a single #define.
*/
#if defined(_DEBUG) || defined(HOMEGROWN_RECURSIVE_MUTEX)
#define MUTEX_NREF 1
#else
#define MUTEX_NREF 0
#endif

/* Each recursive mutex is an instance of the following structure. */
struct mutex {
	pthread_mutex_t mutex;		// Mutex controlling the lock
#if MUTEX_NREF || defined(ENABLE_API_ARMOR)
	MUTEX id;					// Mutex type
#endif
#if MUTEX_NREF
	volatile int refs;			// Number of entrances
	volatile pthread_t owner;	// Thread that is within this mutex
	bool trace;					// True to trace changes
#endif
};

#if MUTEX_NREF
#define MUTEX_INITIALIZER { PTHREAD_MUTEX_INITIALIZER, 0, 0, (pthread_t)nullptr, 0 }
#elif defined(ENABLE_API_ARMOR)
#define MUTEX_INITIALIZER { PTHREAD_MUTEX_INITIALIZER, 0 }
#else
#define MUTEX_INITIALIZER { PTHREAD_MUTEX_INITIALIZER }
#endif
static mutex unixMutexStatics[] = {
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

/* The mutex_held() and mutex_notheld() routine are intended for use only inside assert() statements.  On some platforms,
** there might be race conditions that can cause these routines to deliver incorrect results.  In particular, if pthread_equal() is
** not an atomic operation, then these routines might delivery incorrect results.  On most platforms, pthread_equal() is a 
** comparison of two integers and is therefore atomic.  But we are told that HPUX is not such a platform.  If so, then these routines
** will not always work correctly on HPUX.
**
** On those platforms where pthread_equal() is not atomic, Libcu should be compiled without -DDEBUG and with -DNDEBUG to
** make sure no assert() statements are evaluated and hence these routines are never called.
*/
#if !defined(NDEBUG) || defined(_DEBUG)
static bool unixMutexHeld(mutex *m) { return m->refs && pthread_equal(m->owner, pthread_self()); }
static bool unixMutexNotHeld(mutex *m) { return !m->refs || !pthread_equal(m->owner, pthread_self()); }
#endif

/* Try to provide a memory barrier operation, needed for initialization and also for the implementation of xShmBarrier in the VFS in cases
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

/* Initialize and deinitialize the mutex subsystem. */
static RC unixMutexInitialize() { return RC_OK; }
static RC unixMutexShutdown() { return RC_OK; }

/* The mutex_alloc() routine allocates a new mutex and returns a pointer to it.  If it returns NULL
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
static mutex *unixMutexAlloc(MUTEX id)
{
	mutex *m;
	switch (id) {
	case MUTEX_RECURSIVE: {
		m = (mutex *)allocZero(sizeof(*m));
		if (m) {
#ifdef HOMEGROWN_RECURSIVE_MUTEX
			// If recursive mutexes are not available, we will have to build our own.  See below.
			pthread_mutex_init(&m->mutex, 0);
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
			pthread_mutex_init(&m->mutex, 0);
		break; }
	default: {
#ifdef ENABLE_API_ARMOR
		if (id-2 < 0 || id-2 >= _ARRAYSIZE(unixMutexStatics)) {
			(void)RC_MISUSE_BKPT;
			return 0;
		}
#endif
		m = &unixMutexStatics[id-2];
		break; }
	}
#if MUTEX_NREF || defined(ENABLE_API_ARMOR)
	if (m) m->id = id;
#endif
	return m;
}

/* This routine deallocates a previously allocated mutex.  Libcu is careful to deallocate every mutex that it allocates. */
static void unixMutexFree(mutex *m)
{
	assert(!m->refs);
#if ENABLE_API_ARMOR
	if (m->id == MUTEX_FAST || m->id == MUTEX_RECURSIVE)
#endif
	{
		pthread_mutex_destroy(&m->mutex);
		mfree(p);
	}
#ifdef ENABLE_API_ARMOR
	else (void)RC_MISUSE_BKPT;
#endif
}

/* The mutex_enter() and mutex_tryenter() routines attempt to enter a mutex.  If another thread is already within the mutex,
** mutex_enter() will block and mutex_tryenter() will return true.  The mutex_tryenter() interface returns false
** upon successful entry.  Mutexes created using MUTEX_RECURSIVE can be entered multiple times by the same thread.  In such cases the,
** mutex must be exited an equal number of times before another thread can enter.  If the same thread tries to enter any other kind of mutex
** more than once, the behavior is undefined.
*/
static void unixMutexEnter(mutex *m)
{
	assert(m->id == MUTEX_RECURSIVE || unixMutexNotheld(m));
#ifdef HOMEGROWN_RECURSIVE_MUTEX
	// If recursive mutexes are not available, then we have to grow our own.  This implementation assumes that pthread_equal()
	// is atomic - that it cannot be deceived into thinking self and m->Owner are equal if m->Owner changes between two values
	// that are not equal to self while the comparison is taking place. This implementation also assumes a coherent cache - that 
	// separate processes cannot read different values from the same address at the same time.  If either of these two conditions
	// are not met, then the mutexes will fail and problems will result.
	{
		pthread_t self = pthread_self();
		if (p->Refs && pthread_equal(p->owner, self))
			p->refs++;
		else {
			pthread_mutex_lock(&m->mutex);
			assert(!m->refs);
			m->owner = self;
			m->refs = 1;
		}
	}
#else
	// Use the built-in recursive mutexes if they are available.
	pthread_mutex_lock(&m->mutex);
#if MUTEX_NREF
	assert(m->refs || !m->owner);
	m->owner = pthread_self();
	m->refs++;
#endif
#endif
#ifdef _DEBUG
	if (m->trace)
		printf("enter mutex %p (%d) with refs=%d\n", m, m->trace, m->refs);
#endif
}

static bool unixMutexTryEnter(mutex *m)
{
	assert(m->id == MUTEX_RECURSIVE || unixMutexNotheld(m));
	bool rc;
#ifdef HOMEGROWN_RECURSIVE_MUTEX
	// If recursive mutexes are not available, then we have to grow our own.  This implementation assumes that pthread_equal()
	// is atomic - that it cannot be deceived into thinking self and m->Owner are equal if m->Owner changes between two values
	// that are not equal to self while the comparison is taking place. This implementation also assumes a coherent cache - that 
	// separate processes cannot read different values from the same address at the same time.  If either of these two conditions
	// are not met, then the mutexes will fail and problems will result.
	{
		pthread_t self = pthread_self();
		if (m->refs && pthread_equal(m->owner, self)) {
			m->refs++;
			rc = false;
		}
		else if (!pthread_mutex_trylock(&m->mutex)) {
			assert(!m->refs);
			m->owner = self;
			m->refs = 1;
			rc = false;
		}
		else rc = true;
	}
#else
	// Use the built-in recursive mutexes if they are available.
	if (!pthread_mutex_trylock(&m->mutex)) {
#if MUTEX_NREF
		m->owner = pthread_self();
		m->refs++;
#endif
		rc = false;
	}
	else rc = true;
#endif

#ifdef _DEBUG
	if (!rc && m->trace)
		printf("enter mutex %p (%d) with refs=%d\n", m, m->trace, m->refs);
#endif
	return rc;
}

/* The mutex_leave() routine exits a mutex that was previously entered by the same thread.  The behavior
** is undefined if the mutex is not currently entered or is not currently allocated.  Libcu will never do either.
*/
static void unixMutexLeave(mutex *m)
{
	assert(unixMutexHeld(p));
#if MUTEX_NREF
	m->refs--;
	if (!m->refs) m->owner = 0;
#endif
	assert(!m->refs || m->id == MUTEX_RECURSIVE);
#ifdef HOMEGROWN_RECURSIVE_MUTEX
	if (!m->Refs)
		pthread_mutex_unlock(&m->Mutex);
#else
	pthread_mutex_unlock(&m->mutex);
#endif
#ifdef _DEBUG
	if (m->trace)
		printf("leave mutex %p (%d) with refs=%d\n", m, m->trace, m->refs);
#endif
}

static const mutex_methods _unixDefaultMethods = {
	unixMutexInit,
	unixMutexEnd,
	unixMutexAlloc,
	unixMutexFree,
	unixMutexEnter,
	unixMutexTry,
	unixMutexLeave,
#ifdef _DEBUG
	unixMutexHeld,
	unixMutexNotheld
#else
	nullptr,
	nullptr
#endif
};

__host_device__ mutex_methods const *__mutexsystemDefault() { return &_unixDefaultMethods; }

#endif
#pragma endregion
