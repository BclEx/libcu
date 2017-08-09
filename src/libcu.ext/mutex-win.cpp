#include <ext/global.h>
#include <assert.h>

#if LIBCU_OS_WIN
/*
** Include code that is common to all os_*.c files
*/
#include "os_common.h"

/*
** Include the header file for the Windows VFS.
*/
#include "os_win.h"
#endif

//////////////////////
// MUTEX W32
#pragma region MUTEX W32
#ifdef LIBCU_MUTEX_W32
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>

struct mutex {
	CRITICAL_SECTION Mutex;		// Mutex controlling the lock
	MUTEX id;					// Mutex type
#ifdef _DEBUG
	volatile int refs;			// Number of enterances
	volatile DWORD owner;		// Thread holding this mutex
	bool trace;					// True to trace changes
#endif
};

/* These are the initializer values used when declaring a "static" mutex on Win32.  It should be noted that all mutexes require initialization on the Win32 platform. */
#define W32_MUTEX_INITIALIZER { 0 }
#ifdef _DEBUG
#define MUTEX_INITIALIZER { W32_MUTEX_INITIALIZER, 0, 0L, (DWORD)0, 0 }
#else
#define MUTEX_INITIALIZER { W32_MUTEX_INITIALIZER, 0 }
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

#ifdef _DEBUG
/* The mutex_held() and mutex_notheld() routine are intended for use only inside assert() statements. */
static bool mutexHeld(mutex *m) { return m->refs && m->owner == GetCurrentThreadId(); }
static bool mutexNotheld2(mutex *m, DWORD tid) { return m->refs || m->owner != tid; }
static bool mutexNotheld(mutex *m) { DWORD tid = GetCurrentThreadId(); return mutexNotheld2(m, tid); }
#endif

/* Try to provide a memory barrier operation, needed for initialization and also for the xShmBarrier method of the VFS in cases when Libcu is compiled without mutexes (THREADSAFE=0). */
void mutexMemoryBarrier()
{
#if defined(MEMORY_BARRIER)
	MEMORY_BARRIER;
#elif defined(__GNUC__)
	__sync_synchronize();
#elif MSVC_VERSION>=1300
	_ReadWriteBarrier();
#elif defined(MemoryBarrier)
	MemoryBarrier();
#endif
}

/* Initialize and deinitialize the mutex subsystem. */
static bool _mutexIsInit = false;
static int _mutexIsNt = -1; // <0 means "need to query"

/* As the winMutexInit() and winMutexEnd() functions are called as part of the sqlite3_initialize() and sqlite3_shutdown() processing, the "interlocked" magic used here is probably not strictly necessary. */
static LONG WIN32_VOLATILE _mutexLock = 0;

//int sqlite3_win32_is_nt(void); /* os_win.c */
//void sqlite3_win32_sleep(DWORD milliseconds); /* os_win.c */

RC mutexInitialize()
{ 
	// The first to increment to 1 does actual initialization
	if (!InterlockedCompareExchange(&_mutexLock, 1, 0)) {
		for (int i = 0; i < _LENGTHOF(MutexStatics); i++) {
#if OS_WINRT
			InitializeCriticalSectionEx(&MutexStatics[i].Mutex, 0, 0);
#else
			InitializeCriticalSection(&MutexStatics[i].Mutex);
#endif
		}
		_mutexIsInit = true;
	}
	else
		// Another thread is (in the process of) initializing the static mutexes
		while (!_mutexIsInit)
			Sleep(1);
	return RC_OK; 
}

RC MutexShutdown()
{
	// The first to decrement to 0 does actual shutdown (which should be the last to shutdown.)
	if (InterlockedCompareExchange(&_mutexLock, 0, 1) == 1) {
		if (_mutexIsInit) {
			for (int i =0 ; i < _LENGTHOF(MutexStatics); i++)
				DeleteCriticalSection(&MutexStatics[i].Mutex);
			_mutexIsInit = false;
		}
	}
	return RC_OK;
}

/*
** The mutexAlloc() routine allocates a new mutex and returns a pointer to it.  If it returns NULL
** that means that a mutex could not be allocated.  Libcu will unwind its stack and return an error.  The argument
** to mutexAlloc() is one of these integer constants:
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
** The first two constants cause mutexAlloc() to create a new mutex.  The new mutex is recursive when MUTEX_RECURSIVE
** is used but not necessarily so when MUTEX_FAST is used. The mutex implementation does not need to make a distinction
** between MUTEX_RECURSIVE and MUTEX_FAST if it does not want to.  But Libcu will only request a recursive mutex in
** cases where it really needs one.  If a faster non-recursive mutex implementation is available on the host platform, the mutex subsystem
** might return such a mutex in response to MUTEX_FAST.
**
** The other allowed parameters to mutexAlloc() each return a pointer to a static preexisting mutex.  Six static mutexes are
** used by the current version of Libcu.  Future versions of Libcu may add additional static mutexes.  Static mutexes are for internal
** use by Libcu only.  Applications that use Libcu mutexes should use only the dynamic mutexes returned by MUTEX_FAST or MUTEX_RECURSIVE.
**
** Note that if one of the dynamic mutex parameters (MUTEX_FAST or MUTEX_RECURSIVE) is used then mutexAalloc() returns a different mutex on every call.  But for the static
** mutex types, the same mutex is returned on every call that has the same type number.
*/
mutex *mutexAlloc(MUTEX id)
{
	mutex *m;
	switch (id) {
	case MUTEX_FAST:
	case MUTEX_RECURSIVE: {
		m = (mutex *)allocZero(sizeof(*m));
		if (m) {  
			m->id = id;
#ifdef _DEBUG
#ifdef WIN32_MUTEX_TRACE_DYNAMIC
			p->trace = true;
#endif
#endif
#if OS_WINRT
			InitializeCriticalSectionEx(&m->Mutex, 0, 0);
#else
			InitializeCriticalSection(&m->Mutex);
#endif
		}
		break; }
	default: {
#ifdef ENABLE_API_ARMOR
		if (id-2 < 0 || id-2 >= _LENGTHOF(MutexStatics)) {
			(void)RC_MISUSE_BKPT;
			return 0;
		}
#endif
		m = &MutexStatics[id-2];
		m->id = id;
#ifdef _DEBUG
#ifdef WIN32_MUTEX_TRACE_STATIC
		m->trace = true;
#endif
#endif
		break; }
	}
	return m;
}

/* This routine deallocates a previously allocated mutex.  SQLite is careful to deallocate every mutex that it allocates. */
void mutexFree(mutex *m)
{
	assert(m);
	assert(!m->refs && !m->owner);
	if (m->id == MUTEX_FAST || m->id == MUTEX_RECURSIVE) {
		DeleteCriticalSection(&m->Mutex);
		mfree(m);
	}
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
void mutexEnter(mutex *m)
{
#if defined(_DEBUG) || defined(_TEST)
	DWORD tid = GetCurrentThreadId();
#endif
#ifdef _DEBUG
	assert(m);
	assert(m->id == MUTEX_RECURSIVE || mutexNotheld2(m, tid));
#else
	assert(m);
#endif
	assert(_mutexIsInit);
	EnterCriticalSection(&m->Mutex);
#ifdef _DEBUG
	assert(m->refs > 0 || !m->owner);
	m->owner = tid;
	m->refs++;
	if (m->trace)
		printf("ENTER-MUTEX tid=%lu, mutex(%d)=%p (%d), Refs=%d\n", tid, m->id, m, m->trace, m->refs);
#endif
}

bool mutexTryEnter(mutex *m)
{
#if defined(_DEBUG) || defined(_TEST)
	DWORD tid = GetCurrentThreadId();
#endif
	bool rc = false;
	assert(m);
	assert(m->id == MUTEX_RECURSIVE || mutexNotheld2(m, tid));
	// The mutex_try() routine is very rarely used, and when it is used it is merely an optimization.  So it is OK for it to always fail.
	//
	// The TryEnterCriticalSection() interface is only available on WinNT. And some windows compilers complain if you try to use it without
	// first doing some #defines that prevent SQLite from building on Win98. For that reason, we will omit this optimization for now.  See ticket #2685.
#if defined(_WIN32_WINNT) && _WIN32_WINNT >= 0x0400
	assert(_mutexIsInit);
	assert(_mutexIsNt >= -1 && _mutexIsNt <= 1);
	if (_mutexIsNt < 0)
		_mutexIsNt = sqlite3_win32_is_nt();
	assert(_mutexIsNt == 0 || _mutexIsNt == 1);
	if (_mutexIsNt && TryEnterCriticalSection(&m->Mutex)) {
#ifdef _DEBUG
		m->owner = tid;
		m->refs++;
#endif
		rc = true;
	}
#else
	UNUSED_SYMBOL(m);
#endif
#ifdef _DEBUG
	if (m->trace)
		printf("TRY-MUTEX tid=%lu, mutex(%d)=%p (%d), owner=%lu, Refs=%d, rc=%s\n", tid, m->id, m, m->trace, m->owner, m->refs, rc?"OK":"BUSY");
#endif
	return rc;
}

/*
** The mutex_leave() routine exits a mutex that was previously entered by the same thread.  The behavior
** is undefined if the mutex is not currently entered or is not currently allocated.  Libcu will never do either.
*/
void mutexLeave(mutex *m)
{
#if defined(_DEBUG) || defined(_TEST)
	DWORD tid = GetCurrentThreadId();
#endif
	assert(m);
#ifdef _DEBUG
	assert(m->refs > 0);
	assert(m->owner == tid);
	m->refs--;
	if (!m->refs) m->owner = 0;
	assert(!m->refs || m->id == MUTEX_RECURSIVE);
#endif
	assert(_mutexIsInit);
	LeaveCriticalSection(&m->Mutex);
#ifdef _DEBUG
	if (m->trace)
		printf("LEAVE-MUTEX tid=%lu, mutex(%d)=%p (%d), Refs=%d\n", tid, m->id, m, m->trace, m->refs);
#endif
}

static const mutex_methods _defaultMethods = {
	mutexInitialize,
	mutexShutdown,
	mutexAlloc,
	mutexFree,
	mutexEnter,
	mutexTryEnter,
	mutexLeave,
#ifdef _DEBUG
	mutexHeld,
	mutexNotheld
#else
	nullptr,
	nullptr
#endif
};

__host_device__ mutex_methods const *__mutexsystemDefault() { return &_defaultMethods; }

#endif
#pragma endregion