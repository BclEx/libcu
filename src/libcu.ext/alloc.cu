#include <ext/alloc.h> //: malloc.c
#include <stringcu.h>
#include <assert.h>

/* Attempt to release up to n bytes of non-essential memory currently held by Libcu. An example of non-essential memory is memory used to
** cache database pages that are not currently in use.
*/
__host_device__ int alloc_releasememory(int n) //: sqlite3_release_memory
{
#ifdef ENABLE_MEMORY_MANAGEMENT
	return pcacheReleaseMemory(n);
#else
	// IMPLEMENTATION-OF: R-34391-24921 The alloc_releasememory() routine is a no-op returning zero if Libcu is not compiled with ENABLE_MEMORY_MANAGEMENT.
	UNUSED_SYMBOL(n);
	return 0;
#endif
}

/* State information local to the memory allocation subsystem. */
static __hostb_device__ _WSD struct Mem0Global {
	mutex *mutex;				// Mutex to serialize access
	int64_t alarmThreshold;		// The soft heap limit
	// True if heap is nearly "full" where "full" is defined by the soft_heap_limit() setting.
	bool nearlyFull;
} _mem0 = { nullptr, 0, false };
#define mem0 _GLOBAL(struct Mem0Global, _mem0)

/* Return the memory allocator mutex. sqlite3_status() needs it. */
__host_device__ mutex *allocMutex() //: sqlite3MallocMutex
{
	return mem0.mutex;
}

/* Set the soft heap-size limit for the library. Passing a zero or negative value indicates no limit. */
__host_device__ int64_t alloc_softheaplimit64(int64_t size) //: sqlite3_soft_heap_limit64
{
#ifndef OMIT_AUTOINIT
	RC rc = runtimeInitialize();
	if (rc) return -1;
#endif
	mutex_enter(mem0.mutex);
	int64_t priorLimit = mem0.alarmThreshold;
	if (size < 0) {
		mutex_leave(mem0.mutex);
		return priorLimit;
	}
	mem0.alarmThreshold = size;
	int64_t used = status_now(STATUS_MEMORY_USED);
	mem0.nearlyFull = size > 0 && size <= used;
	mutex_leave(mem0.mutex);
	int64_t excess = alloc_memoryused() - size;
	if (excess > 0) alloc_releasememory((int)(excess & 0x7fffffff));
	return priorLimit;
}

__host_device__ void alloc_softheaplimit(int size) //: sqlite3_soft_heap_limit
{
	if (size < 0) size = 0;
	alloc_softheaplimit64(size);
}

/* Initialize the memory allocation subsystem. */
__host_device__ RC allocInitialize() //: sqlite3MallocInit
{
	if (!__allocsystem.alloc)
		__allocsystemSetDefault();
	memset(&mem0, 0, sizeof(mem0));
	mem0.mutex = mutex_alloc(MUTEX_STATIC_MEM);
	if (!_runtimeConfig.page || _runtimeConfig.pageSize < 512 || _runtimeConfig.pages <= 0) {
		_runtimeConfig.page = nullptr;
		_runtimeConfig.pageSize = 0;
	}
	RC rc = __allocsystem.initialize(__allocsystem.appData);
	if (rc != RC_OK) memset(&mem0, 0, sizeof(mem0)); //?
	return rc;
}

/* Return true if the heap is currently under memory pressure - in other words if the amount of heap used is close to the limit set by alloc_softheaplimit(). */
__host_device__ bool allocHeapNearlyFull() //: sqlite3HeapNearlyFull
{
	return mem0.nearlyFull;
}

/* Deinitialize the memory allocation subsystem. */
__host_device__ RC allocShutdown() //: sqlite3MallocEnd
{
	RC rc = RC_OK;
	if (__allocsystem.shutdown)
		rc = __allocsystem.shutdown(__allocsystem.appData);
	memset(&mem0, 0, sizeof(mem0));
	return rc;
}

/* Return the amount of memory currently checked out. */
__host_device__ int64_t alloc_memoryused() //: sqlite3_memory_used
{
	int64_t res, max;
	status64(STATUS_MEMORY_USED, &res, &max, false);
	return res;
}

/* Return the maximum amount of memory that has ever been checked out since either the beginning of this process or since the most recent reset. */
__host_device__ int64_t alloc_memoryhighwater(bool resetFlag) //: sqlite3_memory_highwater
{
	int64_t res, max;
	status64(STATUS_MEMORY_USED, &res, &max, resetFlag);
	return max;
}

/* Trigger the alarm */
static __host_device__ void allocAlarm(int bytes)
{
	if (mem0.alarmThreshold <= 0) return;
	mutex_leave(mem0.mutex);
	alloc_releasememory(bytes);
	mutex_enter(mem0.mutex);
}

/* Do a memory allocation with statistics and alarms.  Assume the lock is already held. */
static __host_device__ void allocWithAlarm(int size, void **pp)
{
	assert(mutex_held(mem0.mutex));
	assert(size > 0);

	// In Firefox (circa 2017-02-08), xRoundup() is remapped to an internal implementation of malloc_good_size(), which must be called in debug
	// mode and specifically when the DMD "Dark Matter Detector" is enabled or else a crash results.  Hence, do not attempt to optimize out the
	// following xRoundup() call.
	int fullSize = __allocsystem.roundup(size);
#ifdef MAX_MEMORY
	if (status_now(STATUS_MEMORY_USED) + fullSize > MAX_MEMORY) {
		*pp = nullptr;
		return;
	}
#endif
	status_max(STATUS_MALLOC_SIZE, size);
	if (mem0.alarmThreshold > 0) {
		int64_t used = status_now(STATUS_MEMORY_USED);
		if (used >= mem0.alarmThreshold - fullSize) {
			mem0.nearlyFull = true;
			allocAlarm(fullSize);
		}
		else mem0.nearlyFull = false;
	}
	void *p = __allocsystem.alloc(fullSize);
#ifdef ENABLE_MEMORY_MANAGEMENT
	if (!p && mem0.alarmThreshold > 0) {
		allocAlarm(fullSize);
		p = __allocsystem.alloc(fullSize);
	}
#endif
	if (p) {
		fullSize = allocSize(p);
		status_inc(STATUS_MEMORY_USED, fullSize);
		status_inc(STATUS_MALLOC_COUNT, 1);
	}
	*pp = p;
}

/* Allocate memory.  This routine is like alloc() except that it assumes the memory subsystem has already been initialized. */
__host_device__ void *alloc(uint64_t size) //: sqlite3Malloc
{
	void *p;
	if (!size || size >= 0x7fffff00)
		/* A memory allocation of a number of bytes which is near the maximum signed integer value might cause an integer overflow inside of the
		** _.alloc().  Hence we limit the maximum size to 0x7fffff00, giving 255 bytes of overhead.  Libcu itself will never use anything near
		** this amount.  The only way to reach the limit is with alloc32() */
		p = nullptr;
	else if (_runtimeConfig.memstat) {
		mutex_enter(mem0.mutex);
		allocWithAlarm((int)size, &p);
		mutex_leave(mem0.mutex);
	}
	else p = __allocsystem.alloc((int)size);
	assert(_HASALIGNMENT8(p)); // IMP: R-04675-44850
	return p;
}

/* This version of the memory allocation is for use by the application. First make sure the memory subsystem is initialized, then do the allocation. */
__host_device__ void *alloc32(int n) //: sqlite3_malloc
{
#ifndef OMIT_AUTOINIT
	if (runtimeInitialize()) return nullptr;
#endif
	return n <= 0 ? nullptr : alloc(n);
}
__host_device__ void *alloc64(uint64_t n) //: sqlite3_malloc64
{
#ifndef OMIT_AUTOINIT
	if (runtimeInitialize()) return nullptr;
#endif
	return alloc(n);
}

/* TRUE if p is a lookaside memory allocation from tag */
#ifndef LIBCU_OMITLOOKASIDE
static __host_device__ bool isLookaside(tagbase_t *tag, void *p) { return _WITHIN(p, tag->lookaside.start, tag->lookaside.end); }
#else
#define isLookaside(tag, p) false
#endif

/* Return the size of a memory allocation previously obtained from alloc() or alloc32(). */
__host_device__ int allocSize(void *p) //: sqlite3MallocSize
{
	assert(memdbg_hastype(p, MEMTYPE_HEAP));
	return __allocsystem.size(p);
}

__host_device__ int tagallocSize(tagbase_t *tag, void *p) //: sqlite3DbMallocSize
{
	assert(p);
	if (!tag || !isLookaside(tag, p)) {
#ifdef _DEBUG
		if (!tag) {
			assert(memdbg_nottype(p, (uint8_t)~MEMTYPE_HEAP));
			assert(memdbg_hastype(p, MEMTYPE_HEAP));
		}
		else {
			assert(memdbg_hastype(p, (MEMTYPE_LOOKASIDE|MEMTYPE_HEAP)));
			assert(memdbg_nottype(p, (uint8_t)~(MEMTYPE_LOOKASIDE|MEMTYPE_HEAP)));
		}
#endif
		return __allocsystem.size(p);
	}
	else {
		assert(mutex_held(tag->mutex));
		return tag->lookaside.size;
	}
}

__host_device__ uint64_t alloc_msize(void *p) //: sqlite3_msize
{
	assert(memdbg_nottype(p, (uint8_t)~MEMTYPE_HEAP));
	assert(memdbg_hastype(p, MEMTYPE_HEAP));
	return p ? __allocsystem.size(p) : 0;
}

/* Free memory previously obtained from alloc(). */
__host_device__ void mfree(void *p) //: sqlite3_free
{
	if (!p) return; // IMP: R-49053-54554
	assert(memdbg_hastype(p, MEMTYPE_HEAP));
	assert(memdbg_nottype(p, (uint8_t)~MEMTYPE_HEAP));
	if (_runtimeConfig.memstat) {
		mutex_enter(mem0.mutex);
		status_dec(STATUS_MEMORY_USED, allocSize(p));
		status_dec(STATUS_MALLOC_COUNT, 1);
		__allocsystem.free(p);
		mutex_leave(mem0.mutex);
	}
	else __allocsystem.free(p);
}

/* Add the size of memory allocation "p" to the count in *tag->BytesFreed. */
static __host_device__ void measureAllocationSize(tagbase_t *tag, void *p) { *tag->bytesFreed += tagallocSize(tag, p); }

/* Free memory that might be associated with a particular database connection.  Calling alloc_tagfree(D,X) for X==0 is a harmless no-op.
** The alloc_tagfreeNN(D,X) version requires that X be non-NULL.
*/
__host_device__ void tagfreeNN(tagbase_t *tag, void *p) //: sqlite3DbFreeNN
{
	assert(!tag || mutex_held(tag->mutex));
	assert(p);
	if (tag) {
		if (tag->bytesFreed) {
			measureAllocationSize(tag, p);
			return;
		}
		if (isLookaside(tag, p)) {
			LookasideSlot *b = (LookasideSlot *)p;
#if _DEBUG
			// Trash all content in the buffer being freed
			memset(p, 0xaa, tag->lookaside.size);
#endif
			b->next = tag->lookaside.free_;
			tag->lookaside.free_ = b;
			return;
		}
	}
	assert(memdbg_hastype(p, (MEMTYPE_LOOKASIDE|MEMTYPE_HEAP)));
	assert(memdbg_nottype(p, (uint8_t)~(MEMTYPE_LOOKASIDE|MEMTYPE_HEAP)));
	assert(tag || memdbg_nottype(p, MEMTYPE_LOOKASIDE));
	memdbg_settype(p, MEMTYPE_HEAP);
	mfree(p);
}

__host_device__ void tagfree(tagbase_t *tag, void *p) //: sqlite3DbFree
{ 
	assert(!tag || mutex_held(tag->mutex));
	if (p) tagfreeNN(tag, p);
}

/* Change the size of an existing memory allocation */
__host_device__ void *allocRealloc(void *prior, uint64_t newSize) //: sqlite3Realloc
{
	assert(memdbg_hastype(prior, MEMTYPE_HEAP));
	assert(memdbg_nottype(prior, (uint8_t)~MEMTYPE_HEAP));
	if (!prior) return alloc(newSize); // IMP: R-04300-56712
	if (!newSize) { mfree(prior); return nullptr; } // IMP: R-26507-47431
	if (newSize >= 0x7fffff00) return nullptr; // The 0x7ffff00 limit term is explained in comments on alloc()
	int oldSize = allocSize(prior);
	// IMPLEMENTATION-OF: R-46199-30249 Libcu guarantees that the second argument to _.xRealloc is always a value returned by a prior call to _.Roundup.
	void *p;
	int newSize2 = __allocsystem.roundup((int)newSize);
	if (oldSize == newSize2)
		p = prior;
	else if (_runtimeConfig.memstat) {
		mutex_enter(mem0.mutex);
		status_max(STATUS_MALLOC_SIZE, (int)newSize);
		int sizeDiff = newSize2 - oldSize;
		if (sizeDiff > 0 && status_now(STATUS_MEMORY_USED) >= mem0.alarmThreshold - sizeDiff)
			allocAlarm(sizeDiff);
		p = __allocsystem.realloc(prior, newSize2);
		if (!p && mem0.alarmThreshold > 0) {
			allocAlarm((int)newSize);
			p = __allocsystem.realloc(prior, newSize2);
		}
		if (p) {
			newSize2 = allocSize(p);
			status_inc(STATUS_MEMORY_USED, newSize2 - oldSize);
		}
		mutex_leave(mem0.mutex);
	}
	else p = __allocsystem.realloc(prior, newSize2);
	assert(_HASALIGNMENT8(p)); // IMP: R-11148-40995
	return p;
}

/* The public interface to alloc_realloc_.  Make sure that the memory subsystem is initialized prior to invoking alloc_realloc_. */
__host_device__ void *alloc_realloc32(void *prior, int newSize) //: sqlite3_realloc
{
#ifndef OMIT_AUTOINIT
	if (runtimeInitialize()) return nullptr;
#endif
	if (newSize < 0) newSize = 0;  // IMP: R-26507-47431
	return allocRealloc(prior, newSize);
}

__host_device__ void *alloc_realloc64(void *prior, uint64_t newSize) //: sqlite3_realloc64
{
#ifndef OMIT_AUTOINIT
	if (runtimeInitialize()) return nullptr;
#endif
	return allocRealloc(prior, newSize);
}

/* Allocate and zero memory. */
__host_device__ void *allocZero(uint64_t size) //: sqlite3MallocZero
{
	void *p = alloc(size);
	if (p) memset(p, 0, (size_t)size);
	return p;
}

/* Allocate and zero memory.  If the allocation fails, make the mallocFailed flag in the connection pointer. */
__host_device__ void *tagallocZero(tagbase_t *tag, uint64_t size) //: sqlite3DbMallocZero
{
	ASSERTCOVERAGE(tag);
	void *p = tagallocRaw(tag, size);
	if (p) memset(p, 0, (size_t)size);
	return p;
}

/* Finish the work of tagallocrawNN for the unusual and slower case when the allocation cannot be fulfilled using lookaside. */
static __host_device__ void *tagallocRawFinish(tagbase_t *tag, uint64_t size)
{
	assert(tag);
	void *p = alloc(size);
	if (!p) tagOomFault(tag);
	memdbg_settype(p, (!tag->lookaside.disable ? MEMTYPE_LOOKASIDE : MEMTYPE_HEAP));
	return p;
}

/* Allocate memory, either lookaside (if possible) or heap.   If the allocation fails, set the MallocFailed flag in
** the tag object.
**
** If tag!=0 and tag->MallocFailed is true (indicating a prior malloc failure on the same tag object) then always return nullptr.
** Hence for a particular tag, once malloc starts failing, it fails consistently until MallocFailed is reset.
** This is an important assumption.  There are many places in the code that do things like this:
**
**         int *a = (int*)sqlite3DbMallocRaw(db, 100);
**         int *b = (int*)sqlite3DbMallocRaw(db, 200);
**         if( b ) a[10] = 9;
**
** In other words, if a subsequent malloc (ex: "b") worked, it is assumed that all prior mallocs (ex: "a") worked too.
**
** The tagallocrawNN() variant guarantees that the "tag" parameter is not a NULL pointer.
*/
__host_device__ void *tagallocRaw(tagbase_t *tag, uint64_t size) //: sqlite3DbMallocRaw
{
	if (tag) return tagallocRawNN(tag, size);
	void *p = alloc(size);
	memdbg_settype(p, MEMTYPE_HEAP);
	return p;
}

__host_device__ void *tagallocRawNN(tagbase_t *tag, uint64_t size) //: sqlite3DbMallocRawNN
{
#ifndef LIBCU_OMITLOOKASIDE
	assert(tag);
	assert(mutex_held(tag->mutex));
	assert(!tag->bytesFreed);
	if (!tag->lookaside.disable) {
		LookasideSlot *b;
		assert(!tag->mallocFailed);
		if (size > tag->lookaside.size)
			tag->lookaside.stats[1]++;
		else if (!(b = tag->lookaside.free_)) {
			tag->lookaside.free_ = b->next;
			tag->lookaside.stats[0]++;
			return (void *)b;
		}
		else if ((b = tag->lookaside.init)) {
			tag->lookaside.init = b->next;
			tag->lookaside.stats[0]++;
			return (void *)b;
		}
		else tag->lookaside.stats[2]++;
	}
	else if (tag->mallocFailed)
		return nullptr;
#else
	assert(tag);
	assert(mutex_held(tag->Mutex));
	assert(!tag->BytesFreed);
	if (tag->MallocFailed)
		return nullptr;
#endif
	return tagallocRawFinish(tag, size);
}

/* Forward declaration */
static __host_device__ void *tagreallocFinish(tagbase_t *tag, void *prior, uint64_t size);

/* Resize the block of memory pointed to by p to n bytes. If the resize fails, set the MallocFailed flag in the tag object. */
__host_device__ void *tagrealloc(tagbase_t *tag, void *prior, uint64_t size) //: sqlite3DbRealloc
{
	assert(tag);
	if (!prior) return tagallocRawNN(tag, size);
	assert(mutex_held(tag->mutex));
	if (isLookaside(tag, prior) && size <= tag->lookaside.size) return prior;
	return tagreallocFinish(tag, prior, size);
}

static __host_device__ void *tagreallocFinish(tagbase_t *tag, void *prior, uint64_t size)
{
	assert(tag);
	assert(prior);
	void *p = nullptr;
	if (!tag->mallocFailed) {
		if (isLookaside(tag, prior)) {
			p = tagallocRawNN(tag, size);
			if (p) {
				memcpy(p, prior, tag->lookaside.size);
				tagfree(tag, prior);
			}
		}
		else {
			assert(memdbg_hastype(prior, (MEMTYPE_LOOKASIDE|MEMTYPE_HEAP)) );
			assert(memdbg_nottype(prior, (uint8_t)~(MEMTYPE_LOOKASIDE|MEMTYPE_HEAP)) );
			memdbg_settype(prior, MEMTYPE_HEAP);
			p = alloc_realloc64(prior, size);
			if (!p)
				tagOomFault(tag);
			memdbg_settype(p, (!tag->lookaside.disable ? MEMTYPE_LOOKASIDE : MEMTYPE_HEAP));
		}
	}
	return p;
}

/* Attempt to reallocate p.  If the reallocation fails, then free p and set the MallocFailed flag in the tag object. */
__host_device__ void *tagreallocOrFree(tagbase_t *tag, void *prior, uint64_t newSize) //: sqlite3DbReallocOrFree
{
	void *p = tagrealloc(tag, prior, newSize);
	if (!p) tagfree(tag, prior);
	return p;
}

/* Make a copy of a string in memory obtained from sqliteMalloc(). These functions call sqlite3MallocRaw() directly instead of sqliteMalloc(). This
** is because when memory debugging is turned on, these two functions are called via macros that record the current file and line number in the ThreadData structure.
*/
__host_device__ char *tagstrdup(tagbase_t *tag, const char *z) //: sqlite3DbStrDup
{
	if (!z) return nullptr;
	size_t size = strlen(z) + 1;
	assert((size & 0x7fffffff) == size);
	char *newZ = (char *)tagallocRaw(tag, size);
	if (newZ) memcpy(newZ, z, size);
	return newZ;
}

__host_device__ char *tagstrndup(tagbase_t *tag, const char *z, uint64_t size) //: sqlite3DbStrNDup
{
	assert(tag);
	if (!z) return nullptr;
	assert((size & 0x7fffffff) == size);
	char *newZ = (char *)tagallocRawNN(tag, size + 1);
	if (newZ) { memcpy(newZ, z, (size_t)size); newZ[size] = 0; }
	return newZ;
}

/* Free any prior content in *pz and replace it with a copy of zNew. */
__host_device__ void tagstrset(char **z, tagbase_t *tag, const char *newZ) //: sqlite3SetString
{
	tagfree(tag, *z);
	*z = tagstrdup(tag, newZ);
}

/* Call this routine to record the fact that an OOM (out-of-memory) error has happened.  This routine will set tag->MallocFailed, and also
** temporarily disable the lookaside memory allocator and interrupt any running VDBEs.
*/
__host_device__ void tagOomFault(tagbase_t *tag)
{
	if (!tag->mallocFailed && !tag->benignMalloc) {
		tag->mallocFailed = true;
		if (tag->vdbeExecs > 0)
			tag->u1.isInterrupted = true;
		tag->lookaside.disable++;
	}
}

/* This routine reactivates the memory allocator and clears the tag->MallocFailed flag as necessary.
**
** The memory allocator is not restarted if there are running VDBEs.
*/
__host_device__ void tagOomClear(tagbase_t *tag) //: sqlite3OomClear
{
	if (tag->mallocFailed && !tag->vdbeExecs) {
		tag->mallocFailed = false;
		tag->u1.isInterrupted = false;
		assert(tag->lookaside.disable > 0);
		tag->lookaside.disable--;
	}
}

/* Take actions at the end of an API call to indicate an OOM error */
static __host_device__ RC tagOomError(tagbase_t *tag)
{
	tagOomClear(tag);
	tagError(tag, RC_NOMEM);
	return RC_NOMEM_BKPT;
}

/* This function must be called before exiting any API function (i.e. returning control to the user) that has called sqlite3_malloc or sqlite3_realloc.
**
** The returned value is normally a copy of the second argument to this function. However, if a malloc() failure has occurred since the previous
** invocation SQLITE_NOMEM is returned instead. 
**
** If an OOM as occurred, then the connection error-code (the value returned by sqlite3_errcode()) is set to SQLITE_NOMEM.
*/
__host_device__ RC tagApiExit(tagbase_t *tag, RC rc) //: sqlite3ApiExit
{
	// If the tag handle must hold the connection handle mutex here. Otherwise the read (and possible write) of tag->mallocFailed  is unsafe, as is the call to sqlite3Error().
	assert(tag);
	assert(mutex_held(tag->mutex));
	if (tag->mallocFailed || rc == RC_IOERR_NOMEM)
		return tagOomError(tag);
	return rc & tag->errMask;
}

#pragma region from: fault.c

#ifndef LIBCU_UNTESTABLE

typedef struct BenignMallocHooks BenignMallocHooks;
static __hostb_device__ _WSD struct BenignMallocHooks {
	void (*benignBegin)();
	void (*benignEnd)();
} _benignMallocHooks = { nullptr, nullptr };
#ifdef OMIT_WSD
#define _benignMallocHooksInit BenignMallocHooks *x = &GLOBAL(BenignMallocHooks, _benignMallocHooks)
#define _benignMallocHooks x[0]
#else
#define _benignMallocHooksInit
#define _benignMallocHooks _benignMallocHooks
#endif

/* Register hooks to call when allocBenignBbegin() and allocBenignEnd() are called, respectively */
__host_device__ void allocBenignHook(void (*benignBegin)(), void (*benignEnd)()) //: sqlite3BenignMallocHooks
{
	_benignMallocHooksInit;
	_benignMallocHooks.benignBegin = benignBegin;
	_benignMallocHooks.benignEnd = benignEnd;
}

/* This (allocBenignEnd()) is called by Libcu code to indicate that subsequent malloc failures are benign. A call to allocBenignBegin()
** indicates that subsequent malloc failures are non-benign.
*/
__host_device__ void allocBenignBegin() //: sqlite3BeginBenignMalloc
{
	_benignMallocHooksInit;
	if (_benignMallocHooks.benignBegin)
		_benignMallocHooks.benignBegin();
}

__host_device__ void allocBenignEnd() //: sqlite3EndBenignMalloc
{
	_benignMallocHooksInit;
	if (_benignMallocHooks.benignEnd)
		_benignMallocHooks.benignEnd();
}

#endif

#pragma endregion