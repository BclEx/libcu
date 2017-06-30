#include <stdlibcu.h>
#include <ext/mutex.h>
#include <ext/alloc.h>
#include <stdint.h>
#include <assert.h>

__host__ __device__ int systemInitialize();

/*
** Attempt to release up to n bytes of non-essential memory currently held by Libcu. An example of non-essential memory is memory used to
** cache database pages that are not currently in use.
*/
static __host__ __device__ int alloc_releasememory(int n)
{
#ifdef ENABLE_MEMORY_MANAGEMENT
	return sqlite3PcacheReleaseMemory(n);
#else
	// IMPLEMENTATION-OF: R-34391-24921 The alloc_releasememory() routine is a no-op returning zero if Libcu is not compiled with ENABLE_MEMORY_MANAGEMENT.
	UNUSED_SYMBOL(n);
	return 0;
#endif
}

/* An instance of the following object records the location of each unused scratch buffer. */
typedef struct ScratchFreeslot {
	struct ScratchFreeslot *Next; // Next unused scratch buffer
} ScratchFreeslot;

/* State information local to the memory allocation subsystem. */
static __host__ __device__ _WSD struct Mem0Global {
	mutex *Mutex;				// Mutex to serialize access
	int64_t AlarmThreshold;		// The soft heap limit

	// Pointers to the end of __allocsystem.Scratch memory (so that a range test can be used to determine if an allocation
	// being freed came from Scratch) and a pointer to the list of unused scratch allocations.
	void *ScratchEnd;
	ScratchFreeslot *ScratchFree;
	uint32_t ScratchFreeLength;

	// True if heap is nearly "full" where "full" is defined by the soft_heap_limit() setting.
	bool NearlyFull;
} g_mem0 = { nullptr, 0, nullptr, nullptr, 0, false };
#define mem0 _GLOBAL(struct Mem0Global, g_mem0)

/* Return the memory allocator mutex. sqlite3_status() needs it. */
mutex *allocMallocMutex() //: sqlite3MallocMutex
{
	return mem0.Mutex;
}

/* Set the soft heap-size limit for the library. Passing a zero or negative value indicates no limit. */
__host__ __device__ int64_t alloc_softheaplimit64(int64_t size) //: sqlite3_soft_heap_limit64
{
#ifndef OMIT_AUTOINIT
	int rc = systemInitialize();
	if (rc) return -1;
#endif
	mutex_enter(mem0.Mutex);
	int64_t priorLimit = mem0.AlarmThreshold;
	if (size < 0) {
		mutex_leave(mem0.Mutex);
		return priorLimit;
	}
	mem0.AlarmThreshold = size;
	int64_t used = status_value(STATUS_MEMORY_USED);
	mem0.NearlyFull = (size > 0 && size <= used);
	mutex_leave(mem0.Mutex);
	int64_t excess = alloc_memoryused() - size;
	if (excess > 0) alloc_releasememory((int)(excess & 0x7fffffff));
	return priorLimit;
}

__host__ __device__ void alloc_softheaplimit(int size) //: sqlite3_soft_heap_limit
{
	if (size < 0) size = 0;
	alloc_softheaplimit64(size);
}

/* Initialize the memory allocation subsystem. */
__host__ __device__ int allocInitialize() //: sqlite3MallocInit
{
	if (!__allocsystem.Alloc)
		__allocsystemSetDefault();
	memset(&mem0, 0, sizeof(mem0));
	mem0.Mutex = mutex_alloc(MUTEX_STATIC_MEM);
	if (g_runtimeStatics.Scratch && g_runtimeStatics.ScratchSize >= 100 && g_runtimeStatics.Scratchs > 0) {
		int size = _ROUNDDOWN8(g_runtimeStatics.ScratchSize);
		g_runtimeStatics.ScratchSize = size;
		ScratchFreeslot *slot = (ScratchFreeslot *)g_runtimeStatics.Scratch;
		int n = g_runtimeStatics.Scratchs;
		mem0.ScratchFree = slot;
		mem0.ScratchFreeLength = n;
		for (int i = 0; i < n-1; i++) {
			slot->Next = (ScratchFreeslot *)(size + (char *)slot);
			slot = slot->Next;
		}
		slot->Next = nullptr;
		mem0.ScratchEnd = (void *)&slot[1];
	}
	else {
		mem0.ScratchEnd = nullptr;
		g_runtimeStatics.Scratch = nullptr;
		g_runtimeStatics.ScratchSize = 0;
		g_runtimeStatics.Scratchs = 0;
	}
	//if (!g_runtimeStatics.Page || g_runtimeStatics.PageSize < 512 || g_runtimeStatics.Pages <= 0) {
	//	g_runtimeStatics.Page = nullptr;
	//	g_runtimeStatics.PageSize = 0;
	//}
	int rc = __allocsystem.Initialize(__allocsystem.AppData);
	if (rc) memset(&mem0, 0, sizeof(mem0));
	return rc; 
}

/* Return true if the heap is currently under memory pressure - in other words if the amount of heap used is close to the limit set by alloc_softheaplimit(). */
__host__ __device__ bool allocHeapNearlyFull() //: sqlite3HeapNearlyFull
{
	return mem0.NearlyFull;
}

/* Deinitialize the memory allocation subsystem. */
__host__ __device__ void allocShutdown() //: sqlite3MallocEnd
{
	if (__allocsystem.Shutdown) {
		__allocsystem.Shutdown(__allocsystem.AppData);
	}
	memset(&mem0, 0, sizeof(mem0));
}

/* Return the amount of memory currently checked out. */
__host__ __device__ int64_t alloc_memoryused() //: sqlite3_memory_used
{
	int64_t res, max;
	status64(STATUS_MEMORY_USED, &res, &max, false);
	return res;
}

/* Return the maximum amount of memory that has ever been checked out since either the beginning of this process or since the most recent reset. */
__host__ __device__ int64_t alloc_memoryhighwater(bool resetFlag) //: sqlite3_memory_highwater
{
	int64_t res, max;
	status64(STATUS_MEMORY_USED, &res, &max, resetFlag);
	return max;
}

/* Trigger the alarm */
static __host__ __device__ void allocAlarm(size_t bytes)
{
	if ( mem0.AlarmThreshold <= 0) return;
	mutex_leave(mem0.mutex);
	alloc_releasememory(bytes);
	mutex_enter(mem0.Mutex);
}

/* Do a memory allocation with statistics and alarms.  Assume the lock is already held. */
static __host__ __device__ void allocWithAlarm(size_t size, void **pp)
{
	assert(mutex_held(mem0.Mutex));
	assert(size > 0);

	// In Firefox (circa 2017-02-08), xRoundup() is remapped to an internal implementation of malloc_good_size(), which must be called in debug
	// mode and specifically when the DMD "Dark Matter Detector" is enabled or else a crash results.  Hence, do not attempt to optimize out the
	// following xRoundup() call.
	size_t fullSize = __allocsystem.Roundup(size);

#ifdef MAX_MEMORY
	if (status_value(STATUS_MEMORY_USED) + fullSize > MAX_MEMORY) {
		*pp = nullptr;
		return;
	}
#endif
	status_set(STATUS_MALLOC_SIZE, size);
	if (mem0.AlarmThreshold > 0) {
		int64_t used = status_value(STATUS_MEMORY_USED);
		if (used >= mem0.AlarmThreshold - fullSize)
		{
			mem0.NearlyFull = true;
			allocAlarm(fullSize);
		}
		else mem0.NearlyFull = false;
	}
	void *p = __allocsystem.Alloc(fullSize);
#ifdef ENABLE_MEMORY_MANAGEMENT
	if (!p && mem0.AlarmThreshold > 0) {
		allocAlarm(fullSize);
		p = __allocsystem.Alloc(fullSize);
	}
#endif
	if (p) {
		fullSize = allocSize(p);
		status_inc(STATUS_MEMORY_USED, fullSize);
		status_inc(STATUS_MALLOC_COUNT, 1);
	}
	*pp = p;
}

/*
** Allocate memory.  This routine is like alloc() except that it assumes the memory subsystem has already been initialized.
*/
__host__ __device__ void *alloc_(size_t size) //: sqlite3Malloc
{
	void *p;
	if (!size || size >= 0x7fffff00)
		/* A memory allocation of a number of bytes which is near the maximum signed integer value might cause an integer overflow inside of the
		** _.Alloc().  Hence we limit the maximum size to 0x7fffff00, giving 255 bytes of overhead.  Libcu itself will never use anything near
		** this amount.  The only way to reach the limit is with sqlite3_malloc() */
		p = nullptr;
	else if (g_runtimeStatics.Memstat) {
		mutex_enter(mem0.Mutex);
		allocWithAlarm(size, &p);
		mutex_leave(mem0.Mutex);
	}
	else p = __allocsystem.Alloc(size);
	assert(_HASALIGNMENT8(p)); // IMP: R-04675-44850
	return p;
}

/* This version of the memory allocation is for use by the application. First make sure the memory subsystem is initialized, then do the allocation. */
__host__ __device__ void *alloc(int n) //: sqlite3_malloc
{
#ifndef OMIT_AUTOINIT
	if (systemInitialize()) return nullptr;
#endif
	return n <= 0 ? nullptr : alloc(n);
}
__host__ __device__ void *alloc64(uint64_t n) //: sqlite3_malloc64
{
#ifndef OMIT_AUTOINIT
	if (systemInitialize()) return nullptr;
#endif
	return alloc(n);
}

/*
** Each thread may only have a single outstanding allocation from _.ScratchMalloc().  We verify this constraint in the single-threaded
** case by setting scratchAllocOut to 1 when an allocation is outstanding clearing it when the allocation is freed.
*/
#if THREADSAFE == 0 && !defined(NDEBUG)
static __host__ __device__ int g_scratchAllocOut = 0;
#endif

/*
** Allocate memory that is to be used and released right away. This routine is similar to alloca() in that it is not intended
** for situations where the memory might be held long-term.  This routine is intended to get memory to old large transient data
** structures that would not normally fit on the stack of an embedded processor.
*/
__host__ __device__ void *scratch_alloc(size_t size) //: sqlite3ScratchMalloc
{
	assert(size > 0);
	mutex_enter(mem0.Mutex);
	void *p;
	status_set(STATUS_SCRATCH_SIZE, size);
	if (mem0.ScratchFreeLength && g_runtimeStatics.ScratchSize >= size)
	{
		p = mem0.ScratchFree;
		mem0.ScratchFree = mem0.ScratchFree->Next;
		mem0.ScratchFreeLength--;
		status_inc(STATUS_SCRATCH_USED, 1);
		mutex_leave(mem0.Mutex);
	}
	else {
		mutex_leave(mem0.Mutex);
		p = alloc_(size);
		if (g_runtimeStatics.Memstat && p) {
			mutex_enter(mem0.Mutex);
			status_inc(STATUS_SCRATCH_OVERFLOW, alloc_size(p));
			mutex_leave(mem0.Mutex);
		}
		memdbg_settype(p, MEMTYPE_SCRATCH);
	}
	assert(mutex_notheld(mem0.Mutex));
#if THREADSAFE == 0 && !defined(NDEBUG)
	/* EVIDENCE-OF: R-12970-05880 Libcu will not use more than one scratch buffers per thread.
	**
	** This can only be checked in single-threaded mode.
	*/
	assert(!g_scratchAllocOut);
	if (p) g_scratchAllocOut++;
#endif
	return p;
}

__device__ void scratch_free(void *p) //: sqlite3ScratchFree
{
	if (p) {
#if THREADSAFE == 0 && !defined(NDEBUG)
		// Verify that no more than two scratch allocation per thread is outstanding at one time.  (This is only checked in the
		// single-threaded case since checking in the multi-threaded case would be much more complicated.)
		assert(g_scratchAllocOut >= 1 && g_scratchAllocOut <= 2);
		g_scratchAllocOut--;
#endif
		if (_WITHIN(p, g_runtimeStatics.Scratch, mem0.ScratchEnd))
		{
			// Release memory from the CONFIG_SCRATCH allocation
			ScratchFreeslot *slot = (ScratchFreeslot *)p;
			mutex_enter(mem0.Mutex);
			slot->Next = mem0.ScratchFree;
			mem0.ScratchFree = slot;
			mem0.ScratchFreeLength++;
			assert(mem0.ScratchFreeLength <= (uint32_t)g_runtimeStatics.Scratchs);
			status_dec(STATUS_SCRATCH_USED, 1);
			mutex_leave(mem0.Mutex);
		}
		else {
			// Release memory back to the heap
			assert(memdbg_hastype(p, MEMTYPE_SCRATCH));
			assert(memdbg_nottype(p, (uint8_t)~MEMTYPE_SCRATCH));
			memdbg_settype(p, MEMTYPE_HEAP);
			if (g_runtimeStatics.Memstat) {
				size_t size = alloc_size(p);
				mutex_enter(mem0.Mutex);
				status_dec(STATUS_SCRATCH_OVERFLOW, size);
				status_dec(STATUS_MEMORY_USED, size);
				status_dec(STATUS_MALLOC_COUNT, 1);
				__allocsystem.Free(p);
				mutex_leave(mem0.Mutex);
			}
			else __allocsystem.Free(p);
		}
	}
}

/* TRUE if p is a lookaside memory allocation from tag */
#ifndef OMIT_LOOKASIDE
static __host__ __device__ bool IsLookaside(tagbase_t *tag, void *p) { return _WITHIN(p, tag->Lookaside.Start, tag->Lookaside.End); }
#else
#define IsLookaside(A,B) false
#endif

/*
/* Return the size of a memory allocation previously obtained from alloc_() or alloc(). */
*/
	__host__ __device__ size_t alloc_size(void *p) //: sqlite3MallocSize
{
	assert(memdbg_hastype(p, MEMTYPE_HEAP));
	return __allocsystem.Size(p);
}

__device__ size_t alloc_tagsize(tagbase_t *tag, void *p) //: sqlite3DbMallocSize
{
	assert(p);
	if (!tag || !IsLookaside(tag, p)) {
#ifdef DEBUG
		if (!tag) {
			assert(memdbg_nottype(p, (uint8_t)~MEMTYPE_HEAP));
			assert(memdbg_hastype(p, MEMTYPE_HEAP));
		}
		else {
			assert(memdbg_hastype(p, (MEMTYPE_LOOKASIDE | MEMTYPE_HEAP)));
			assert(memdbg_nottype(p, (uint8_t)~(MEMTYPE_LOOKASIDE | MEMTYPE_HEAP)));
		}
#endif
		return __allocsystem.Size(p);
	}
	else {
		assert(mutex_held(tag->Mutex));
		return tag->Lookaside.Size;
	}
	_assert(_memdbg_hastype(p, MEMTYPE_TAG));
	_assert(_memdbg_hastype(p, (MEMTYPE)(MEMTYPE_LOOKASIDE | MEMTYPE_HEAP)));
	_assert(tag || memdbg_nottype(p, MEMTYPE_LOOKASIDE));
	return __allocsystem.Size(p);
}

__host__ __device__ uint64_t alloc_msize(void *p) //: sqlite3_msize
{
	assert(memdbg_notype(p, (uint8_t)~MEMTYPE_HEAP));
	assert(memdbg_hasType(p, MEMTYPE_HEAP));
	return p ? __allocsystem.Size(p) : nullptr;
}

/* Free memory previously obtained from alloc(). */
__device__ void alloc_free(void *p) //: sqlite3_free
{
	if (!p) return; // IMP: R-49053-54554
	assert(memdbg_hastype(p, MEMTYPE_HEAP));
	assert(memdbg_nottype(p, (uint8_t)~MEMTYPE_HEAP));
	if (g_runtimeStatics.Memstat) {
		mutex_enter(mem0.Mutex);
		status_dec(STATUS_MEMORY_USED, alloc_size(p));
		status_dec(STATUS_MALLOC_COUNT, 1);
		__allocsystem.Free(p);
		mutex_leave(mem0.Mutex);
	}
	else __allocsystem.Free(p);
}

/* Add the size of memory allocation "p" to the count in *tag->BytesFreed. */
static __host__ __device__ void MeasureAllocationSize(tagbase_t *tag, void *p) { *tag->BytesFreed += alloc_tagsize(tag, p); }

/*
** Free memory that might be associated with a particular database connection.  Calling alloc_tagfree(D,X) for X==0 is a harmless no-op.
** The alloc_tagfreeNN(D,X) version requires that X be non-NULL.
*/
__host__ __device__ void alloc_tagfreeNN(tagbase_t *tag, void *p) //: sqlite3DbFreeNN
{
	assert(!tag || mutex_held(tag->Mutex));
	assert(p);
	if (tag) {
		if (tag->BytesFreed) {
			MeasureAllocationSize(tag, p);
			return;
		}
		if (IsLookaside(tag, p)) {
			LookasideSlot *b = (LookasideSlot *)p;
#if DEBUG
			// Trash all content in the buffer being freed
			memset(p, 0xaa, tag->Lookaside.Size);
#endif
			b->Next = tag->Lookaside.Free;
			tag->Lookaside.Free = b;
			tag->Lookaside.Outs--;
			return;
		}
	}
	assert(memdbg_hastype(p, (MEMTYPE_LOOKASIDE | MEMTYPE_HEAP)));
	assert(memdbg_nottype(p, (uint8_t)~(MEMTYPE_LOOKASIDE | MEMTYPE_HEAP)));
	assert(tag || memdbg_nottype(p, MEMTYPE_LOOKASIDE));
	memdbg_settype(p, MEMTYPE_HEAP);
	alloc_free(p);
}

__host__ __device__ void alloc_tagfree(tagbase_t *tag, void *p) //: sqlite3DbFree
{ 
	assert(!tag || mutex_held(tag->Mutex));
	if (p) alloc_tagfreeNN(tag, p);
}


/* Change the size of an existing memory allocation */
__host__ __device__ void *alloc_realloc_(void *old, uint64_t newSize) //: sqlite3Realloc
{
	assert(memdbg_hastype(old, MEMTYPE_HEAP));
	assert(memdbg_nottype(old, (uint8_t)~MEMTYPE_HEAP));
	if (!old) return alloc_(newSize); // IMP: R-04300-56712
	if (!newSize) { alloc_free(old); return nullptr; } // IMP: R-26507-47431
	if (newSize >= 0x7fffff00) return nullptr; // The 0x7ffff00 limit term is explained in comments on alloc_()
	size_t oldSize = alloc_size(old);
	// IMPLEMENTATION-OF: R-46199-30249 Libcu guarantees that the second argument to _.xRealloc is always a value returned by a prior call to _.Roundup.
	void *p;
	size_t newSize2 = __allocsystem.Roundup(newSize);
	if (oldSize == newSize2)
		p = old;
	else if (g_runtimeStatics.Memstat) {
		mutex_enter(mem0.Mutex);
		status_set(STATUS_MALLOC_SIZE, (int)newSize);
		int sizeDiff = newSize2 - oldSize;
		if (sizeDiff > 0 && status_value(STATUS_MEMORY_USED) >= mem0.AlarmThreshold - sizeDiff)
			allocAlarm(sizeDiff);
		p = __allocsystem.Realloc(old, newSize2);
		if (!p && mem0.AlarmThreshold > 0) {
			allocAlarm(newSize);
			p = __allocsystem.Realloc(old, newSize2);
		}
		if (p) {
			newSize2 = alloc_size(p);
			status_inc(STATUS_MEMORY_USED, newSize2 - oldSize);
		}
		mutex_leave(mem0.Mutex);
	}
	else p = __allocsystem.Realloc(old, newSize2);
	assert(HASALIGNMENT8(p)); // IMP: R-11148-40995
	return p;
}

/* The public interface to alloc_realloc_.  Make sure that the memory subsystem is initialized prior to invoking alloc_realloc_. */
__host__ __device__ void *alloc_realloc(void *old, int newSize) //: sqlite3_realloc
{
#ifndef OMIT_AUTOINIT
	if (systemInitialize()) return nullptr;
#endif
	if (newSize < 0) newSize = 0;  // IMP: R-26507-47431
	return alloc_realloc_(p, newSize);
}

__host__ __device__ void *alloc_realloc64(void *p, uint64_t newSize)
{
#ifndef OMIT_AUTOINIT
	if (systemInitialize()) return nullptr;
#endif
	return alloc_realloc_(p, newSize);
}

/* Allocate and zero memory. */
__host__ __device__ void *allocZero(uint64_t size) //: sqlite3MallocZero
{
	void *p = alloc_(n);
	if (p) memset(p, 0, (size_t)size);
	return p;
}

/* Allocate and zero memory.  If the allocation fails, make the mallocFailed flag in the connection pointer. */
__device__ void *tagallocZero(tagbase_t *tag, uint64_t size) //: sqlite3DbMallocZero
{
	ASSERTCOVERAGE(tag);
	void *p = alloc_tagraw(tag, size);
	if (p) memset(p, 0, (size_t)size);
	return p;
}

/* Finish the work of tagallocrawNN for the unusual and slower case when the allocation cannot be fulfilled using lookaside. */
static __host__ __device__ void *tagallocrawFinish(tagbase_t *tag, uint64_t size)
{
	assert(tag);
	void *p = alloc_(size);
	if (!p) sqlite3OomFault(db);
	memdbg_settype(p, (!tag->Lookaside.Disable ? MEMTYPE_LOOKASIDE : MEMTYPE_HEAP));
	return p;
}

/*
** Allocate memory, either lookaside (if possible) or heap.   If the allocation fails, set the MallocFailed flag in
** the connection pointer.
**
** If tag!=0 and tag->MallocFailed is true (indicating a prior malloc failure on the same database connection) then always return nullptr.
** Hence for a particular tag, once malloc starts failing, it fails consistently until MallocFailed is reset.
** This is an important assumption.  There are many places in the code that do things like this:
**
**         int *a = (int*)sqlite3DbMallocRaw(db, 100);
**         int *b = (int*)sqlite3DbMallocRaw(db, 200);
**         if( b ) a[10] = 9;
**
** In other words, if a subsequent malloc (ex: "b") worked, it is assumed that all prior mallocs (ex: "a") worked too.
**
** The sqlite3MallocRawNN() variant guarantees that the "db" parameter is not a NULL pointer.
*/
__host__ __device__ void *tagallocraw(tagbase_t *tag, uint64_t size) //: sqlite3DbMallocRaw
{
	if (tag) return tagallocrawNN(tag, size);
	void *p = alloc_(size);
	memdbg_settype(p, MEMTYPE_HEAP);
	return p;
}

__host__ __device__ void *tagallocrawNN(tagbase_t *tag, uint64_t size) //: sqlite3DbMallocRawNN
{
#ifndef OMIT_LOOKASIDE
	assert(tag);
	assert(mutex_held(tag->Mutex));
	assert(!tag->BytesFreed);
	if (!tag->Lookaside.Disable) {
		LookasideSlot *b;
		assert(!tag->MallocFailed);
		if (size > tag->Lookaside.Size)
			tag->Lookaside.Stats[1]++;
		else if (!(b = tag->Lookaside.Free))
			tag->lookaside.Stats[2]++;
		else {
			tag->Lookaside.Free = b->Next;
			tag->Lookaside.Outs++;
			tag->Lookaside.Stats[0]++;
			if (tag->Lookaside.Outs > tag->Lookaside.MaxOuts)
				tag->Lookaside.MaxOuts = tag->Lookaside.Outs;
			return (void *)b;
		}
	else if (tag->MallocFailed)
		return nullptr;
#else
	assert(tag);
	assert(mutex_held(tag->Mutex));
	assert(!tag->BytesFreed);
	if (tag->MallocFailed)
		return nullptr;
#endif
	return tagallocrawFinish(db, n);
}

































// Resize the block of memory pointed to by old to size bytes. If the resize fails, set the mallocFailed flag in the connection object.
__device__ void *_tagreallocG(TagBase *tag, void *old, size_t size)
{
	void *p = nullptr;
	_assert(tag != nullptr);
	_assert(_mutex_held(tag->Mutex));
	if (!tag->MallocFailed)
	{
		if (!old) return _tagallocG(tag, size);
		if (IsLookaside(tag, old))
		{
			if (size <= tag->Lookaside.Size) return old;
			p = _tagallocG(tag, size);
			if (p)
			{
				memcpy(p, old, tag->Lookaside.Size);
				_tagfreeG(tag, old);
			}
		}
		else
		{
			_assert(_memdbg_hastype(old, MEMTYPE_DB));
			_assert(_memdbg_hastype(old, (MEMTYPE)(MEMTYPE_LOOKASIDE|MEMTYPE_HEAP)));
			_memdbg_settype(old, MEMTYPE_HEAP);
			p = _reallocG(old, size);
			if (!p)
			{
				_memdbg_settype(old, (MEMTYPE)(MEMTYPE_DB|MEMTYPE_HEAP));
				tag->MallocFailed = true;
			}
			_memdbg_settype(p, (MEMTYPE)(MEMTYPE_DB|(tag->Lookaside.Enabled ? MEMTYPE_LOOKASIDE : MEMTYPE_HEAP)));
		}
	}
	return p;
}

#ifdef OMIT_INLINEMEM
#if 0
// Attempt to reallocate p.  If the reallocation fails, then free p and set the mallocFailed flag in the database connection.
__device__ void *_tagrealloc_or_freeG(TagBase *tag, void *old, size_t newSize)
{
	void *p = _tagreallocG(tag, old, newSize);
	if (!p) _tagfreeG(tag, old);
	return p;
}
#endif

__device__ char *__strdupG(const char *z)
{
	if (!z) return nullptr;
	size_t n = _strlen(z) + 1;
	_assert((n & 0x7fffffff) == n);
	char *newZ = (char *)_allocG((int)n);
	if (newZ) memcpy(newZ, (char *)z, n);
	return newZ;
}

__device__ char *_strndupG(const char *z, int n)
{
	if (!z) return nullptr;
	_assert((n & 0x7fffffff) == n);
	char *newZ = (char *)_allocG(n + 1);
	if (newZ) { memcpy(newZ, (char *)z, n); newZ[n] = 0; }
	return newZ;
}

__device__ char *_tagstrdupG(TagBase *tag, const char *z)
{
	if (!z) return nullptr;
	size_t n = _strlen(z) + 1;
	_assert((n & 0x7fffffff) == n);
	char *newZ = (char *)_tagallocG(tag, (int)n);
	if (newZ) memcpy(newZ, (char *)z, n);
	return newZ;
}

__device__ char *_tagstrndupG(TagBase *tag, const char *z, int n)
{
	if (!z) return nullptr;
	_assert((n & 0x7fffffff) == n);
	char *newZ = (char *)_tagallocG(tag, n + 1);
	if (newZ) { memcpy(newZ, (char *)z, n); newZ[n] = 0; }
	return newZ;
}
#endif

RUNTIME_NAMEEND