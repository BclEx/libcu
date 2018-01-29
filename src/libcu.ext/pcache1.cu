#include <ext/global.h> //: pcache1.c
#include <ext/pcache.h>
#include <assert.h>

#pragma region Struct

typedef struct PCache1 PCache1;
typedef struct PgHdr1 PgHdr1;
typedef struct PgFreeslot PgFreeslot;
typedef struct PGroup PGroup;

/* Each cache entry is represented by an instance of the following structure. Unless SQLITE_PCACHE_SEPARATE_HEADER is defined, a buffer of
** PgHdr1.pCache->szPage bytes is allocated directly before this structure in memory.
*/
struct PgHdr1 {
	pcache_page_t page;     // Base class. Must be first. pBuf & pExtra */
	uint key;				// Key value (page number)
	uint8_t isBulkLocal;    // This page from bulk local storage
	uint8_t isAnchor;       // This is the PGroup.lru element
	PgHdr1 *next;			// Next in hash table chain
	PCache1 *cache;			// Cache that currently owns this page
	PgHdr1 *lruNext;		// Next in LRU list of unpinned pages
	PgHdr1 *lruPrev;		// Previous in LRU list of unpinned pages
};

/* A page is pinned if it is no on the LRU list */
#define PAGE_IS_PINNED(p)    ((p)->lruNext==0)
#define PAGE_IS_UNPINNED(p)  ((p)->lruNext!=0)

/* Each page cache (or PCache) belongs to a PGroup.  A PGroup is a set of one or more PCaches that are able to recycle each other's unpinned
** pages when they are under memory pressure.
*/
struct PGroup {
	mutex *mutex;			// MUTEX_STATIC_LRU or NULL
	uint maxPages;			// Sum of nMax for purgeable caches
	uint minPages;			// Sum of nMin for purgeable caches
	uint maxPinned;			// nMaxpage + 10 - nMinPage
	uint purgeables;		// Number of purgeable pages allocated
	PgHdr1 lru;             // The beginning and end of the LRU list
};

/* Each page cache is an instance of the following object.  Every open database file (including each in-memory database and each
** temporary or transient database) has a single page cache which is an instance of this object.
**
** Pointers to structures of this type are cast and returned as opaque sqlite3_pcache* handles.
*/
struct PCache1 {
	// Cache configuration parameters. Page size (szPage) and the purgeable flag (bPurgeable) and the pnPurgeable pointer are all set when the
	// cache is created and are never changed thereafter. nMax may be modified at any time by a call to the pcache1Cachesize() method.
	// The PGroup mutex must be held when accessing nMax.
	PGroup *group;              // PGroup this cache belongs to
	uint *purgeables;			// Pointer to pGroup->nPurgeable
	int sizePage;               // Size of database content section
	int sizeExtra;              // sizeof(MemPage)+sizeof(PgHdr)
	int sizeAlloc;              // Total size of one pcache line
	int purgeable;				// True if cache is purgeable
	uint min;					// Minimum number of pages reserved
	uint max;					// Configured "cache_size" value
	uint n90pct;				// nMax*9/10
	uint maxKey;				// Largest key seen since xTruncate()
	// Hash table of all pages. The following variables may only be accessed when the accessor is holding the PGroup mutex.
	uint recyclables;           // Number of pages in the LRU list
	uint pages;                 // Total number of pages in apHash
	uint hashs;                 // Number of slots in apHash[]
	PgHdr1 **hash;				// Hash table for fast lookup by key
	PgHdr1 *free;				// List of unused pcache-local pages
	void *bulk;					// Bulk memory used by pcache-local
};

/* Free slots in the allocator used to divide up the global page cache buffer provided using the SQLITE_CONFIG_PAGECACHE mechanism. */
struct PgFreeslot {
	PgFreeslot *next;		// Next free slot
};

/* Global data used by this cache. */
struct PCacheGlobal {
	PGroup group;			// The global PGroup for mode (2)
	// Variables related to SQLITE_CONFIG_PAGECACHE settings.  The szSlot, nSlot, pStart, pEnd, nReserve, and isInit values are all
	// fixed at sqlite3_initialize() time and do not require mutex protection. The nFreeSlot and pFree values do require mutex protection.
	bool isInit;                    // True if initialized
	int separateCache;				// Use a new PGroup for each PCache
	int initPages;					// Initial bulk allocation size
	int sizeSlot;                   // Size of each free slot
	int slots;						// The number of pcache slots
	int reserves;					// Try to keep nFreeSlot above this
	void *start, *end;				// Bounds of global page cache memory
	// Above requires no mutex.  Use mutex below for variable that follow.
	mutex *mutex;					// mutex for accessing the following:
	PgFreeslot *free;				// Free page blocks
	int freeSlots;                 // Number of unused pcache slots
	// The following value requires a mutex to change.  We skip the mutex on reading because (1) most platforms read a 32-bit integer atomically and
	// (2) even if an incorrect value is read, no great harm is done since this is really just an optimization. */
	bool underPressure;            // True if low on PAGECACHE memory
};
static __hostb_device__ _WSD PCacheGlobal g_pcache1;
#define _pcache1 _GLOBAL(PCacheGlobal, g_pcache1)

/* Macros to enter and leave the PCache LRU mutex. */
#if !defined(ENABLE_MEMORY_MANAGEMENT) || LIBCU_THREADSAFE == 0
#define pcache1EnterMutex(x) assert(!(x)->mutex)
#define pcache1LeaveMutex(x) assert(!(x)->mutex)
#define PCACHE1_MIGHT_USE_GROUP_MUTEX 0
#else
#define pcache1EnterMutex(X) mutex_enter((x)->mutex)
#define pcache1LeaveMutex(X) mutex_leave((x)->mutex)
#define PCACHE1_MIGHT_USE_GROUP_MUTEX 1
#endif

#pragma endregion

#pragma region Page Allocation/CONFIG_PCACHE Related Functions

__host_device__ void pcacheBufferSetup(void *buf, int size, int n) // sqlite3PCacheBufferSetup
{
	if (_pcache1.isInit) {
		if (!buf) size = n = 0;
		if (!n) size = 0;
		size = _ROUNDDOWN8(size);
		_pcache1.sizeSlot = size;
		_pcache1.slots = _pcache1.freeSlots = n;
		_pcache1.reserves = n > 90 ? 10 : (n / 10 + 1);
		_pcache1.start = buf;
		_pcache1.free = nullptr;
		_pcache1.underPressure = false;
		while (n--) {
			PgFreeslot *p = (PgFreeslot *)buf;
			p->next = _pcache1.free;
			_pcache1.free = p;
			buf = (void *)&((char *)buf)[size];
		}
		_pcache1.end = buf;
	}
}

/* Try to initialize the pCache->pFree and pCache->pBulk fields.  Return true if pCache->pFree ends up containing one or more free pages. */
static __host_device__ int pcache1InitBulk(PCache1 *cache)
{
	if (!_pcache1.initPages) return 0;
	// Do not bother with a bulk allocation if the cache size very small
	if (cache->max < 3) return 0;
	allocBenignBegin();
	int64_t sizeBulk = (_pcache1.initPages > 0 ? cache->sizeAlloc * (int64_t)_pcache1.initPages : -1024 * (int64_t)_pcache1.initPages);
	if (sizeBulk > cache->sizeAlloc * (int64_t)cache->max)
		sizeBulk = cache->sizeAlloc * (int64_t)cache->max;
	char *bulk = (char *)(cache->bulk = alloc(sizeBulk));
	allocBenignEnd();
	if (bulk) {
		int bulks = allocSize(bulk) / cache->sizeAlloc;
		do {
			PgHdr1 *p = (PgHdr1 *)&bulk[cache->sizePage];
			p->page.buf = bulk;
			p->page.extra = &p[1];
			p->isBulkLocal = 1;
			p->isAnchor = 0;
			p->next = cache->free;
			cache->free = p;
			bulk += cache->sizeAlloc;
		} while (--bulks);
	}
	return cache->free != 0;
}

/* Malloc function used within this file to allocate space from the buffer configured using sqlite3_config(SQLITE_CONFIG_PAGECACHE) option. If no 
** such buffer exists or there is no space left in it, this function falls back to sqlite3Malloc().
**
** Multiple threads can run this routine at the same time.  Global variables in pcache1 need to be protected via mutex.
*/
static __host_device__ void *pcache1Alloc(int bytes)
{
	assert(mutex_notheld(_pcache1.group.mutex));
	void *p = nullptr;
	if (bytes <= _pcache1.sizeSlot) {
		mutex_enter(_pcache1.mutex);
		p = (PgHdr1 *)_pcache1.free;
		if (p) {
			_pcache1.free = _pcache1.free->next;
			_pcache1.freeSlots--;
			_pcache1.underPressure = _pcache1.freeSlots < _pcache1.reserves;
			assert(_pcache1.freeSlots >= 0);
			status_max(STATUS_PAGECACHE_SIZE, bytes);
			status_inc(STATUS_PAGECACHE_USED, 1);
		}
		mutex_leave(_pcache1.mutex);
	}
	if (!p) {
		// Memory is not available in the SQLITE_CONFIG_PAGECACHE pool.  Get it from sqlite3Malloc instead.
		p = alloc(bytes);
#ifndef DISABLE_PAGECACHE_OVERFLOW_STATS
		if (p) {
			int size = allocSize(p);
			mutex_enter(_pcache1.mutex);
			status_max(STATUS_PAGECACHE_SIZE, bytes);
			status_inc(STATUS_PAGECACHE_OVERFLOW, size);
			mutex_leave(_pcache1.mutex);
		}
#endif
		memdbg_settype(p, MEMTYPE_PCACHE);
	}
	return p;
}

/* Free an allocated buffer obtained from pcache1Alloc(). */
static __host_device__ void pcache1Free(void *p)
{
	if (!p) return;
	if (_WITHIN(p, _pcache1.start, _pcache1.end)) {
		mutex_enter(_pcache1.mutex);
		status_dec(STATUS_PAGECACHE_USED, 1);
		PgFreeslot *slot = (PgFreeslot *)p;
		slot->next = _pcache1.free;
		_pcache1.free = slot;
		_pcache1.freeSlots++;
		_pcache1.underPressure = _pcache1.freeSlots < _pcache1.reserves;
		assert(_pcache1.freeSlots <= _pcache1.slots);
		mutex_leave(_pcache1.mutex);
	}
	else {
		assert(memdbg_hastype(p, MEMTYPE_PCACHE));
		memdbg_settype(p, MEMTYPE_HEAP);
#ifndef DISABLE_PAGECACHE_OVERFLOW_STATS
		{
			int freed = allocSize(p);
			mutex_enter(_pcache1.mutex);
			status_dec(STATUS_PAGECACHE_OVERFLOW, freed);
			mutex_leave(_pcache1.mutex);
		}
#endif
		mfree(p);
	}
}

#ifdef ENABLE_MEMORY_MANAGEMENT
static __host_device__ int pcache1MemSize(void *p)
{
	if (p >= _pcache1.start && p < _pcache1.end)
		return _pcache1.sizeSlot;
	assert(memdbg_hastype(p, MEMTYPE_PCACHE));
	memdbg_settype(p, MEMTYPE_HEAP);
	int size = allocSize(p);
	memdbg_settype(p, MEMTYPE_PCACHE);
	return size;
}
#endif

static __host_device__ PgHdr1 *pcache1AllocPage(PCache1 *cache, bool benignMalloc)
{
	// The group mutex must be released before pcache1Alloc() is called. This is because it may call sqlite3_release_memory(), which assumes that this mutex is not held.
	assert(mutex_held(cache->group->mutex));
	PgHdr1 *p = nullptr;
	if (cache->free || (!cache->pages && pcache1InitBulk(cache))) {
		p = cache->free;
		cache->free = p->next;
		p->next = nullptr;
	}
	else {
#ifdef ENABLE_MEMORY_MANAGEMENT
		// The group mutex must be released before pcache1Alloc() is called. This is because it might call sqlite3_release_memory(), which assumes that 
		// this mutex is not held.
		assert(!_pcache1.separateCache);
		assert(cache->group == &_pcache1.group);
		pcache1LeaveMutex(cache->group);
#endif
		if (benignMalloc) allocBenignBegin();
		void *pg;
#ifdef PCACHE_SEPARATE_HEADER
		pg = pcache1Alloc(cache->sizePage);
		p = (PgHdr1 *)alloc(sizeof(PgHdr1) + cache->sizeExtra);
		if (!pg || !p) {
			pcache1Free(pg);
			mfree(p);
			pg = nullptr;
		}
#else
		pg = pcache1Alloc(cache->sizeAlloc);
		p = (PgHdr1 *)&((uint8_t *)pg)[cache->sizePage];
#endif
		if (benignMalloc) allocBenignEnd();
#ifdef ENABLE_MEMORY_MANAGEMENT
		pcache1EnterMutex(cache->group);
#endif
		if (!pg) return nullptr;
		p->page.buf = pg;
		p->page.extra = &p[1];
		p->isBulkLocal = 0;
		p->isAnchor = 0;
	}
	(*cache->purgeables)++;
	return p;
}

/* Free a page object allocated by pcache1AllocPage(). */
static __host_device__ void pcache1FreePage(PgHdr1 *p)
{
	assert(p);
	PCache1 *cache = p->cache;
	assert(mutex_held(p->cache->group->mutex));
	if (p->isBulkLocal) {
		p->next = cache->free;
		cache->free = p;
	}
	else {
		pcache1Free(p->page.buf);
#ifdef PCACHE_SEPARATE_HEADER
		mfree(p);
#endif
	}
	(*cache->purgeables)--;
}

/* Malloc function used by SQLite to obtain space from the buffer configured using sqlite3_config(SQLITE_CONFIG_PAGECACHE) option. If no such buffer
** exists, this function falls back to sqlite3Malloc().
*/
__host_device__ void *sqlite3PageMalloc(int size) { return pcache1Alloc(size); } //: sqlite3PageMalloc

/* Free an allocated buffer obtained from sqlite3PageMalloc(). */
__host_device__ void sqlite3PageFree(void *p) { pcache1Free(p); } //: sqlite3PageFree

/* Return true if it desirable to avoid allocating a new page cache entry.
**
** If memory was allocated specifically to the page cache using SQLITE_CONFIG_PAGECACHE but that memory has all been used, then
** it is desirable to avoid allocating a new page cache entry because presumably SQLITE_CONFIG_PAGECACHE was suppose to be sufficient
** for all page cache needs and we should not need to spill the allocation onto the heap.
**
** Or, the heap is used for all page cache memory but the heap is under memory pressure, then again it is desirable to avoid
** allocating a new page cache entry in order to avoid stressing the heap even further.
*/
static __host_device__ bool pcache1UnderMemoryPressure(PCache1 *cache)
{
	return _pcache1.slots && (cache->sizePage + cache->sizeExtra) <= _pcache1.sizeSlot ? _pcache1.underPressure : allocHeapNearlyFull();
}

#pragma endregion

#pragma region General Implementation Functions

static __host_device__ void pcache1ResizeHash(PCache1 *p)
{
	assert(mutex_held(p->group->mutex));
	uint newHashs = p->hashs * 2;
	if (newHashs < 256)
		newHashs = 256;
	mutex_leave(p->group->mutex);
	if (p->hashs) allocBenignBegin();
	PgHdr1 **newHash = (PgHdr1 **)allocZero(sizeof(PgHdr1 *) * newHashs);
	if (p->hashs) allocBenignEnd();
	mutex_enter(p->group->mutex);
	if (newHash) {
		for (uint i = 0; i < (uint)p->hashs; i++) {
			PgHdr1 *page;
			PgHdr1 *next = p->hash[i];
			while (page = next) {
				uint h = page->key % newHashs;
				next = page->next;
				page->next = newHash[h];
				newHash[h] = page;
			}
		}
		mfree(p->hash);
		p->hash = newHash;
		p->hashs = newHashs;
	}
}

/* This function is used internally to remove the page pPage from the PGroup LRU list, if is part of it. If pPage is not part of the PGroup
** LRU list, then this function is a no-op.
**
** The PGroup mutex must be held when this function is called.
*/
static __host_device__ PgHdr1 *pcache1PinPage(PgHdr1 *page)
{
	assert(page);
	assert(PAGE_IS_UNPINNED(page));
	assert(page->lruNext);
	assert(page->lruPrev);
	assert(mutex_held(page->cache->group->mutex));
	page->lruPrev->lruNext = page->lruNext;
	page->lruNext->lruPrev = page->lruPrev;
	page->lruNext = nullptr;
	page->lruPrev = nullptr;
	assert(!page->isAnchor);
	assert(page->cache->group->lru.isAnchor);
	page->cache->recyclables--;
	return page;
}

/* Remove the page supplied as an argument from the hash table (PCache1.apHash structure) that it is currently stored in.
** Also free the page if freePage is true.
**
** The PGroup mutex must be held when this function is called.
*/
static __host_device__ void pcache1RemoveFromHash(PgHdr1 *page, bool freeFlag)
{
	PCache1 *cache = page->cache;
	assert(mutex_held(cache->group->mutex));
	uint h = page->key % cache->hashs;
	PgHdr1 **pp; for (pp = &cache->hash[h]; (*pp) != page; pp = &(*pp)->next);
	*pp = (*pp)->next;
	cache->pages--;
	if (freeFlag) pcache1FreePage(page);
}

/* If there are currently more than nMaxPage pages allocated, try to recycle pages to reduce the number allocated to nMaxPage. */
static __host_device__ void pcache1EnforceMaxPage(PCache1 *cache)
{
	PGroup *group = cache->group;
	assert(mutex_held(group->mutex));
	PgHdr1 *p; while (group->purgeables > group->maxPages && !(p = group->lru.lruPrev)->isAnchor) {
		assert(p->cache->group == group);
		assert(PAGE_IS_UNPINNED(p));
		pcache1PinPage(p);
		pcache1RemoveFromHash(p, true);
	}
	if (!cache->pages && cache->bulk) {
		mfree(cache->bulk);
		cache->bulk = cache->free = nullptr;
	}
}

/* Discard all pages from cache pCache with a page number (key value) greater than or equal to iLimit. Any pinned pages that meet this 
** criteria are unpinned before they are discarded.
**
** The PCache mutex must be held when this function is called.
*/
static __host_device__ void pcache1TruncateUnsafe(PCache1 *cache, uint limit)
{
	assert(mutex_held(cache->group->mutex));
	assert(cache->maxKey >= limit);
	assert(cache->hashs > 0);
	uint h, stop;
	ASSERTONLY(uint pages = 0);
	if (cache->maxKey - limit < cache->hashs) {
		// If we are just shaving the last few pages off the end of the cache, then there is no point in scanning the entire hash table.
		// Only scan those hash slots that might contain pages that need to be removed.
		h = limit % cache->hashs;
		stop = cache->maxKey % cache->hashs;
		ASSERTONLY(pages = -10); // Disable the pCache->nPage validity check
	}
	else {
		// This is the general case where many pages are being removed. It is necessary to scan the entire hash table */
		h = cache->hashs / 2;
		stop = h - 1;
	}
	for(;;) {
		assert(h < cache->hashs);
		PgHdr1 **pp = &cache->hash[h]; 
		PgHdr1 *page; while (page = *pp) {
			if (page->key >= limit) {
				cache->pages--;
				*pp = page->next;
				if (PAGE_IS_UNPINNED(page)) pcache1PinPage(page);
				pcache1FreePage(page);
			}
			else {
				pp = &page->next;
				ASSERTONLY(if (pages >= 0) pages++);
			}
		}
		if (h == stop) break;
		h = (h + 1) % cache->hashs;
	}
	assert(pages < 0 || cache->pages == pages);
}

#pragma endregion

#pragma region pcache Methods

/* Implementation of the sqlite3_pcache.xInit method. */
static __host_device__ RC pcache1Init(void *notUsed)
{
	UNUSED_SYMBOL(notUsed);
	assert(!_pcache1.isInit);
	memset(&_pcache1, 0, sizeof(_pcache1));

	// The pcache1.separateCache variable is true if each PCache has its own private PGroup (mode-1).  pcache1.separateCache is false if the single
	// PGroup in pcache1.grp is used for all page caches (mode-2).
	//   *  Always use a unified cache (mode-2) if ENABLE_MEMORY_MANAGEMENT
	//   *  Use a unified cache in single-threaded applications that have configured a start-time buffer for use as page-cache memory using
	//      sqlite3_config(SQLITE_CONFIG_PAGECACHE, pBuf, sz, N) with non-NULL pBuf argument.
	//   *  Otherwise use separate caches (mode-1)
#if defined(ENABLE_MEMORY_MANAGEMENT)
	_pcache1.separateCache = 0;
#elif LIBCU_THREADSAFE
	_pcache1.separateCache = _runtimeConfig.page == nullptr || _runtimeConfig.coreMutex;
#else
	_pcache1.separateCache = _runtimeConfig.page == nullptr;
#endif

#if LIBCU_THREADSAFE
	if (_runtimeConfig.coreMutex) {
		_pcache1.group.mutex = mutexAlloc(MUTEX_STATIC_LRU);
		_pcache1.mutex = mutexAlloc(MUTEX_STATIC_PMEM);
	}
#endif
	_pcache1.initPages = _pcache1.separateCache && _runtimeConfig.pages && !_runtimeConfig.page ? _runtimeConfig.pages : 0;
	_pcache1.group.maxPinned = 10;
	_pcache1.isInit = true;
	return RC_OK;
}

/* Implementation of the sqlite3_pcache.xShutdown method. Note that the static mutex allocated in xInit does 
** not need to be freed.
*/
static __host_device__ void pcache1Shutdown(void *notUsed)
{
	UNUSED_SYMBOL(notUsed);
	assert(_pcache1.isInit);
	memset(&_pcache1, 0, sizeof(_pcache1));
}

/* forward declaration */
static __host_device__ void pcache1Destroy(pcache_t *p);

static __host_device__ pcache_t *pcache1Create(int sizePage, int sizeExtra, int purgeable)
{
	assert((sizePage & (sizePage - 1)) == 0 && sizePage >= 512 && sizePage <= 65536);
	assert(sizeExtra < 300);
	int size = sizeof(PCache1) + sizeof(PGroup) * (int)_pcache1.separateCache; // Bytes of memory required to allocate the new cache
	PCache1 *cache = (PCache1 *)allocZero(size); // The newly created page cache
	PGroup *group; // The group the new page cache will belong to
	if (cache) {
		if (_pcache1.separateCache) { group = (PGroup *)&cache[1]; group->maxPinned = 10; }
		else group = &_pcache1.group;
		if (!group->lru.isAnchor) {
			group->lru.isAnchor = true;
			group->lru.lruPrev = group->lru.lruNext = &group->lru;
		}
		cache->group = group;
		cache->sizePage = sizePage;
		cache->sizeExtra = sizeExtra;
		cache->sizeAlloc = sizePage + sizeExtra + _ROUND8(sizeof(PgHdr1));
		cache->purgeable = purgeable;
		pcache1EnterMutex(group);
		pcache1ResizeHash(cache);
		if (purgeable) {
			cache->min = 10;
			group->minPages += cache->min;
			group->maxPinned = group->maxPages + 10 - group->minPages;
			cache->purgeables = &group->purgeables;
		}
		else {
			static uint dummyCurrentPage;
			cache->purgeables = &dummyCurrentPage;
		}
		pcache1LeaveMutex(group);
		if (!cache->hashs){
			pcache1Destroy((pcache_t *)cache);
			cache = nullptr;
		}
	}
	return (pcache_t *)cache;
}

/* Implementation of the sqlite3_pcache.xCachesize method. 
**
** Configure the cache_size limit for a cache.
*/
static __host_device__ void pcache1Cachesize(pcache_t *p, int max)
{
	PCache1 *cache = (PCache1 *)p;
	if (cache->purgeable) {
		PGroup *group = cache->group;
		pcache1EnterMutex(group);
		group->maxPages += (max - cache->max);
		group->maxPinned = group->maxPages + 10 - group->minPages;
		cache->max = max;
		cache->n90pct = cache->max * 9 / 10;
		pcache1EnforceMaxPage(cache);
		pcache1LeaveMutex(group);
	}
}

/* Implementation of the sqlite3_pcache.xShrink method. 
**
** Free up as much memory as possible.
*/
static __host_device__ void pcache1Shrink(pcache_t *p)
{
	PCache1 *cache = (PCache1 *)p;
	if (cache->purgeable) {
		PGroup *group = cache->group;
		pcache1EnterMutex(group);
		int savedMaxPages = group->maxPages;
		group->maxPages = 0;
		pcache1EnforceMaxPage(cache);
		group->maxPages = savedMaxPages;
		pcache1LeaveMutex(group);
	}
}

/* Implementation of the sqlite3_pcache.xPagecount method. */
static __host_device__ int pcache1Pagecount(pcache_t *p)
{
	PCache1 *cache = (PCache1 *)p;
	pcache1EnterMutex(cache->group);
	int pages = cache->pages;
	pcache1LeaveMutex(cache->group);
	return pages;
}

/*
** Implement steps 3, 4, and 5 of the pcache1Fetch() algorithm described
** in the header of the pcache1Fetch() procedure.
**
** This steps are broken out into a separate procedure because they are
** usually not needed, and by avoiding the stack initialization required
** for these steps, the main pcache1Fetch() procedure can run faster.
*/
static __host_device__ PgHdr1 *pcache1FetchStage2(PCache1 *cache, uint key, int createFlag)
{
	// Step 3: Abort if createFlag is 1 but the cache is nearly full
	assert(cache->pages >= cache->recyclables);
	uint pinned = cache->pages - cache->recyclables;
	PGroup *group = cache->group;
	assert(group->maxPinned == group->maxPages + 10 - group->minPages);
	assert(cache->n90pct == cache->max * 9 / 10);
	if (createFlag && (pinned >= group->maxPinned || pinned >= cache->n90pct || (pcache1UnderMemoryPressure(cache) && cache->recyclables < pinned)))
		return nullptr;

	if (cache->pages >= cache->hashs) pcache1ResizeHash(cache);
	assert(cache->hashs > 0 && cache->hash);

	// Step 4. Try to recycle a page.
	PgHdr1 *page = nullptr;
	if (cache->purgeable && !group->lru.lruPrev->isAnchor && ((cache->pages + 1 >= cache->max) || pcache1UnderMemoryPressure(cache))) {
		page = group->lru.lruPrev;
		assert(PAGE_IS_UNPINNED(page));
		pcache1RemoveFromHash(page, false);
		pcache1PinPage(page);
		PCache1 *other = page->cache;
		if (other->sizeAlloc != cache->sizeAlloc) {
			pcache1FreePage(page);
			page = nullptr;
		}
		else group->purgeables -= (other->purgeable - cache->purgeable);
	}

	// Step 5. If a usable page buffer has still not been found, attempt to allocate a new one. 
	if (!page)
		page = pcache1AllocPage(cache, createFlag);

	if (page) {
		uint h = key % cache->hashs;
		cache->pages++;
		page->key = key;
		page->next = cache->hash[h];
		page->cache = cache;
		page->lruPrev = nullptr;
		page->lruNext = nullptr;
		*(void **)page->page.extra = nullptr;
		cache->hash[h] = page;
		if (key > cache->maxKey)
			cache->maxKey = key;
	}
	return page;
}

/* Implementation of the sqlite3_pcache.xFetch method. 
**
** Fetch a page by key value.
**
** Whether or not a new page may be allocated by this function depends on
** the value of the createFlag argument.  0 means do not allocate a new
** page.  1 means allocate a new page if space is easily available.  2 
** means to try really hard to allocate a new page.
**
** For a non-purgeable cache (a cache used as the storage for an in-memory
** database) there is really no difference between createFlag 1 and 2.  So
** the calling function (pcache.c) will never have a createFlag of 1 on
** a non-purgeable cache.
*/
static __host_device__ PgHdr1 *pcache1FetchNoMutex(pcache_t *p, uint key, int createFlag)
{
	PCache1 *cache = (PCache1 *)p;
	// Step 1: Search the hash table for an existing entry.
	PgHdr1 *page = cache->hash[key % cache->hashs];
	while (page && page->key != key) page = page->next;

	// Step 2: If the page was found in the hash table, then return it. If the page was not in the hash table and createFlag is 0, abort.
	// Otherwise (page not in hash and createFlag!=0) continue with subsequent steps to try to create the page. */
	if (page) return PAGE_IS_UNPINNED(page) ? pcache1PinPage(page) : page;
	/* Steps 3, 4, and 5 implemented by this subroutine */
	else if (createFlag) return pcache1FetchStage2(cache, key, createFlag);
	else return nullptr;
}

#if PCACHE1_MIGHT_USE_GROUP_MUTEX
static __host_device__ PgHdr1 *pcache1FetchWithMutex(pcache_t *p, uint key, int createFlag)
{
	PCache1 *cache = (PCache1 *)p;
	pcache1EnterMutex(cache->group);
	PgHdr1 *page = pcache1FetchNoMutex(p, key, createFlag);
	assert(!page || cache->maxKey >= key);
	pcache1LeaveMutex(cache->group);
	return page;
}
#endif

static __host_device__ pcache_page_t *pcache1Fetch(pcache_t *p, uint key, int createFlag)
{
#if PCACHE1_MIGHT_USE_GROUP_MUTEX || defined(_DEBUG)
	PCache1 *cache = (PCache1 *)p;
#endif
	assert(!offsetof(PgHdr1, page));
	assert(cache->purgeable || createFlag!=1 );
	assert(cache->purgeable || cache->min == 0);
	assert(!cache->purgeable || cache->min == 10);
	assert(cache->min == 0 || cache->purgeable);
	assert(cache->hashs > 0);
#if PCACHE1_MIGHT_USE_GROUP_MUTEX
	if (cache->group->mutex) return (pcache_page_t *)pcache1FetchWithMutex(p, key, createFlag);
	else
#endif
		return (pcache_page_t *)pcache1FetchNoMutex(p, key, createFlag);
}

/* Implementation of the sqlite3_pcache.xUnpin method.
**
** Mark a page as unpinned (eligible for asynchronous recycling).
*/
static __host_device__ void pcache1Unpin(pcache_t *p, pcache_page_t *pg, int reuseUnlikely)
{
	PCache1 *cache = (PCache1 *)p;
	PgHdr1 *page = (PgHdr1 *)pg;
	PGroup *group = cache->group;
	assert(page->cache == cache);
	pcache1EnterMutex(group);
	// It is an error to call this function if the page is already part of the PGroup LRU list.
	assert(!page->lruPrev && !page->lruNext);
	assert(PAGE_IS_PINNED(page));
	if (reuseUnlikely || group->purgeables > group->maxPages)
		pcache1RemoveFromHash(page, 1);
	else {
		// Add the page to the PGroup LRU list.
		PgHdr1 **first = &group->lru.lruNext;
		page->lruPrev = &group->lru;
		(page->lruNext = *first)->lruPrev = page;
		*first = page;
		cache->recyclables++;
	}
	pcache1LeaveMutex(cache->group);
}

/* Implementation of the sqlite3_pcache.xRekey method. */
static __host_device__ void pcache1Rekey(pcache_t *p, pcache_page_t *pg, uint old, uint new_)
{
	PCache1 *cache = (PCache1 *)p;
	PgHdr1 *page = (PgHdr1 *)pg;
	assert(page->key == old);
	assert(page->cache == cache);
	pcache1EnterMutex(cache->group);

	uint h = old % cache->hashs;
	PgHdr1 **pp = &cache->hash[h];
	while ((*pp) != page)
		pp = &(*pp)->next;
	*pp = page->next;

	h = new_ % cache->hashs;
	page->key = new_;
	page->next = cache->hash[h];
	cache->hash[h] = page;
	if (new_ > cache->maxKey)
		cache->maxKey = new_;
	pcache1LeaveMutex(cache->group);
}

/* Implementation of the sqlite3_pcache.xTruncate method. 
**
** Discard all unpinned pages in the cache with a page number equal to or greater than parameter iLimit. Any pinned pages with a page number
** equal to or greater than iLimit are implicitly unpinned.
*/
static __host_device__ void pcache1Truncate(pcache_t *p, uint limit)
{
	PCache1 *cache = (PCache1 *)p;
	pcache1EnterMutex(cache->group);
	if (limit <= cache->maxKey) {
		pcache1TruncateUnsafe(cache, limit);
		cache->maxKey = limit - 1;
	}
	pcache1LeaveMutex(cache->group);
}

/* Implementation of the sqlite3_pcache.xDestroy method. 
**
** Destroy a cache allocated using pcache1Create().
*/
static __host_device__ void pcache1Destroy(pcache_t *p)
{
	PCache1 *cache = (PCache1 *)p;
	PGroup *group = cache->group;
	assert(cache->purgeable || (!cache->max && !cache->min));
	pcache1EnterMutex(group);
	if (cache->pages) pcache1TruncateUnsafe(cache, 0);
	assert(group->maxPages >= cache->max);
	group->maxPages -= cache->max;
	assert(group->minPages >= cache->min);
	group->minPages -= cache->min;
	group->maxPinned = group->maxPages + 10 - group->minPages;
	pcache1EnforceMaxPage(cache);
	pcache1LeaveMutex(group);
	mfree(cache->bulk);
	mfree(cache->hash);
	mfree(cache);
}

/* This function is called during initialization (sqlite3_initialize()) to install the default pluggable cache module, assuming the user has not
** already provided an alternative.
*/
static __constant__ const pcache_methods _pcache1DefaultMethods = {
	1,
	0,
	pcache1Init,
	pcache1Shutdown,
	pcache1Create,
	pcache1Cachesize,
	pcache1Pagecount,
	pcache1Fetch,
	pcache1Unpin,
	pcache1Rekey,
	pcache1Truncate,
	pcache1Destroy,
	pcache1Shrink
};
__host_device__ void __pcachesystemSetDefault() //: sqlite3PCacheSetDefault
{
	//sqlite3_config(CONFIG_PCACHE2, &defaultMethods);
	__pcachesystem = _pcache1DefaultMethods;
}

/* Return the size of the header on each page of this PCACHE implementation. */
__host_device__ int pcache1HeaderSize() { return _ROUND8(sizeof(PgHdr1)); } //: sqlite3HeaderSizePcache1

/* Return the global mutex used by this PCACHE implementation.  The sqlite3_status() routine needs access to this mutex. */
__host_device__ mutex *pcacheMutex() { return _pcache1.mutex; } //: sqlite3Pcache1Mutex

#ifdef ENABLE_MEMORY_MANAGEMENT
/* This function is called to free superfluous dynamically allocated memory held by the pager system. Memory in use by any SQLite pager allocated
** by the current thread may be sqlite3_free()ed.
**
** nReq is the number of bytes of memory required. Once this much has been released, the function returns. The return value is the total number 
** of bytes of memory released.
*/
__host_device__ int pcacheReleaseMemory(int required) //: sqlite3PcacheReleaseMemory
{
	assert(mutex_notheld(_pcache1.group.mutex));
	assert(mutex_notheld(_pcache1.mutex));
	int free = 0;
	if (!_runtimeConfig.page) {
		pcache1EnterMutex(&_pcache1.group);
		PgHdr1 *p; while ((required < 0 || free < required) && (p = _pcache1.group.lru.lruPrev) && !p->isAnchor) {
			free += pcache1MemSize(p->page.buf);
#ifdef PCACHE_SEPARATE_HEADER
			free += sqlite3MemSize(p);
#endif
			assert(PAGE_IS_UNPINNED(p));
			pcache1PinPage(p);
			pcache1RemoveFromHash(p, true);
		}
		pcache1LeaveMutex(&_pcache1.group);
	}
	return free;
}
#endif /* ENABLE_MEMORY_MANAGEMENT */

#pragma endregion

#pragma	region Tests
#ifndef _TEST
/* This function is used by test procedures to inspect the internal state of the global cache. */
__host_device__ void pcacheStats(int *current, int *max, int *min, int *recyclables) // : sqlite3PcacheStats
{
	int recyclables2 = 0;
	for (PgHdr1 *p = _pcache1.group.lru.lruNext; p && !p->isAnchor; p = p->lruNext) { assert(PAGE_IS_UNPINNED(p)); recyclables2++; }
	*current = _pcache1.group.purgeables;
	*max = (int)_pcache1.group.maxPages;
	*min = (int)_pcache1.group.minPages;
	*recyclables = recyclables2;
}
#endif
#pragma endregion
