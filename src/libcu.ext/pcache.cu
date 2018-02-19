#include <ext/global.h> //: pcache.c
#include <ext/pcache.h>
#include <stdiocu.h>
#include <assert.h>

struct PCache {
	PgHdr *dirty, *dirtyTail;       // List of dirty pages in LRU order
	PgHdr *synced;                  // Last synced page in dirty page list
	int refSum;						// Sum of ref counts over all pages
	int sizeCache;                  // Configured cache size
	int sizeSpill;                  // Size before spilling occurs
	int sizePage;                   // Size of every page in this cache
	int sizeExtra;                  // Size of extra space for each page
	int purgeable;					// True if pages are on backing store
	uint8_t create;                 // eCreate value for for xFetch()
	int (*stress)(void*,PgHdr*);    // Call to try make a page clean
	void *stressArg;                // Argument to xStress
	pcache_t *cache;				// Pluggable cache module
};

#pragma region Test and Debug Logic

/* Debug tracing macros.  Enable by by changing the "0" to "1" and recompiling.
**
** When sqlite3PcacheTrace is 1, single line trace messages are issued.
** When sqlite3PcacheTrace is 2, a dump of the pcache showing all cache entries is displayed for many operations, resulting in a lot of output.
*/
#if defined(_DEBUG) && 0
__host_device__ int _pcacheTrace = 2;       // 0: off  1: simple  2: cache dumps
__host_device__ int _pcacheMaxDump = 9999;   // Max cache entries for pcacheDump()
#define pcacheTrace(x) if (_pcacheTrace) { _debug x; }
__host_device__ void pcacheDump(PCache *cache)
{
	if (_pcacheTrace < 2) return;
	if (!cache->cache) return;
	int n = pcachePagecount(cache);
	if (n > _pcacheMaxDump) n = _pcacheMaxDump;
	for (int i = 1; i <= n; i++) {
		pcache_page_t *lower = __pcachesystem.fetch(cache->cache, i, 0);
		if (!lower) continue;
		PgHdr *pg = (PgHdr *)lower->extra;
		printf("%3d: nRef %2d flgs %02x data ", i, pg->refs, pg->flags);
		unsigned char *a = (unsigned char *)lower->buf;
		for (int j = 0; j < 12; j++) printf("%02x", a[j]);
		printf("\n");
		if (!pg->page)
			__pcachesystem.unpin(cache->cache, lower, 0);
	}
}
#else
#define pcacheTrace(x)
#define pcacheDump(x)
#endif

/* Check invariants on a PgHdr entry.  Return true if everything is OK. Return false if any invariant is violated.
**
** This routine is for use inside of assert() statements only.  For example:
**		assert(sqlite3PcachePageSanity(pg));
*/
#ifdef _DEBUG
__device__ int pcachePageSanity(PgHdr *pg) //: sqlite3PcachePageSanity
{
	assert(pg);
	assert(pg->pgno > 0 || !pg->pager); // Page number is 1 or more
	PCache *cache = pg->cache;
	assert(cache); // Every page has an associated PCache
	if (pg->flags & PGHDR_CLEAN) {
		assert((pg->flags & PGHDR_DIRTY) == 0); // Cannot be both CLEAN and DIRTY
		assert(cache->dirty != pg);				// CLEAN pages not on dirty list
		assert(cache->dirtyTail != pg);
	}
	// WRITEABLE pages must also be DIRTY
	if (pg->flags & PGHDR_WRITEABLE)
		assert(pg->flags & PGHDR_DIRTY);		// WRITEABLE implies DIRTY
	// NEED_SYNC can be set independently of WRITEABLE.  This can happen, for example, when using the sqlite3PagerDontWrite() optimization:
	//    (1)  Page X is journalled, and gets WRITEABLE and NEED_SEEK.
	//    (2)  Page X moved to freelist, WRITEABLE is cleared
	//    (3)  Page X reused, WRITEABLE is set again
	// If NEED_SYNC had been cleared in step 2, then it would not be reset in step 3, and page might be written into the database without first
	// syncing the rollback journal, which might cause corruption on a power loss.
	//
	// Another example is when the database page size is smaller than the disk sector size.  When any page of a sector is journalled, all pages
	// in that sector are marked NEED_SYNC even if they are still CLEAN, just in case they are later modified, since all pages in the same sector
	// must be journalled and synced before any of those pages can be safely written.
	return 1;
}
#endif
#pragma endregion

#pragma region Linked List Management

/* Allowed values for second argument to pcacheManageDirtyList() */
#define PCACHE_DIRTYLIST_REMOVE   1    // Remove pPage from dirty list
#define PCACHE_DIRTYLIST_ADD      2    // Add pPage to the dirty list
#define PCACHE_DIRTYLIST_FRONT    3    // Move pPage to the front of the list

/* Manage pPage's participation on the dirty list.  Bits of the addRemove argument determines what operation to do.  The 0x01 bit means first
** remove pPage from the dirty list.  The 0x02 means add pPage back to the dirty list.  Doing both moves pPage to the front of the dirty list.
*/
static __device__ void pcacheManageDirtyList(PgHdr *page, uint8_t addRemove)
{
	PCache *p = page->cache;
	pcacheTrace(("%p.DIRTYLIST.%s %d\n", p, addRemove == 1 ? "REMOVE" : addRemove == 2 ? "ADD" : "FRONT", page->pgno));
	if (addRemove & PCACHE_DIRTYLIST_REMOVE) {
		assert(page->dirtyNext || page == p->dirtyTail);
		assert(page->dirtyPrev || page == p->dirty);

		// Update the PCache1.pSynced variable if necessary.
		if (p->synced == page)
			p->synced = page->dirtyPrev;

		if (page->dirtyNext)
			page->dirtyNext->dirtyPrev = page->dirtyPrev;
		else {
			assert(page == p->dirtyTail);
			p->dirtyTail = page->dirtyPrev;
		}
		if (page->dirtyPrev)
			page->dirtyPrev->dirtyNext = page->dirtyNext;
		else {
			// If there are now no dirty pages in the cache, set eCreate to 2. This is an optimization that allows sqlite3PcacheFetch() to skip
			// searching for a dirty page to eject from the cache when it might otherwise have to.
			assert(page == p->dirty);
			p->dirty = page->dirtyNext;
			assert(p->purgeable || p->create == 2);
			if (!p->dirty) { /*OPTIMIZATION-IF-TRUE*/
				assert(!p->purgeable || p->create == 1);
				p->create = 2;
			}
		}
	}
	if (addRemove & PCACHE_DIRTYLIST_ADD) {
		page->dirtyPrev = nullptr;
		page->dirtyNext = p->dirty;
		if (page->dirtyNext) {
			assert(!page->dirtyNext->dirtyPrev);
			page->dirtyNext->dirtyPrev = page;
		}
		else {
			p->dirtyTail = page;
			if (p->purgeable) {
				assert(p->create == 2);
				p->create = 1;
			}
		}
		p->dirty = page;

		// If pSynced is NULL and this page has a clear NEED_SYNC flag, set pSynced to point to it. Checking the NEED_SYNC flag is an 
		// optimization, as if pSynced points to a page with the NEED_SYNC flag set sqlite3PcacheFetchStress() searches through all newer 
		// entries of the dirty-list for a page with NEED_SYNC clear anyway.
		if (!p->synced && !(page->flags & PGHDR_NEED_SYNC))   /*OPTIMIZATION-IF-FALSE*/
			p->synced = page;
	}
	pcacheDump(p);
}

/* Wrapper around the pluggable caches xUnpin method. If the cache is being used for an in-memory database, this function is a no-op. */
static __device__ void pcacheUnpin(PgHdr *p)
{
	if (p->cache->purgeable) {
		pcacheTrace(("%p.UNPIN %d\n", p->cache, p->pgno));
		__pcachesystem.unpin(p->cache->cache, p->page, 0);
		pcacheDump(p->pCache);
	}
}

/* Compute the number of pages of cache requested.   p->szCache is the cache size requested by the "PRAGMA cache_size" statement. */
static __device__ int numberOfCachePages(PCache *p){
	// IMPLEMENTATION-OF: R-42059-47211 If the argument N is positive then the suggested cache size is set to N. 
	if (p->sizeCache >= 0) return p->sizeCache;
	// IMPLEMENTATION-OF: R-61436-13639 If the argument N is negative, then the number of cache pages is adjusted to use approximately abs(N*1024) bytes of memory.
	return (int)((-1024*(int64_t)p->sizeCache)/(p->sizePage+p->sizeExtra));
}

#pragma endregion

#pragma region General Interfaces

/* Initialize and shutdown the page cache subsystem. Neither of these functions are threadsafe. */
__device__ RC pcacheInitialize() //: sqlite3PcacheInitialize
{
	// IMPLEMENTATION-OF: R-26801-64137 If the xInit() method is NULL, then the built-in default page cache is used instead of the application defined page cache.
	if (!__pcachesystem.init)
		__pcachesystemSetDefault();
	return __pcachesystem.init(__pcachesystem.arg);
}
__device__ void pcacheShutdown() //: sqlite3PcacheShutdown
{
	// IMPLEMENTATION-OF: R-26000-56589 The xShutdown() method may be NULL.
	if (__pcachesystem.shutdown)
		__pcachesystem.shutdown(__pcachesystem.arg);
}

/* Return the size in bytes of a PCache object. */
__device__ int pcacheSize() { return sizeof(PCache); } //: sqlite3PcacheSize

/* Create a new PCache object. Storage space to hold the object has already been allocated and is passed in as the p pointer. 
** The caller discovers how much space needs to be allocated by calling sqlite3PcacheSize().
**
** szExtra is some extra space allocated for each page.  The first 8 bytes of the extra space will be zeroed as the page is allocated,
** but remaining content will be uninitialized.  Though it is opaque to this module, the extra space really ends up being the MemPage
** structure in the pager.
*/
__device__ int sqlite3PcacheOpen(int sizePage, int sizeExtra, int purgeable, int (*stress)(void*,PgHdr*), void *stressArg, PCache *p) //: sqlite3PcacheOpen
{
	memset(p, 0, sizeof(PCache));
	p->sizePage = 1;
	p->sizeExtra = sizeExtra;
	assert(sizeExtra >= 8); // First 8 bytes will be zeroed
	p->purgeable = purgeable;
	p->create = 2;
	p->stress = stress;
	p->stressArg = stressArg;
	p->sizeCache = 100;
	p->sizeSpill = 1;
	pcacheTrace(("%p.OPEN szPage %d bPurgeable %d\n", p, sizePage, purgeable));
	return pcacheSetPageSize(p, sizePage);
}

/* Change the page size for PCache object. The caller must ensure that there are no outstanding page references when this function is called. */
__device__ RC pcacheSetPageSize(PCache *cache, int sizePage) //: sqlite3PcacheSetPageSize
{
	assert(!cache->refSum && !cache->dirty);
	if (cache->sizePage) {
		pcache_t *newCache;
		newCache = __pcachesystem.create(sizePage, cache->sizeExtra + ROUND8_(sizeof(PgHdr)), cache->purgeable);
		if (!newCache) return RC_NOMEM_BKPT;
		__pcachesystem.cachesize(newCache, numberOfCachePages(cache));
		if (cache->cache)
			__pcachesystem.destroy(cache->cache);
		cache->cache = newCache;
		cache->sizePage = sizePage;
		pcacheTrace(("%p.PAGESIZE %d\n", cache, sizePage));
	}
	return RC_OK;
}

/* Try to obtain a page from the cache.
**
** This routine returns a pointer to an sqlite3_pcache_page object if such an object is already in cache, or if a new one is created.
** This routine returns a NULL pointer if the object was not in cache and could not be created.
**
** The createFlags should be 0 to check for existing pages and should be 3 (not 1, but 3) to try to create a new page.
**
** If the createFlag is 0, then NULL is always returned if the page is not already in the cache.  If createFlag is 1, then a new page
** is created only if that can be done without spilling dirty pages and without exceeding the cache size limit.
**
** The caller needs to invoke sqlite3PcacheFetchFinish() to properly initialize the sqlite3_pcache_page object and convert it into a
** PgHdr object.  The sqlite3PcacheFetch() and sqlite3PcacheFetchFinish() routines are split this way for performance reasons. When separated
** they can both (usually) operate without having to push values to the stack on entry and pop them back off on exit, which saves a
** lot of pushing and popping.
*/
__device__ pcache_page_t *pcacheFetch(PCache *cache, Pgno pgno, int createFlag) //: sqlite3PcacheFetch
{
	assert(cache);
	assert(cache->cache);
	assert(createFlag == 3 || !createFlag);
	assert(cache->create == cache->purgeable && cache->dirty ? 1 : 2);

	// eCreate defines what to do if the page does not exist.
	//    0     Do not allocate a new page.  (createFlag==0)
	//    1     Allocate a new page if doing so is inexpensive. (createFlag==1 AND bPurgeable AND pDirty)
	//    2     Allocate a new page even it doing so is difficult. (createFlag==1 AND !(bPurgeable AND pDirty)
	int create = createFlag & cache->create;
	assert(create == 0 || create == 1 || create == 2 );
	assert(!createFlag || cache->create == create);
	assert(!createFlag || create == 1 + (!cache->purgeable || !cache->dirty));
	pcache_page_t *r = __pcachesystem.fetch(cache->cache, pgno, create);
	pcacheTrace(("%p.FETCH %d%s (result: %p)\n", cache, pgno, createFlag ? " create" : "", r));
	return r;
}

/* If the sqlite3PcacheFetch() routine is unable to allocate a new page because no clean pages are available for reuse and the cache
** size limit has been reached, then this routine can be invoked to try harder to allocate a page.  This routine might invoke the stress
** callback to spill dirty pages to the journal.  It will then try to allocate the new page and will only fail to allocate a new page on
** an OOM error.
**
** This routine should be invoked only after sqlite3PcacheFetch() fails.
*/
__device__ int sqlite3PcacheFetchStress(PCache *cache, Pgno pgno, pcache_page_t **ppPage) //: sqlite3PcacheFetchStress
{
	if (cache->create == 2) return 0;
	if (pcachePagecount(cache) > cache->sizeSpill) {
		// Find a dirty page to write-out and recycle. First try to find a page that does not require a journal-sync (one with PGHDR_NEED_SYNC
		// cleared), but if that is not possible settle for any other unreferenced dirty page.
		//
		// If the LRU page in the dirty list that has a clear PGHDR_NEED_SYNC flag is currently referenced, then the following may leave pSynced
		// set incorrectly (pointing to other than the LRU page with NEED_SYNC cleared). This is Ok, as pSynced is just an optimization.
		PgHdr *pg; for (pg = cache->synced; pg && (pg->refs || (pg->flags & PGHDR_NEED_SYNC)); pg = pg->dirtyPrev);
		cache->synced = pg;
		if (!pg)
			for (pg = cache->dirtyTail; pg && pg->refs; pg = pg->dirtyPrev);
		if (pg) {
#ifdef LOG_CACHE_SPILL
			_log(RC_FULL, "spill page %d making room for %d - cache used: %d/%d", pg->pgno, pgno, __pcachesystem.pagecount(cache->cache), numberOfCachePages(cache));
#endif
			pcacheTrace(("%p.SPILL %d\n", cache, pg->pgno));
			RC rc = cache->stress(cache->stressArg, pg);
			pcacheDump(pCache);
			if (rc != RC_OK && rc != RC_BUSY)
				return rc;
		}
	}
	*ppPage = __pcachesystem.fetch(cache->cache, pgno, 2);
	return !*ppPage ? RC_NOMEM_BKPT : RC_OK;
}

/* This is a helper routine for sqlite3PcacheFetchFinish()
**
** In the uncommon case where the page being fetched has not been initialized, this routine is invoked to do the initialization.
** This routine is broken out into a separate function since it requires extra stack manipulation that can be avoided in the common case.
*/
static __device__ PgHdr *pcacheFetchFinishWithInit(PCache *cache, Pgno pgno, pcache_page_t *page)
{
	assert(page);
	PgHdr *pgHdr = (PgHdr *)page->extra;
	assert(!pgHdr->page);
	memset(&pgHdr->dirty, 0, sizeof(PgHdr) - offsetof(PgHdr, dirty));
	pgHdr->page = page;
	pgHdr->data = page->buf;
	pgHdr->extra = (void *)&pgHdr[1];
	memset(pgHdr->extra, 0, 8);
	pgHdr->cache = cache;
	pgHdr->pgno = pgno;
	pgHdr->flags = PGHDR_CLEAN;
	return pcacheFetchFinish(cache, pgno, page);
}

/* This routine converts the sqlite3_pcache_page object returned by sqlite3PcacheFetch() into an initialized PgHdr object.  This routine
** must be called after sqlite3PcacheFetch() in order to get a usable result.
*/
__device__ PgHdr *pcacheFetchFinish(PCache *cache, Pgno pgno, pcache_page_t *page) //: sqlite3PcacheFetchFinish
{
	assert(page);
	PgHdr *pgHdr = (PgHdr *)page->extra;
	if (!pgHdr->page)
		return pcacheFetchFinishWithInit(cache, pgno, page);
	cache->refSum++;
	pgHdr->refs++;
	assert(pcachePageSanity(pgHdr));
	return pgHdr;
}

/* Decrement the reference count on a page. If the page is clean and the reference count drops to 0, then it is made eligible for recycling. */
__device__ void pcacheRelease(PgHdr *p) //: sqlite3PcacheRelease
{
	assert(p->refs > 0);
	p->cache->refSum--;
	if (!(--p->refs)) {
		if (p->flags & PGHDR_CLEAN) pcacheUnpin(p);
		else pcacheManageDirtyList(p, PCACHE_DIRTYLIST_FRONT);
	}
}

/* Increase the reference count of a supplied page by 1. */
__device__ void pcacheRef(PgHdr *p) //: sqlite3PcacheRef
{
	assert(p->refs > 0);
	assert(pcachePageSanity(p));
	p->refs++;
	p->cache->refSum++;
}

/* Drop a page from the cache. There must be exactly one reference to the page. This function deletes that reference, so after it returns the
** page pointed to by p is invalid.
*/
__device__ void pcacheDrop(PgHdr *p) //: sqlite3PcacheDrop
{
	assert(p->refs == 1);
	assert(pcachePageSanity(p));
	if (p->flags & PGHDR_DIRTY)
		pcacheManageDirtyList(p, PCACHE_DIRTYLIST_REMOVE);
	p->cache->refSum--;
	__pcachesystem.unpin(p->cache->cache, p->page, 1);
}

/* Make sure the page is marked as dirty. If it isn't dirty already, make it so. */
__device__ void sqlite3PcacheMakeDirty(PgHdr *p){
	assert(p->refs > 0);
	assert(pcachePageSanity(p));
	if (p->flags & (PGHDR_CLEAN|PGHDR_DONT_WRITE)) { /*OPTIMIZATION-IF-FALSE*/
		p->flags &= ~PGHDR_DONT_WRITE;
		if (p->flags & PGHDR_CLEAN) {
			p->flags ^= (PGHDR_DIRTY|PGHDR_CLEAN);
			pcacheTrace(("%p.DIRTY %d\n",p->cache, p->pgno));
			assert((p->flags & (PGHDR_DIRTY|PGHDR_CLEAN)) == PGHDR_DIRTY);
			pcacheManageDirtyList(p, PCACHE_DIRTYLIST_ADD);
		}
		assert(pcachePageSanity(p));
	}
}

/* Make sure the page is marked as clean. If it isn't clean already, make it so. */
__device__ void pcacheMakeClean(PgHdr *p) //: sqlite3PcacheMakeClean
{
	assert(pcachePageSanity(p) );
	if (ALWAYS_(p->flags & PGHDR_DIRTY)) {
		assert(!(p->flags & PGHDR_CLEAN));
		pcacheManageDirtyList(p, PCACHE_DIRTYLIST_REMOVE);
		p->flags &= ~(PGHDR_DIRTY|PGHDR_NEED_SYNC|PGHDR_WRITEABLE);
		p->flags |= PGHDR_CLEAN;
		pcacheTrace(("%p.CLEAN %d\n", p->cache, p->pgno));
		assert(pcachePageSanity(p));
		if (!p->refs) pcacheUnpin(p);
	}
}

/* Make every page in the cache clean. */
__device__ void pcacheCleanAll(PCache *pCache) //: sqlite3PcacheCleanAll
{
	pcacheTrace(("%p.CLEAN-ALL\n", cache));
	PgHdr *p; while (p = pCache->dirty) pcacheMakeClean(p);
}

/* Clear the PGHDR_NEED_SYNC and PGHDR_WRITEABLE flag from all dirty pages. */
__device__ void pcacheClearWritable(PCache *cache) //: sqlite3PcacheClearWritable
{
	pcacheTrace(("%p.CLEAR-WRITEABLE\n", cache));
	for (PgHdr *p = cache->dirty; p; p = p->dirtyNext) p->flags &= ~(PGHDR_NEED_SYNC|PGHDR_WRITEABLE);
	cache->synced = cache->dirtyTail;
}

/* Clear the PGHDR_NEED_SYNC flag from all dirty pages. */
__device__ void pcacheClearSyncFlags(PCache *cache) //: sqlite3PcacheClearSyncFlags
{
	for (PgHdr *p = cache->dirty; p; p = p->dirtyNext) p->flags &= ~PGHDR_NEED_SYNC;
	cache->synced = cache->dirtyTail;
}

/* Change the page number of page p to newPgno. */
__device__ void pcacheMove(PgHdr *p, Pgno newPgno) //: sqlite3PcacheMove
{
	PCache *cache = p->cache;
	assert(p->refs > 0);
	assert(newPgno > 0);
	assert(pcachePageSanity(p));
	pcacheTrace(("%p.MOVE %d -> %d\n", cache, p->pgno, newPgno));
	__pcachesystem.rekey(cache->cache, p->page, p->pgno, newPgno);
	p->pgno = newPgno;
	if ((p->flags & PGHDR_DIRTY) && (p->flags & PGHDR_NEED_SYNC))
		pcacheManageDirtyList(p, PCACHE_DIRTYLIST_FRONT);
}

/* Drop every cache entry whose page number is greater than "pgno". The caller must ensure that there are no outstanding references to any pages
** other than page 1 with a page number greater than pgno.
**
** If there is a reference to page 1 and the pgno parameter passed to this function is 0, then the data area associated with page 1 is zeroed, but
** the page object is not dropped.
*/
__device__ void pcacheTruncate(PCache *cache, Pgno pgno) //: sqlite3PcacheTruncate
{
	if (cache->cache) {
		pcacheTrace(("%p.TRUNCATE %d\n", cache, pgno));
		PgHdr *next; for (PgHdr *p = cache->dirty; p; p = next) {
			next = p->dirtyNext;
			// This routine never gets call with a positive pgno except right after sqlite3PcacheCleanAll().  So if there are dirty pages,
			// it must be that pgno==0.
			assert(p->pgno > 0);
			if (p->pgno > pgno) {
				assert(p->flags & PGHDR_DIRTY);
				pcacheMakeClean(p);
			}
		}
		if (pgno == 0 && cache->refSum) {
			pcache_page_t *page1 = __pcachesystem.fetch(cache->cache, 1, 0);
			if (ALWAYS_(page1)) { memset(page1->buf, 0, cache->sizePage); pgno = 1; } // Page 1 is always available in cache, because pCache->nRefSum>0
		}
		__pcachesystem.truncate(cache->cache, pgno + 1);
	}
}

/* Close a cache. */
__device__ void pcacheClose(PCache *cache) //: sqlite3PcacheClose
{
	assert(cache->cache);
	pcacheTrace(("%p.CLOSE\n", cache));
	__pcachesystem.destroy(cache->cache);
}

/* Discard the contents of the cache. */
__device__ void pcacheClear(PCache *cache) //: sqlite3PcacheClear
{
	pcacheTruncate(cache, 0);
}

/* Merge two lists of pages connected by pDirty and in pgno order. Do not bother fixing the pDirtyPrev pointers. */
static __device__ PgHdr *pcacheMergeDirtyList(PgHdr *a, PgHdr *b)
{
	assert(a && b);
	PgHdr result, *tail = &result;
	for(;;) {
		if (a->pgno < b->pgno) {
			tail->dirty = a; tail = a; a = a->dirty;
			if (!a) { tail->dirty = b; break; }
		}
		else {
			tail->dirty = b; tail = b; b = b->dirty;
			if (!b) { tail->dirty = a; break; }
		}
	}
	return result.dirty;
}

/* Sort the list of pages in accending order by pgno.  Pages are connected by pDirty pointers.  The pDirtyPrev pointers are
** corrupted by this sort.
**
** Since there cannot be more than 2^31 distinct pages in a database, there cannot be more than 31 buckets required by the merge sorter.
** One extra bucket is added to catch overflow in case something ever changes to make the previous sentence incorrect.
*/
#define N_SORT_BUCKET 32
static __device__ PgHdr *pcacheSortDirtyList(PgHdr *in)
{
	PgHdr *a[N_SORT_BUCKET], *p;
	memset(a, 0, sizeof(a));
	int i; while (in) {
		p = in; in = p->dirty; p->dirty = 0;
		for (i = 0; ALWAYS_(i < N_SORT_BUCKET - 1); i++) {
			if (!a[i]) { a[i] = p; break; }
			else { p = pcacheMergeDirtyList(a[i], p); a[i] = 0; }
		}
		// To get here, there need to be 2^(N_SORT_BUCKET) elements in the input list.  But that is impossible.
		if (NEVER_(i == N_SORT_BUCKET - 1))
			a[i] = pcacheMergeDirtyList(a[i], p);
	}
	p = a[0];
	for (i = 1; i < N_SORT_BUCKET; i++) {
		if (!a[i]) continue;
		p = p ? pcacheMergeDirtyList(p, a[i]) : a[i];
	}
	return p;
}

/* Return a list of all dirty pages in the cache, sorted by page number. */
__device__ PgHdr *pcacheDirtyList(PCache *cache) //: sqlite3PcacheDirtyList
{
	for (PgHdr *p = cache->dirty; p; p = p->dirtyNext) p->dirty = p->dirtyNext;
	return pcacheSortDirtyList(cache->dirty);
}

/* Return the total number of references to all pages held by the cache.
**
** This is not the total number of pages referenced, but the sum of the reference count for all pages.
*/
__device__ int pcacheRefCount(PCache *cache) { return cache->refSum; } //: sqlite3PcacheRefCount

/* Return the number of references to the page supplied as an argument. */
__device__ int pcachePageRefcount(PgHdr *p) { return p->refs; } //: sqlite3PcachePageRefcount

/* Return the total number of pages in the cache. */
__device__ int pcachePagecount(PCache *cache) { assert(cache->cache); return __pcachesystem.pagecount(cache->cache); } //: sqlite3PcachePagecount

#ifdef _TEST
/* Get the suggested cache-size value. */
__device__ int pcacheGetCachesize(PCache *cache) { return numberOfCachePages(cache); } //: sqlite3PcacheGetCachesize
#endif

/* Set the suggested cache-size value. */
__device__ void pcacheSetCachesize(PCache *cache, int maxPage) //: sqlite3PcacheSetCachesize
{
	assert(cache->cache);
	cache->sizeCache = maxPage;
	__pcachesystem.cachesize(cache->cache, numberOfCachePages(cache));
}

/* Set the suggested cache-spill value.  Make no changes if if the argument is zero.  Return the effective cache-spill size, which will
** be the larger of the szSpill and szCache.
*/
__device__ int sqlite3PcacheSetSpillsize(PCache *p, int maxPage) //: sqlite3PcacheSetSpillsize
{
	assert(p->cache);
	if (maxPage) {
		if (maxPage < 0) maxPage = (int)((-1024 * (int64_t)maxPage)/(p->sizePage+p->sizeExtra));
		p->sizeSpill = maxPage;
	}
	int r = numberOfCachePages(p);
	if (r < p->sizeSpill) r = p->sizeSpill; 
	return r;
}

/* Free up as much memory as possible from the page cache. */
__device__ void pcacheShrink(PCache *cache) { assert(cache->cache); __pcachesystem.shrink(cache->cache); } //: sqlite3PcacheShrink

/* Return the size of the header added by this middleware layer in the page-cache hierarchy. */
__device__ int pcacheHeaderSize() { return ROUND8_(sizeof(PgHdr)); } //: sqlite3HeaderSizePcache

/* Return the number of dirty pages currently in the cache, as a percentage of the configured cache size. */
__device__ int pcachePercentDirty(PCache *cache) //: sqlite3PCachePercentDirty
{
	int cached = numberOfCachePages(cache);
	int dirtys = 0; for (PgHdr *dirty = cache->dirty; dirty; dirty = dirty->dirtyNext) dirtys++;
	return cached ? (int)(((int64_t)dirtys * 100) / cached) : 0;
}

#if defined(LIBCU_CHECK_PAGES) || defined(_DEBUG)
/* For all dirty pages currently in the cache, invoke the specified callback. This is only used if the SQLITE_CHECK_PAGES macro is defined. */
__device__ void pcacheIterateDirty(PCache *cache, void (*iter)(PgHdr*)) //: sqlite3PcacheIterateDirty
{
	for (PgHdr *dirty = cache->dirty; dirty; dirty = dirty->dirtyNext) iter(dirty);
}
#endif

#pragma endregion
