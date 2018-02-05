/*
pcache.h - xxx
The MIT License

Copyright (c) 2016 Sky Morey

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <ext\global.h>
#ifndef _EXT_PCACHE_H
#define _EXT_PCACHE_H
__BEGIN_DECLS;

#pragma region Base

// CAPI3REF: Custom Page Cache Object
typedef struct pcache_t pcache_t;

// CAPI3REF: Custom Page Cache Object
typedef struct pcache_page_t pcache_page_t;
struct pcache_page_t {
	void *buf;        // The content of the page
	void *extra;      // Extra information associated with the page
};

// CAPI3REF: Application Defined Page Cache.
typedef struct pcache_methods pcache_methods;
struct pcache_methods {
	int version;
	void *arg;
	int (*init)(void *);
	void (*shutdown)(void *);
	pcache_t *(*create)(int sizePage, int sizeExtra, int purgeable);
	void (*cachesize)(pcache_t *, int cachesize);
	int (*pagecount)(pcache_t *);
	pcache_page_t *(*fetch)(pcache_t *, uint key, int createFlag);
	void (*unpin)(pcache_t *, pcache_page_t *, int discard);
	void (*rekey)(pcache_t *, pcache_page_t *, uint oldKey, uint newKey);
	void (*truncate)(pcache_t *, uint limit);
	void (*destroy)(pcache_t *);
	void (*shrink)(pcache_t *);
};
#define __pcachesystem _runtimeConfig.pcache2System

#pragma endregion

typedef struct PgHdr PgHdr;
typedef struct PCache PCache;

typedef struct Pager Pager; //: Sky
typedef uint Pgno; //: Sky

/* Every page in the cache is controlled by an instance of the following structure. */
struct PgHdr {
	pcache_page_t *page;			// Pcache object page handle
	void *data;						// Page data
	void *extra;					// Extra content
	PCache *cache;					// PRIVATE: Cache that owns this page
	PgHdr *dirty;					// Transient list of dirty sorted by pgno
	Pager *pager;					// The pager this page is part of
	Pgno pgno;						// Page number for this page
#ifdef LIBCU_CHECK_PAGES
	uint32_t pageHash;              // Hash of page content
#endif
	uint16_t flags;                 // PGHDR flags defined below
	// Elements above, except pCache, are public.  All that follow are private to pcache.c and should not be accessed by other modules.
	// pCache is grouped with the public elements for efficiency.
	int16_t refs;                   // Number of users of this page
	PgHdr *dirtyNext;				// Next element in list of dirty pages
	PgHdr *dirtyPrev;				// Previous element in list of dirty pages
	// NB: dirtyNext and dirtyPrev are undefined if the PgHdr object is not dirty
};

/* Bit values for PgHdr.flags */
#define PGHDR_CLEAN           0x001  // Page not on the PCache.pDirty list
#define PGHDR_DIRTY           0x002  // Page is on the PCache.pDirty list
#define PGHDR_WRITEABLE       0x004  // Journaled and ready to modify
#define PGHDR_NEED_SYNC       0x008  // Fsync the rollback journal before writing this page to the database
#define PGHDR_DONT_WRITE      0x010  // Do not write content to disk
#define PGHDR_MMAP            0x020  // This is an mmap page object
#define PGHDR_WAL_APPEND      0x040  // Appended to wal file

/* Initialize and shutdown the page cache subsystem */
__device__ RC pcacheInitialize(); //: sqlite3PcacheInitialize
__device__ void pcacheShutdown(); //: sqlite3PcacheShutdown

/* Page cache buffer management:
** These routines implement SQLITE_CONFIG_PAGECACHE.
*/
__device__ void pcacheBufferSetup(void *, int size, int n); // sqlite3PCacheBufferSetup

/* Create a new pager cache.
** Under memory stress, invoke xStress to try to make pages clean.
** Only clean and unpinned pages can be reclaimed.
*/
__device__ int pcacheOpen(int sizePage, int sizeExtra, int purgeable, int (*stress)(void*,PgHdr*), void *stressArg, PCache *toInit); //: sqlite3PcacheOpen

/* Modify the page-size after the cache has been created. */
__device__ RC pcacheSetPageSize(PCache *, int); //: sqlite3PcacheSetPageSize

/* Return the size in bytes of a PCache object.  Used to preallocate storage space. */
__device__ int pcacheSize(); //: sqlite3PcacheSize

/* One release per successful fetch.  Page is pinned until released. Reference counted.  */
__device__ pcache_page_t *pcacheFetch(PCache *, Pgno, int createFlag); //: sqlite3PcacheFetch
__device__ int pcacheFetchStress(PCache *, Pgno, pcache_page_t **); //: sqlite3PcacheFetchStress
__device__ PgHdr *pcacheFetchFinish(PCache *, Pgno, pcache_page_t *page); //: sqlite3PcacheFetchFinish
__device__ void pcacheRelease(PgHdr *); //: sqlite3PcacheRelease

/* Remove page from cache */
__device__ void pcacheDrop(PgHdr *); //: sqlite3PcacheDrop
/* Make sure page is marked dirty */
__device__ void pcacheMakeDirty(PgHdr *); //: sqlite3PcacheMakeDirty
/* Mark a single page as clean */
__device__ void pcacheMakeClean(PgHdr *); //: sqlite3PcacheMakeClean
/* Mark all dirty list pages as clean */
__device__ void pcacheCleanAll(PCache *); //: sqlite3PcacheCleanAll  
__device__ void pcacheClearWritable(PCache *); //: sqlite3PcacheClearWritable

/* Change a page number.  Used by incr-vacuum. */
__device__ void pcacheMove(PgHdr *, Pgno); //: sqlite3PcacheMove

/* Remove all pages with pgno>x.  Reset the cache if x==0 */
__device__ void pcacheTruncate(PCache *, Pgno x); //: sqlite3PcacheTruncate

/* Get a list of all dirty pages in the cache, sorted by page number */
__device__ PgHdr *pcacheDirtyList(PCache *); //: sqlite3PcacheDirtyList

/* Reset and close the cache object */
__device__ void pcacheClose(PCache *); //: sqlite3PcacheClose

/* Clear flags from pages of the page cache */
__device__ void pcacheClearSyncFlags(PCache *); //: sqlite3PcacheClearSyncFlags

/* Discard the contents of the cache */
__device__ void pcacheClear(PCache *); //: sqlite3PcacheClear

/* Return the total number of outstanding page references */
__device__ int pcacheRefCount(PCache *); //: sqlite3PcacheRefCount

/* Increment the reference count of an existing page */
__device__ void pcacheRef(PgHdr *); //: sqlite3PcacheRef

__device__ int pcachePageRefcount(PgHdr *); //: sqlite3PcachePageRefcount

/* Return the total number of pages stored in the cache */
__device__ int pcachePagecount(PCache *); //: sqlite3PcachePagecount

#if defined(LIBCU_CHECK_PAGES) || defined(_DEBUG)
/* Iterate through all dirty pages currently stored in the cache. This interface is only available if SQLITE_CHECK_PAGES is defined when the library is built. */
__device__ void pcacheIterateDirty(PCache *cache, void (*iter)(PgHdr*)); //: sqlite3PcacheIterateDirty
#endif

#if defined(_DEBUG)
/* Check invariants on a PgHdr object */
__device__ int pcachePageSanity(PgHdr *); //: sqlite3PcachePageSanity
#endif

/* Set and get the suggested cache-size for the specified pager-cache.
**
** If no global maximum is configured, then the system attempts to limit the total number of pages cached by purgeable pager-caches to the sum
** of the suggested cache-sizes.
*/
__device__ void pcacheSetCachesize(PCache *, int); //: sqlite3PcacheSetCachesize
#ifdef _TEST
__device__ int pcacheGetCachesize(PCache *); //: sqlite3PcacheGetCachesize
#endif

/* Set or get the suggested spill-size for the specified pager-cache.
**
** The spill-size is the minimum number of pages in cache before the cache will attempt to spill dirty pages by calling xStress.
*/
__device__ int pcacheSetSpillsize(PCache *, int); //: sqlite3PcacheSetSpillsize

/* Free up as much memory as possible from the page cache */
__device__ void pcacheShrink(PCache *); //: sqlite3PcacheShrink

#ifdef ENABLE_MEMORY_MANAGEMENT
/* Try to return memory used by the pcache module to the main memory heap */
__device__ int pcacheReleaseMemory(int); //: sqlite3PcacheReleaseMemory
#endif

#ifdef _TEST
__device__ void pcacheStats(int *, int *, int *, int *); //: sqlite3PcacheStats
#endif

__device__ void __pcachesystemSetDefault();  //: sqlite3PCacheSetDefault

/* Return the header size */
__device__ int pcacheHeaderSize(); //: sqlite3HeaderSizePcache
__device__ int pcache1HeaderSize(); //: sqlite3HeaderSizePcache1

/* Number of dirty pages as a percentage of the configured cache size */
__device__ int pcachePercentDirty(PCache *); //: sqlite3PCachePercentDirty

__END_DECLS;
#endif	/* _EXT_PCACHE_H */