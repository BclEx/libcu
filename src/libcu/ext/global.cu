#include <stdlibcu.h>
#include <ext/global.h>

/* EVIDENCE-OF: R-02982-34736 In order to maintain full backwards compatibility for legacy applications, the URI filename capability is
** disabled by default.
**
** EVIDENCE-OF: R-38799-08373 URI filenames can be enabled or disabled using the LIBCU_USEURI=1 or LIBCU_USEURI=0 compile-time options.
**
** EVIDENCE-OF: R-43642-56306 By default, URI handling is globally disabled. The default value may be changed by compiling with the
** LIBCU_USEURI symbol defined.
*/
#ifndef LIBCU_USEURI
#define LIBCU_USEURI 0
#endif

/*
** The default lookaside-configuration, the format "SZ,N".  SZ is the number of bytes in each lookaside slot (should be a multiple of 8)
** and N is the number of slots.  The lookaside-configuration can be changed as start-time using sqlite3_config(SQLITE_CONFIG_LOOKASIDE)
** or at run-time for an individual database connection using sqlite3_db_config(db, SQLITE_DBCONFIG_LOOKASIDE);
*/
#ifndef LIBCU_DEFAULTLOOKASIDE
#define LIBCU_DEFAULTLOOKASIDE 1200, 100
#endif

/*
** The default initial allocation for the pagecache when using separate pagecaches for each database connection.  A positive number is the
** number of pages.  A negative number N translations means that a buffer of -1024*N bytes is allocated and used for as many pages as it will hold.
**
** The default value of "20" was choosen to minimize the run-time of the speedtest1 test program with options: --shrink-memory --reprepare
*/
#ifndef LIBCU_DEFAULTPCACHEINITSZ
#define LIBCU_DEFAULTPCACHEINITSZ 20
#endif

/* The following singleton contains the global configuration for the Libcu library. */
_WSD struct RuntimeStatics _runtimeStatics = {
	//{nullptr, nullptr},			// appendFormat
	LIBCU_DEFAULTMEMSTATUS,		// memstat
	true,						// coreMutex
	LIBCU_THREADSAFE == 1,		// fullMutex
	LIBCU_USEURI,				// openUri
	0x7ffffffe,					// maxStrlen
	0,							// neverCorrupt
	LIBCU_DEFAULTLOOKASIDE,		// lookasideSize, lookasides
	{0,0,0,0,0,0,0,0},			// allocSystem
	{0,0,0,0,0,0,0,0,0},		// mutexSystem
	(void *)nullptr,            // heap
	0,							// heapSize
	0, 0,						// minHeapSize, maxHeapSize
	LIBCU_DEFAULTMMAPSIZE,		// sizeMmap
	LIBCU_MAXMMAPSIZE,			// maxMmap
	(void *)nullptr,            // scratch
	0,							// scratchSize
	0,							// scratchs
	(void *)nullptr,			// page
	0,							// pageSize
	LIBCU_DEFAULTPCACHEINITSZ,	// pages
	/* All the rest should always be initialized to zero */
	0,							// isInit
	0,							// inProgress
	0,							// isMutexInit
	0,							// isMallocInit
	0,							// initMutexRefs
	0,							// initMutex
	nullptr,					// log
	0,							// logArg
#ifndef LIBCU_UNTESTABLE
	0,							// xTestCallback
#endif
};

//__device__ bool TagBase::SetupLookaside(void *buf, int size, int count)
//{
//	if (Lookaside.Outs)
//		return false;
//	// Free any existing lookaside buffer for this handle before allocating a new one so we don't have to have space for both at the same time.
//	if (Lookaside.Malloced)
//		_free(Lookaside.Start);
//	// The size of a lookaside slot after ROUNDDOWN8 needs to be larger than a pointer to be useful.
//	size = _ROUNDDOWN8(size); // IMP: R-33038-09382
//	if (size <= (int)sizeof(TagBase::LookasideSlot *)) size = 0;
//	if (count < 0) count = 0;
//	void *start;
//	if (size == 0 || count == 0)
//	{
//		size = 0;
//		start = nullptr;
//	}
//	else if (!buf)
//	{
//		_benignalloc_begin();
//		start = _alloc(size * count); // IMP: R-61949-35727
//		_benignalloc_end();
//		if (start) count = (int)_allocsize(start) / size;
//	}
//	else
//		start = buf;
//	Lookaside.Start = start;
//	Lookaside.Free = nullptr;
//	Lookaside.Size = (uint16)size;
//	if (start)
//	{
//		_assert(size > (int)sizeof(TagBase::LookasideSlot *));
//		TagBase::LookasideSlot *p = (TagBase::LookasideSlot *)start;
//		for (int i = count - 1; i >= 0; i--)
//		{
//			p->Next = Lookaside.Free;
//			Lookaside.Free = p;
//			p = (TagBase::LookasideSlot *)&((uint8 *)p)[size];
//		}
//		Lookaside.End = p;
//		Lookaside.Enabled = true;
//		Lookaside.Malloced = (!buf);
//	}
//	else
//	{
//		Lookaside.End = nullptr;
//		Lookaside.Enabled = false;
//		Lookaside.Malloced = false;
//	}
//	return true;
//}
