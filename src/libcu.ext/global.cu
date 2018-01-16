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

/* EVIDENCE-OF: R-38720-18127 The default setting is determined by the SQLITE_ALLOW_COVERING_INDEX_SCAN compile-time option, or is "on" if
** that compile-time option is omitted.
*/
#ifndef LIBCU_ALLOWCOVERINGINDEXSCAN
#define LIBCU_ALLOWCOVERINGINDEXSCAN true
#endif

/* The minimum PMA size is set to this value multiplied by the database page size in bytes. */
#ifndef LIBCU_SORTERPMASZ
#define LIBCU_SORTERPMASZ 250
#endif

/* Statement journals spill to disk when their size exceeds the following threshold (in bytes). 0 means that statement journals are created and
** written to disk immediately (the default behavior for SQLite versions before 3.12.0).  -1 means always keep the entire statement journal in
** memory.  (The statement journal is also always held entirely in memory if journal_mode=MEMORY or if temp_store=MEMORY, regardless of this setting.)
*/
#ifndef LIBCU_STMTJRNLSPILL 
#define LIBCU_STMTJRNLSPILL (64*1024)
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
_WSD struct RuntimeConfig _runtimeConfig = {
	{0,0},						// appendFormat
	LIBCU_DEFAULTMEMSTATUS,		// memstat
	true,						// coreMutex
	LIBCU_THREADSAFE == 1,		// fullMutex
	LIBCU_USEURI,				// openUri
	LIBCU_ALLOWCOVERINGINDEXSCAN, // useCis
	0x7ffffffe,					// maxStrlen
	0,							// neverCorrupt
	LIBCU_DEFAULTLOOKASIDE,		// lookasideSize, lookasides
	LIBCU_STMTJRNLSPILL,		// stmtSpills
	{0,0,0,0,0,0,0,0},			// allocSystem
	{0,0,0,0,0,0,0,0,0},		// mutexSystem
	nullptr, //{0,0,0,0,0,0,0,0,0,0,0,0,0},// pcache2System
	(void *)nullptr,            // heap
	0,							// heapSize
	0, 0,						// minHeapSize, maxHeapSize
	LIBCU_DEFAULTMMAPSIZE,		// sizeMmap
	LIBCU_MAXMMAPSIZE,			// maxMmap
	(void *)nullptr,			// page
	0,							// pageSize
	LIBCU_DEFAULTPCACHEINITSZ,	// pages
	0,							// maxParserStack
	false,						// sharedCacheEnabled
	LIBCU_SORTERPMASZ,			// szPma
	/* All the rest should always be initialized to zero */
	false,						// isInit
	false,						// inProgress
	false,						// isMutexInit
	false,						// isMallocInit
	false,						// isPCacheInit
	0,							// initMutexRefs
	0,							// initMutex
	nullptr,					// log
	nullptr,					// logArg
#ifdef LIBCU_ENABLE_SQLLOG
	nullptr,					// sqllog
	nullptr,					// sqllogArg
#endif
#ifdef LIBCU_VDBE_COVERAGE
	nullptr,                    // vdbeBranch
	nullptr,                    // vbeBranchArg
#endif
#ifndef LIBCU_UNTESTABLE
	nullptr,					// testCallback
#endif
	0,							// localtimeFault
	0x7ffffffe					// onceResetThreshold
};
