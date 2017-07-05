#include <ext/global.h>
#include <stringcu.h>
#include <stdargcu.h>
#include <assert.h>

#define LIBCU_VERSION "1"
#define LIBCU_SOURCE_ID "1"
#define LIBCU_VERSION_NUMBER 1

#ifndef LIBCU_AMALGAMATION
/* IMPLEMENTATION-OF: R-46656-45156 The runtime_version[] string constant contains the text of LIBCU_VERSION macro. */
const char libcu_version[] = LIBCU_VERSION;
#endif

/* IMPLEMENTATION-OF: R-53536-42575 The sqlite3_libversion() function returns a pointer to the to the sqlite3_version[] string constant.  */
__host_device__ const char *libcu_libversion() { return libcu_version; }

/* IMPLEMENTATION-OF: R-63124-39300 The sqlite3_sourceid() function returns a pointer to a string constant whose value is the same as the
** SQLITE_SOURCE_ID C preprocessor macro. 
*/
__host_device__ const char *libcu_sourceid() { return LIBCU_SOURCE_ID; }

/* IMPLEMENTATION-OF: R-35210-63508 The sqlite3_libversion_number() function returns an integer equal to SQLITE_VERSION_NUMBER. */
__host_device__ int libcu_libversion_number() { return LIBCU_VERSION_NUMBER; }

/* IMPLEMENTATION-OF: R-20790-14025 The sqlite3_threadsafe() function returns zero if and only if Libcu was compiled with mutexing code omitted due to
** the SQLITE_THREADSAFE compile-time option being set to 0.
*/
__host_device__ int libcu_threadsafe() { return LIBCU_THREADSAFE; }

/*
** When compiling the test fixture or with debugging enabled (on Win32), this variable being set to non-zero will cause OSTRACE macros to emit
** extra diagnostic information.
*/
#ifdef LIBCU_HAVE_OSTRACE
#ifndef LIBCU_DEBUG_OSTRACE
#define LIBCU_DEBUG_OSTRACE false
#endif
bool _runtimeOSTrace = LIBCU_DEBUG_OSTRACE;
#endif

#if !defined(LIBCU_OMIT_TRACE) && defined(LIBCU_ENABLE_IOTRACE)
/*
** If the following function pointer is not NULL and if SQLITE_ENABLE_IOTRACE is enabled, then messages describing
** I/O active are written using this function.  These messages are intended for debugging activity only.
*/
//void (*sqlite3IoTrace)(const char*, ...) = 0;
#endif

/*
** If the following global variable points to a string which is the name of a directory, then that directory will be used to store
** temporary files.
**
** See also the "PRAGMA temp_store_directory" SQL command.
*/
__hostb_device__ char *_tempDirectory = nullptr;

/*
** If the following global variable points to a string which is the name of a directory, then that directory will be used to store
** all database files specified with a relative pathname.
**
** See also the "PRAGMA data_store_directory" SQL command.
*/
__hostb_device__ char *_dataDirectory = nullptr;

/*
** Initialize Libcu.  
**
** This routine must be called to initialize the memory allocation, VFS, and mutex subsystems prior to doing any serious work with
** Libcu.  But as long as you do not compile with SQLITE_OMIT_AUTOINIT this routine will be called automatically by key routines such as
** sqlite3_open().  
**
** This routine is a no-op except on its very first call for the process, or for the first call after a call to sqlite3_shutdown.
**
** The first thread to call this routine runs the initialization to completion.  If subsequent threads call this routine before the first
** thread has finished the initialization process, then the subsequent threads must block until the first thread finishes with the initialization.
**
** The first thread might call this routine recursively.  Recursive calls to this routine should not block, of course.  Otherwise the
** initialization process would never complete.
**
** Let X be the first thread to enter this routine.  Let Y be some other thread.  Then while the initial invocation of this routine by X is
** incomplete, it is required that:
**
**    *  Calls to this routine from Y must block until the outer-most call by X completes.
**
**    *  Recursive calls to this routine from thread X return immediately without blocking.
*/
__host_device__ RC runtimeInitialize()
{
	RC rc;
#ifdef OMIT_WSD
	rc = __wsdinit(4096, 24);
	if (rc != RC_OK)
		return rc;
#endif

	/* If the following assert() fails on some obscure processor/compiler combination, the work-around is to set the correct pointer
	** size at compile-time using -DSQLITE_PTRSIZE=n compile-time option */
	assert(_PTRSIZE == sizeof(char *));

	/* If Libcu is already completely initialized, then this call to sqlite3_initialize() should be a no-op.  But the initialization
	** must be complete.  So isInit must not be set until the very end of this routine.
	*/
	if (_runtimeConfig.isInit) return RC_OK;

	/* Make sure the mutex subsystem is initialized.  If unable to initialize the mutex subsystem, return early with the error.
	** If the system is so sick that we are unable to allocate a mutex, there is not much Libcu is going to be able to do.
	**
	** The mutex subsystem must take care of serializing its own initialization.
	*/
	rc = mutexInitialize();
	if (rc) return rc;

	/* Initialize the malloc() system and the recursive pInitMutex mutex. This operation is protected by the STATIC_MASTER mutex.  Note that
	** MutexAlloc() is called for a static mutex prior to initializing the malloc subsystem - this implies that the allocation of a static
	** mutex must not require support from the malloc subsystem.
	*/
	MUTEX_LOGIC(mutex *master = mutexAlloc(MUTEX_STATIC_MASTER);)
		mutex_enter(master);
	_runtimeConfig.isMutexInit = true;
	if (!_runtimeConfig.isMallocInit)
		rc = allocInitialize();
	if (rc == RC_OK) {
		_runtimeConfig.isMallocInit = true;
		if (!_runtimeConfig.initMutex) {
			_runtimeConfig.initMutex = mutexAlloc(MUTEX_RECURSIVE);
			if (_runtimeConfig.coreMutex && !_runtimeConfig.initMutex) 
				rc = RC_NOMEM_BKPT;
		}
	}
	if (rc == RC_OK)
		_runtimeConfig.initMutexRefs++;
	mutex_leave(master);

	/* If rc is not SQLITE_OK at this point, then either the malloc subsystem could not be initialized or the system failed to allocate
	** the pInitMutex mutex. Return an error in either case.  */
	if (rc != RC_OK)
		return rc;

	/* Do the rest of the initialization under the recursive mutex so that we will be able to handle recursive calls into
	** sqlite3_initialize().  The recursive calls normally come through sqlite3_os_init() when it invokes sqlite3_vfs_register(), but other
	** recursive calls might also be possible.
	**
	** IMPLEMENTATION-OF: R-00140-37445 Libcu automatically serializes calls to the xInit method, so the xInit method need not be threadsafe.
	**
	** The following mutex is what serializes access to the appdef pcache xInit methods.  The sqlite3_pcache_methods.xInit() all is embedded in the
	** call to sqlite3PcacheInitialize().
	*/
#ifdef LIBCU_EXTRAINIT
	bool runExtraInit = false; // Extra initialization needed
#endif
	mutex_enter(_runtimeConfig.initMutex);
	if (!_runtimeConfig.isInit && !_runtimeConfig.inProgress) {
		_runtimeConfig.inProgress = true;
#ifdef LIBCU_ENABLE_SQLLOG
		{
			extern void sqlite3_init_sqllog();
			sqlite3_init_sqllog();
		}
#endif
		//memset(&sqlite3BuiltinFunctions, 0, sizeof(sqlite3BuiltinFunctions));
		//sqlite3RegisterBuiltinFunctions();
		if (!_runtimeConfig.isPCacheInit)
			rc = allocCacheInitialize();
		if (rc == RC_OK) {
			_runtimeConfig.isPCacheInit = true;
			rc = vsystemInitialize();
		}
		if (rc == RC_OK) {
			allocCacheBufferSetup(_runtimeConfig.page, _runtimeConfig.pageSize, _runtimeConfig.pages);
			_runtimeConfig.isInit = true;
#ifdef LIBCU_EXTRAINIT
			runExtraInit = true;
#endif
		}
		_runtimeConfig.inProgress = false;
	}
	mutex_leave(_runtimeConfig.initMutex);

	// Go back under the static mutex and clean up the recursive mutex to prevent a resource leak.
	mutex_enter(master);
	_runtimeConfig.initMutexRefs--;
	if (_runtimeConfig.initMutexRefs <= 0) {
		assert(!_runtimeConfig.initMutexRefs);
		mutex_free(_runtimeConfig.initMutex);
		_runtimeConfig.initMutex = nullptr;
	}
	mutex_leave(master);

	// The following is just a sanity check to make sure Libcu has been compiled correctly.  It is important to run this code, but
	// we don't want to run it too often and soak up CPU cycles for no reason.  So we run it once during initialization.
#ifndef NDEBUG
#ifndef OMIT_FLOATING_POINT
	// This section of code's only "output" is via assert() statements.
	if (rc == RC_OK) {
		uint64_t x = (((uint64_t)1)<<63)-1;
		double y;
		assert(sizeof(x) == 8);
		assert(sizeof(x) == sizeof(y));
		memcpy(&y, &x, 8);
		assert(isnan(y));
	}
#endif
#endif

	// Do extra initialization steps requested by the LIBCU_EXTRAINIT compile-time option.
#ifdef LIBCU_EXTRAINIT
	RC LIBCU_EXTRAINIT(const char *);
	if (runExtraInit)
		rc = LIBCU_EXTRAINIT(nullptr);
#endif

	return rc;
}

/*
** Undo the effects of sqlite3_initialize().  Must not be called while there are outstanding database connections or memory allocations or
** while any part of Libcu is otherwise in use in any thread.  This routine is not threadsafe.  But it is safe to invoke this routine
** on when Libcu is already shut down.  If Libcu is already shut down when this routine is invoked, then this routine is a harmless no-op.
*/
__host_device__ RC runtimeShutdown()
{
#ifdef OMIT_WSD
	int rc = wsdinit(4096, 24);
	if (rc != RC_OK)
		return rc;
#endif
	if (_runtimeConfig.isInit) {
#ifdef LIBCU_EXTRASHUTDOWN
		void LIBCU_EXTRASHUTDOWN();
		LIBCU_EXTRASHUTDOWN();
#endif
		vsystemShutdown();
		//sqlite3_reset_auto_extension();
		_runtimeConfig.isInit = false;
	}
	if (_runtimeConfig.isPCacheInit) {
		allocCacheShutdown();
		_runtimeConfig.isPCacheInit = false;
	}
	if (_runtimeConfig.isMallocInit) {
		allocShutdown();
		_runtimeConfig.isMallocInit = false;
#ifndef OMIT_SHUTDOWN_DIRECTORIES
		// The heap subsystem has now been shutdown and these values are supposed to be NULL or point to memory that was obtained from sqlite3_malloc(),
		// which would rely on that heap subsystem; therefore, make sure these values cannot refer to heap memory that was just invalidated when the
		// heap subsystem was shutdown.  This is only done if the current call to this function resulted in the heap subsystem actually being shutdown.
		_dataDirectory = nullptr;
		_tempDirectory = nullptr;
#endif
	}
	if (_runtimeConfig.isMutexInit) {
		mutexShutdown();
		_runtimeConfig.isMutexInit = false;
	}
	return RC_OK;
}

/*
** This API allows applications to modify the global configuration of the Libcu library at run-time.
**
** This routine should only be called when there are no outstanding database connections or memory allocations.  This routine is not
** threadsafe.  Failure to heed these warnings can lead to unpredictable behavior.
*/
__host_device__ RC runtimeConfigv(CONFIG op, va_list va)
{
	RC rc = RC_OK;
	/* runtimeConfig() shall return RC_MISUSE if it is invoked while the Libcu library is in use. */
	if (_runtimeConfig.isInit) return RC_MISUSE_BKPT;
	switch (op) {
		/* Mutex configuration options are only available in a threadsafe compile. */
#if defined(LIBCU_THREADSAFE) && LIBCU_THREADSAFE > 0  // IMP: R-54466-46756
	case CONFIG_SINGLETHREAD: {
		/* EVIDENCE-OF: R-02748-19096 This option sets the threading mode to Single-thread. */
		_runtimeConfig.coreMutex = false;  // Disable mutex on core
		_runtimeConfig.fullMutex = false;  // Disable mutex on connections
		break; }
#endif
#if defined(LIBCU_THREADSAFE) && LIBCU_THREADSAFE > 0 // IMP: R-20520-54086
	case CONFIG_MULTITHREAD: {
		/* EVIDENCE-OF: R-14374-42468 This option sets the threading mode to Multi-thread. */
		_runtimeConfig.coreMutex = true;	// Enable mutex on core
		_runtimeConfig.fullMutex = false;  // Disable mutex on connections
		break; }
#endif
#if defined(LIBCU_THREADSAFE) && LIBCU_THREADSAFE > 0 // IMP: R-59593-21810
	case CONFIG_SERIALIZED: {
		/* EVIDENCE-OF: R-41220-51800 This option sets the threading mode to Serialized. */
		_runtimeConfig.coreMutex = true;	// Enable mutex on core
		_runtimeConfig.fullMutex = true;	// Enable mutex on connections
		break; }
#endif
#if defined(LIBCU_THREADSAFE) && LIBCU_THREADSAFE > 0 // IMP: R-63666-48755
	case CONFIG_MUTEX: {
		/* Specify an alternative mutex implementation */
		_runtimeConfig.mutexSystem = *va_arg(va, mutex_methods *);
		break; }
#endif
#if defined(LIBCU_THREADSAFE) && LIBCU_THREADSAFE > 0 // IMP: R-14450-37597
	case CONFIG_GETMUTEX: {
		/* Retrieve the current mutex implementation */
		*va_arg(va, mutex_methods *) = _runtimeConfig.mutexSystem;
		break; }
#endif
	case CONFIG_MALLOC: {
		/* EVIDENCE-OF: R-55594-21030 The SQLITE_CONFIG_MALLOC option takes a single argument which is a pointer to an instance of the
		** sqlite3_mem_methods structure. The argument specifies alternative low-level memory allocation routines to be used in place of the memory
		** allocation routines built into Libcu. */
		__allocsystem = *va_arg(va, alloc_methods *);
		break; }
	case CONFIG_GETMALLOC: {
		/* EVIDENCE-OF: R-51213-46414 The SQLITE_CONFIG_GETMALLOC option takes a single argument which is a pointer to an instance of the
		** sqlite3_mem_methods structure. The sqlite3_mem_methods structure is filled with the currently defined memory allocation routines. */
		if (!__allocsystem.alloc) __allocsystemSetDefault();
		*va_arg(va, alloc_methods *) = __allocsystem;
		break; }
	case CONFIG_MEMSTATUS: {
		/* EVIDENCE-OF: R-61275-35157 The SQLITE_CONFIG_MEMSTATUS option takes single argument of type int, interpreted as a boolean, which enables
		** or disables the collection of memory allocation statistics. */
		_runtimeConfig.memstat = va_arg(va, int);
		break; }
	case CONFIG_SCRATCH: {
		/* EVIDENCE-OF: R-08404-60887 There are three arguments to SQLITE_CONFIG_SCRATCH: A pointer an 8-byte aligned memory buffer from
		** which the scratch allocations will be drawn, the size of each scratch allocation (sz), and the maximum number of scratch allocations (N). */
		_runtimeConfig.scratch = va_arg(va, void *);
		_runtimeConfig.scratchSize = va_arg(va, int);
		_runtimeConfig.scratchs = va_arg(va, int);
		break; }
	case CONFIG_PAGECACHE: {
		/* EVIDENCE-OF: R-18761-36601 There are three arguments to SQLITE_CONFIG_PAGECACHE: A pointer to 8-byte aligned memory (pMem),
		** the size of each page cache line (sz), and the number of cache lines (N). */
		_runtimeConfig.page = va_arg(va, void *);
		_runtimeConfig.pageSize = va_arg(va, int);
		_runtimeConfig.pages = va_arg(va, int);
		break; }
	case CONFIG_PCACHE_HDRSZ: {
		/* EVIDENCE-OF: R-39100-27317 The SQLITE_CONFIG_PCACHE_HDRSZ option takes a single parameter which is a pointer to an integer and writes into
		** that integer the number of extra bytes per page required for each page in SQLITE_CONFIG_PAGECACHE. */
		//*va_arg(va, int *) =  sqlite3HeaderSizeBtree() + sqlite3HeaderSizePcache() + sqlite3HeaderSizePcache1();
		break; }
	case CONFIG_PCACHE: {
		/* no-op */
		break; }
	case CONFIG_GETPCACHE: {
		/* now an error */
		rc = RC_ERROR;
		break; }
	case CONFIG_PCACHE2: {
		/* EVIDENCE-OF: R-63325-48378 The SQLITE_CONFIG_PCACHE2 option takes a single argument which is a pointer to an sqlite3_pcache_methods2
		** object. This object specifies the interface to a custom page cache implementation. */
		_runtimeConfig.pcache2System = *va_arg(va, pcache_methods2 *);
		break; }
	case CONFIG_GETPCACHE2: {
		/* EVIDENCE-OF: R-22035-46182 The SQLITE_CONFIG_GETPCACHE2 option takes a single argument which is a pointer to an sqlite3_pcache_methods2
		** object. Libcu copies of the current page cache implementation into that object. */
		if (!_runtimeConfig.pcache2System.initialize)
			sqlite3PCacheSetDefault();
		*va_arg(va, pcache_methods2 *) = _runtimeConfig.pcache2System;
		break; }
							/* EVIDENCE-OF: R-06626-12911 The SQLITE_CONFIG_HEAP option is only available if Libcu is compiled with either SQLITE_ENABLE_MEMSYS3 or
							** SQLITE_ENABLE_MEMSYS5 and returns SQLITE_ERROR if invoked otherwise. */
#if defined(LIBCU_ENABLE_MEMSYS3) || defined(LIBCU_ENABLE_MEMSYS5)
	case CONFIG_HEAP: {
		/* EVIDENCE-OF: R-19854-42126 There are three arguments to SQLITE_CONFIG_HEAP: An 8-byte aligned pointer to the memory, the
		** number of bytes in the memory buffer, and the minimum allocation size.
		*/
		_runtimeConfig.heap = va_arg(va, void *);
		_runtimeConfig.heapSize = va_arg(va, int);
		_runtimeConfig.minHeapSize = va_arg(va, int);
		if (_runtimeConfig.minHeapSize < 1)
			_runtimeConfig.minHeapSize = 1;
		else if (_runtimeConfig.minHeapSize > (1<<12))
			/* cap min request size at 2^12 */
			_runtimeConfig.minHeapSize = (1<<12);
		if (!_runtimeConfig.heap) 
			/* EVIDENCE-OF: R-49920-60189 If the first pointer (the memory pointer) is NULL, then Libcu reverts to using its default memory allocator
			** (the system malloc() implementation), undoing any prior invocation of SQLITE_CONFIG_MALLOC.
			**
			** Setting _runtimeConfig.m to all zeros will cause malloc to revert to its default implementation when sqlite3_initialize() is run
			*/
			memset(&_runtimeConfig.allocSystem, 0, sizeof(_runtimeConfig.allocSystem));
		else {
			/* EVIDENCE-OF: R-61006-08918 If the memory pointer is not NULL then the alternative memory allocator is engaged to handle all of SQLites
			** memory allocation needs. */
#ifdef LIBCU_ENABLE_MEMSYS3
			_runtimeConfig.alloc = *sqlite3MemGetMemsys3();
#endif
#ifdef LIBCU_ENABLE_MEMSYS5
			_runtimeConfig.alloc = *sqlite3MemGetMemsys5();
#endif
		}
		break; }
#endif
	case CONFIG_LOOKASIDE: {
		_runtimeConfig.lookasideSize = va_arg(va, int);
		_runtimeConfig.lookasides = va_arg(va, int);
		break; }
						   /* Record a pointer to the logger function and its first argument. The default is NULL.  Logging is disabled if the function pointer is NULL. */
	case CONFIG_LOG: {
		/* MSVC is picky about pulling func ptrs from va lists. http://support.microsoft.com/kb/47961
		** _runtimeConfig.xLog = va_arg(va, void(*)(void*,int,const char*));
		*/
		typedef void(*LOGFUNC_t)(void*,int,const char*);
		_runtimeConfig.log = va_arg(va, LOGFUNC_t);
		_runtimeConfig.logArg = va_arg(va, void *);
		break; }
					 /* EVIDENCE-OF: R-55548-33817 The compile-time setting for URI filenames can be changed at start-time using the
					 ** sqlite3_config(SQLITE_CONFIG_URI,1) or sqlite3_config(SQLITE_CONFIG_URI,0) configuration calls.
					 */
	case CONFIG_URI: {
		/* EVIDENCE-OF: R-25451-61125 The SQLITE_CONFIG_URI option takes a single argument of type int. If non-zero, then URI handling is globally
		** enabled. If the parameter is zero, then URI handling is globally disabled. */
		_runtimeConfig.openUri = va_arg(va, int);
		break; }

					 //case CONFIG_COVERING_INDEX_SCAN: {
					 //	/* EVIDENCE-OF: R-36592-02772 The SQLITE_CONFIG_COVERING_INDEX_SCAN option takes a single integer argument which is interpreted as a
					 //	** boolean in order to enable or disable the use of covering indices for full table scans in the query optimizer. */
					 //	_runtimeConfig.bUseCis = va_arg(va, int);
					 //	break; }
#ifdef LIBCU_ENABLE_SQLLOG
	case CONFIG_SQLLOG: {
		typedef void(*SQLLOGFUNC_t)(void*,tagbase_t*,const char*,int);
		_runtimeConfig.sqllog = va_arg(va, SQLLOGFUNC_t);
		_runtimeConfig.sqllogArg = va_arg(va, void *);
		break; }
#endif
	case CONFIG_MMAP_SIZE: {
		/* EVIDENCE-OF: R-58063-38258 SQLITE_CONFIG_MMAP_SIZE takes two 64-bit integer (sqlite3_int64) values that are the default mmap size limit
		** (the default setting for PRAGMA mmap_size) and the maximum allowed mmap size limit. */
		int64_t sizeMmap = va_arg(va, int64_t);
		int64_t maxMmap = va_arg(va, int64_t);
		/* EVIDENCE-OF: R-53367-43190 If either argument to this option is negative, then that argument is changed to its compile-time default.
		**
		** EVIDENCE-OF: R-34993-45031 The maximum allowed mmap size will be silently truncated if necessary so that it does not exceed the
		** compile-time maximum mmap size set by the SQLITE_MAX_MMAP_SIZE compile-time option.
		*/
		if (maxMmap < 0 || maxMmap > LIBCU_MAXMMAPSIZE)
			maxMmap = LIBCU_MAXMMAPSIZE;
		if (sizeMmap < 0) sizeMmap = LIBCU_DEFAULTMMAPSIZE;
		if (sizeMmap > maxMmap) sizeMmap = maxMmap;
		_runtimeConfig.maxMmap = maxMmap;
		_runtimeConfig.sizeMmap = sizeMmap;
		break; }
#if LIBCU_OS_WIN && defined(LIBCU_WIN32_MALLOC) // IMP: R-04780-55815
	case CONFIG_WIN32_HEAPSIZE: {
		/* EVIDENCE-OF: R-34926-03360 SQLITE_CONFIG_WIN32_HEAPSIZE takes a 32-bit unsigned integer value that specifies the maximum size of the created heap. */
		_runtimeConfig.heaps = va_arg(va, int);
		break; }
#endif
	case CONFIG_PMASZ: {
		_runtimeConfig.sizePma = va_arg(va, unsigned int);
		break; }
	case CONFIG_STMTJRNL_SPILL: {
		_runtimeConfig.stmtSpills = va_arg(va, int);
		break; }
	default: {
		rc = RC_ERROR;
		break; }
	}
	return rc;
}
#ifndef __CUDA_ARCH__
__host_device__ RC runtimeConfig(CONFIG op, ...) { va_list va; va_start(va, op); RC rc = runtimeConfigv(op, va); va_end(va); return rc; }
#else
STDARG(RC, runtimeConfig, runtimeConfigv(op, va), CONFIG op);
#endif

/*
** Set up the lookaside buffers for a database connection. Return SQLITE_OK on success.  
** If lookaside is already active, return SQLITE_BUSY.
**
** The size parameter is the number of bytes in each lookaside slot. The count parameter is the number of slots.  If pStart is NULL the
** space for the lookaside memory is obtained from sqlite3_malloc(). If pStart is not NULL then it is size*count bytes of memory to use for
** the lookaside memory.
*/
static __host_device__ bool setupLookaside(tagbase_t *tag, void *buf, int size, int count)
{
#ifndef OMIT_LOOKASIDE
	if (tag->lookaside.outs)
		return false;
	// Free any existing lookaside buffer for this handle before allocating a new one so we don't have to have space for both at the same time.
	if (tag->lookaside.malloced)
		mfree(tag->lookaside.start);
	// The size of a lookaside slot after ROUNDDOWN8 needs to be larger than a pointer to be useful.
	size = _ROUNDDOWN8(size); // IMP: R-33038-09382
	if (size <= (int)sizeof(LookasideSlot *)) size = 0;
	if (count < 0) count = 0;
	void *start;
	if (!size || !count) {
		size = 0;
		start = nullptr;
	}
	else if (!buf) {
		allocBenignBegin();
		start = alloc(size * count); // IMP: R-61949-35727
		allocBenignEnd();
		if (start) count = allocSize(start) / size;
	}
	else start = buf;
	tag->lookaside.start = start;
	tag->lookaside.free = nullptr;
	tag->lookaside.size = (uint16_t)size;
	if (start) {
		assert(size > (int)sizeof(LookasideSlot *));
		LookasideSlot *p = (LookasideSlot *)start;
		for (int i = count-1; i >= 0; i--) {
			p->next = tag->lookaside.free;
			tag->lookaside.free = p;
			p = (LookasideSlot *)&((uint8_t *)p)[size];
		}
		tag->lookaside.end = p;
		tag->lookaside.disable = 0;
		tag->lookaside.malloced = !buf;
	}
	else {
		tag->lookaside.start = tag;
		tag->lookaside.end = tag;
		tag->lookaside.disable = 1;
		tag->lookaside.malloced = false;
	}
#endif
	return true;
}

/*
** This is the routine that actually formats the sqlite3_log() message. We house it in a separate routine from sqlite3_log() to avoid using
** stack space on small-stack systems when logging is disabled.
**
** sqlite3_log() must render into a static buffer.  It cannot dynamically allocate memory because it might be called while the memory allocator
** mutex is held.
**
** sqlite3VXPrintf() might ask for *temporary* memory allocations for certain format characters (%q) or for very large precisions or widths.
** Care must be taken that any sqlite3_log() calls that occur while the memory mutex is held do not use these mechanisms.
*/
static __host_device__ void renderLogMsg(int errCode, const char *format, va_list va)
{
	//strbld_t b; // String accumulator
	//char msg[SQLITE_PRINT_BUF_SIZE*3]; // Complete log message
	//strbldInit(&b, nullptr, msg, sizeof(msg), 0);
	//strbldAppendFormat(&b, format, va);
	//_runtimeConfig.log(_runtimeConfig.logArg, errCode, strbldToString(&b));
}

/* Format and write a message to the log if logging is enabled. */
__host_device__ void runtimeLogv(int errCode, const char *format, va_list va)
{
	if (_runtimeConfig.log)
		renderLogMsg(errCode, format, va);
}
#ifndef __CUDA_ARCH__
__host_device__ void runtimeLog(int errCode, const char *format, ...) { va_list va; va_start(va, format); runtimeLogv(errCode, format, va); va_end(va); }
#else
STDARG1void(runtimeLog, runtimeLogv(errCode, format, va), int errCode, const char *format);
STDARG2void(runtimeLog, runtimeLogv(errCode, format, va), int errCode, const char *format);
STDARG3void(runtimeLog, runtimeLogv(errCode, format, va), int errCode, const char *format);
#endif

#if defined(_DEBUG) || defined(LIBCU_HAVE_OS_TRACE)
/*
** A version of printf() that understands %lld.  Used for debugging. The printf() built into some versions of windows does not understand %lld
** and segfaults if you give it a long long int.
*/
//void sqlite3DebugPrintf(const char *zFormat, ...)
//{
//	va_list ap;
//	StrAccum acc;
//	char zBuf[500];
//	sqlite3StrAccumInit(&acc, 0, zBuf, sizeof(zBuf), 0);
//	va_start(ap,zFormat);
//	sqlite3VXPrintf(&acc, zFormat, ap);
//	va_end(ap);
//	sqlite3StrAccumFinish(&acc);
//	fprintf(stdout,"%s", zBuf);
//	fflush(stdout);
//}
#endif

/*
** The following routines are substitutes for constants SQLITE_CORRUPT, SQLITE_MISUSE, SQLITE_CANTOPEN, SQLITE_NOMEM and possibly other error
** constants.  They serve two purposes:
**
**   1.  Serve as a convenient place to set a breakpoint in a debugger to detect when version error conditions occurs.
**
**   2.  Invoke sqlite3_log() to provide the source code location where a low-level error is first detected.
*/
static __host_device__ int reportError(int err, int lineno, const char *type)
{
	runtimeLog(err, "%s at line %d of [%.10s]", type, lineno, 20+libcu_sourceid());
	return err;
}
__host_device__ int runtimeCorruptError(int lineno) { ASSERTCOVERAGE(_runtimeConfig.log); return reportError(RC_CORRUPT, lineno, "database corruption"); }
__host_device__ int runtimeMisuseError(int lineno) { ASSERTCOVERAGE(_runtimeConfig.log); return reportError(RC_MISUSE, lineno, "misuse"); }
__host_device__ int runtimeCantopenError(int lineno) { ASSERTCOVERAGE(_runtimeConfig.log); return reportError(RC_CANTOPEN, lineno, "cannot open file"); }
#ifdef _DEBUG
__host_device__ int runtimeNomemError(int lineno) { ASSERTCOVERAGE(_runtimeConfig.log); return reportError(RC_NOMEM, lineno, "OOM"); }
__host_device__ int runtimeIoerrnomemError(int lineno){ ASSERTCOVERAGE(_runtimeConfig.log); return reportError(RC_IOERR_NOMEM, lineno, "I/O OOM error"); }
#endif