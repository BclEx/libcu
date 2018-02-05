#include <ext/global.h> //: main.c
#include <stdiocu.h>
#include <stringcu.h>
#include <stdargcu.h>
#include <assert.h>

#define LIBCU_VERSION "1"
#define LIBCU_SOURCE_ID "1"
#define LIBCU_VERSION_NUMBER 1

#ifndef LIBCU_AMALGAMATION
/* IMPLEMENTATION-OF: R-46656-45156 The runtime_version[] string constant contains the text of LIBCU_VERSION macro. */
__host_constant__ const char libcu_version[] = LIBCU_VERSION;
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

/* When compiling the test fixture or with debugging enabled (on Win32), this variable being set to non-zero will cause OSTRACE macros to emit
** extra diagnostic information.
*/
#ifdef LIBCU_HAVE_OS_TRACE
#ifndef LIBCU_DEBUG_OSTRACE
#define LIBCU_DEBUG_OSTRACE 0
#endif
__BEGIN_DECLS;
__hostb_device__ int _libcuOSTrace = LIBCU_DEBUG_OSTRACE;
__END_DECLS;
#endif

#if !defined(LIBCU_OMIT_TRACE) && defined(LIBCU_ENABLE_IOTRACE)
/* If the following function pointer is not NULL and if SQLITE_ENABLE_IOTRACE is enabled, then messages describing
** I/O active are written using this function.  These messages are intended for debugging activity only.
*/
__host_device__ void (*_libcuIoTracev)(const char*,va_list) = nullptr;
#endif

/* If the following global variable points to a string which is the name of a directory, then that directory will be used to store
** temporary files.
**
** See also the "PRAGMA temp_store_directory" SQL command.
*/
__hostb_device__ char *libcu_tempDirectory = nullptr;

/* If the following global variable points to a string which is the name of a directory, then that directory will be used to store
** all database files specified with a relative pathname.
**
** See also the "PRAGMA data_store_directory" SQL command.
*/
__hostb_device__ char *libcu_dataDirectory = nullptr;

/* Initialize Libcu.  
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
	MUTEX_LOGIC(mutex *master = mutexAlloc(MUTEX_STATIC_MASTER));
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
			//extern void sqlite3_init_sqllog();
			//sqlite3_init_sqllog();
		}
#endif
		//memset(&sqlite3BuiltinFunctions, 0, sizeof(sqlite3BuiltinFunctions));
		//sqlite3RegisterBuiltinFunctions();
#if defined(__CUDA_ARCH__)
		if (!_runtimeConfig.isPcacheInit)
			rc = pcacheInitialize();
#endif
		if (rc == RC_OK) {
			_runtimeConfig.isPcacheInit = true;
			rc = vsystemInitialize();
		}
		if (rc == RC_OK) {
#if defined(__CUDA_ARCH__)
			pcacheBufferSetup(_runtimeConfig.page, _runtimeConfig.pageSize, _runtimeConfig.pages);
#endif
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
		assert(math_isnan(y));
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

/* Undo the effects of sqlite3_initialize().  Must not be called while there are outstanding database connections or memory allocations or
** while any part of Libcu is otherwise in use in any thread.  This routine is not threadsafe.  But it is safe to invoke this routine
** on when Libcu is already shut down.  If Libcu is already shut down when this routine is invoked, then this routine is a harmless no-op.
*/
__host_device__ RC runtimeShutdown() //: sqlite3_shutdown
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
#if defined(__CUDA_ARCH__)
	if (_runtimeConfig.isPcacheInit) {
		pcacheShutdown();
		_runtimeConfig.isPcacheInit = false;
	}
#endif
	if (_runtimeConfig.isMallocInit) {
		allocShutdown();
		_runtimeConfig.isMallocInit = false;
#ifndef OMIT_SHUTDOWN_DIRECTORIES
		// The heap subsystem has now been shutdown and these values are supposed to be NULL or point to memory that was obtained from sqlite3_malloc(),
		// which would rely on that heap subsystem; therefore, make sure these values cannot refer to heap memory that was just invalidated when the
		// heap subsystem was shutdown.  This is only done if the current call to this function resulted in the heap subsystem actually being shutdown.
		libcu_dataDirectory = nullptr;
		libcu_tempDirectory = nullptr;
#endif
	}
	if (_runtimeConfig.isMutexInit) {
		mutexShutdown();
		_runtimeConfig.isMutexInit = false;
	}
	return RC_OK;
}

/* This API allows applications to modify the global configuration of the Libcu library at run-time.
**
** This routine should only be called when there are no outstanding database connections or memory allocations.  This routine is not
** threadsafe.  Failure to heed these warnings can lead to unpredictable behavior.
*/
__host_device__ RC runtimeConfigv(CONFIG op, va_list va) //: sqlite3_config
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
		_runtimeConfig.memstat = va_arg(va, int) != 0;
		break; }
	case CONFIG_SMALL_MALLOC: {
		_runtimeConfig.smallMalloc = va_arg(va, int) != 0;
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
#if defined(__CUDA_ARCH__)
		_runtimeConfig.pcache2System = *va_arg(va, pcache_methods *);
#endif
		break; }
	case CONFIG_GETPCACHE2: {
		/* EVIDENCE-OF: R-22035-46182 The SQLITE_CONFIG_GETPCACHE2 option takes a single argument which is a pointer to an sqlite3_pcache_methods2
		** object. Libcu copies of the current page cache implementation into that object. */
#if defined(__CUDA_ARCH__)
		if (!_runtimeConfig.pcache2System.init) __pcachesystemSetDefault();
		*va_arg(va, pcache_methods *) = _runtimeConfig.pcache2System;
#endif
		break; }
#if defined(LIBCU_ENABLE_MEMSYS3) || defined(LIBCU_ENABLE_MEMSYS5)
							/* EVIDENCE-OF: R-06626-12911 The CONFIG_HEAP option is only available if Libcu is compiled with either LIBCU_ENABLE_MEMSYS3 or
							** LIBCU_ENABLE_MEMSYS5 and returns RC_ERROR if invoked otherwise. */
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
		_runtimeConfig.openUri = va_arg(va, int) != 0;
		break; }
	case CONFIG_COVERING_INDEX_SCAN: {
		/* EVIDENCE-OF: R-36592-02772 The SQLITE_CONFIG_COVERING_INDEX_SCAN option takes a single integer argument which is interpreted as a
		** boolean in order to enable or disable the use of covering indices for full table scans in the query optimizer. */
		_runtimeConfig.useCis = va_arg(va, int) != 0;
		break; }
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
		if (maxMmap < 0 || maxMmap > LIBCU_MAXMMAPSIZE) maxMmap = LIBCU_MAXMMAPSIZE;
		if (sizeMmap < 0) sizeMmap = LIBCU_DEFAULTMMAPSIZE;
		if (sizeMmap > maxMmap) sizeMmap = maxMmap;
		_runtimeConfig.maxMmap = maxMmap;
		_runtimeConfig.sizeMmap = sizeMmap;
		break; }
#if __OS_WIN && defined(LIBCU_WIN32_MALLOC) // IMP: R-04780-55815
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

/* Set up the lookaside buffers for a database connection. Return SQLITE_OK on success.  
** If lookaside is already active, return SQLITE_BUSY.
**
** The size parameter is the number of bytes in each lookaside slot. The count parameter is the number of slots.  If pStart is NULL the
** space for the lookaside memory is obtained from sqlite3_malloc(). If pStart is not NULL then it is size*count bytes of memory to use for
** the lookaside memory.
*/
static __host_device__ RC setupLookaside(tagbase_t *tag, void *buf, int size, int count)
{
#ifndef OMIT_LOOKASIDE
	if (taglookasideUsed(tag, 0) > 0) return RC_BUSY;

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
	tag->lookaside.init = nullptr;
	tag->lookaside.free_ = nullptr;
	tag->lookaside.size = (uint16_t)size;
	if (start) {
		assert(size > (int)sizeof(LookasideSlot *));
		LookasideSlot *p = (LookasideSlot *)start;
		for (int i = count-1; i >= 0; i--) {
			p->next = tag->lookaside.init;
			tag->lookaside.init = p;
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
		tag->lookaside.slots = 0;
	}
#endif
	return RC_OK;
}

#if 0

/* Return the mutex associated with a database connection. */
mutex *tagmutex(tagbase_t *tag) //: sqlite3_db_mutex
{
#ifdef ENABLE_API_ARMOR
	if (!tagSafetyCheckOk(tag)) { (void)RC_MISUSE_BKPT; return 0; }
#endif
	return tag->mutex;
}

/* Configuration settings for an individual database connection */
__host_device__ int tagconfig(tagbase_t *db, int op, va_list va) //: sqlite3_db_config
{
	va_list ap;
	int rc;
	va_start(ap, op);
	switch( op ){
	case SQLITE_DBCONFIG_MAINDBNAME: {
		/* IMP: R-06824-28531 */
		/* IMP: R-36257-52125 */
		db->aDb[0].zDbSName = va_arg(ap,char*);
		rc = SQLITE_OK;
		break;
									 }
	case SQLITE_DBCONFIG_LOOKASIDE: {
		void *pBuf = va_arg(ap, void*); /* IMP: R-26835-10964 */
		int sz = va_arg(ap, int);       /* IMP: R-47871-25994 */
		int cnt = va_arg(ap, int);      /* IMP: R-04460-53386 */
		rc = setupLookaside(db, pBuf, sz, cnt);
		break;
									}
	default: {
		static const struct {
			int op;      /* The opcode */
			u32 mask;    /* Mask of the bit in sqlite3.flags to set/clear */
		} aFlagOp[] = {
			{ SQLITE_DBCONFIG_ENABLE_FKEY,           SQLITE_ForeignKeys    },
			{ SQLITE_DBCONFIG_ENABLE_TRIGGER,        SQLITE_EnableTrigger  },
			{ SQLITE_DBCONFIG_ENABLE_FTS3_TOKENIZER, SQLITE_Fts3Tokenizer  },
			{ SQLITE_DBCONFIG_ENABLE_LOAD_EXTENSION, SQLITE_LoadExtension  },
			{ SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE,      SQLITE_NoCkptOnClose  },
			{ SQLITE_DBCONFIG_ENABLE_QPSG,           SQLITE_EnableQPSG     },
		};
		unsigned int i;
		rc = SQLITE_ERROR; /* IMP: R-42790-23372 */
		for(i=0; i<ArraySize(aFlagOp); i++){
			if( aFlagOp[i].op==op ){
				int onoff = va_arg(ap, int);
				int *pRes = va_arg(ap, int*);
				u32 oldFlags = db->flags;
				if( onoff>0 ){
					db->flags |= aFlagOp[i].mask;
				}else if( onoff==0 ){
					db->flags &= ~aFlagOp[i].mask;
				}
				if( oldFlags!=db->flags ){
					sqlite3ExpirePreparedStatements(db);
				}
				if( pRes ){
					*pRes = (db->flags & aFlagOp[i].mask)!=0;
				}
				rc = SQLITE_OK;
				break;
			}
		}
		break;
			 }
	}
	va_end(ap);
	return rc;
}

/* Return true if the buffer z[0..n-1] contains all spaces. */
static __host_device__ int allSpaces(const char *z, int n)
{
	while (n > 0 && z[n - 1] == ' ') n--;
	return n == 0;
}

/* This is the default collating function named "BINARY" which is always available.
**
** If the padFlag argument is not NULL then space padding at the end of strings is ignored.  This implements the RTRIM collation.
*/
static __host_device__ int binCollFunc(void *padFlag, int key1Length, const void *key1, int key2Length, const void *key2)
{
	int n = key1Length < key2Length ? key1Length : key2Length;
	// EVIDENCE-OF: R-65033-28449 The built-in BINARY collation compares strings byte by byte using the memcmp() function from the standard C library.
	assert(key1 && key2);
	int rc = memcmp(key1, key2, n);
	if (!rc){
		// EVIDENCE-OF: R-31624-24737 RTRIM is like BINARY except that extra spaces at the end of either string do not change the result. In other
		// words, strings will compare equal to one another as long as they differ only in the number of spaces at the end.
		if (padFlag && allSpaces(((char *)key1) + n, key1Length - n) && allSpaces(((char *)key2) + n, key2Length - n)) { }
		else rc = key1Length - key2Length;
	}
	return rc;
}

/* Another built-in collating sequence: NOCASE. 
**
** This collating sequence is intended to be used for "case independent comparison". SQLite's knowledge of upper and lower case equivalents
** extends only to the 26 characters used in the English language.
**
** At the moment there is only a UTF-8 implementation.
*/
static __host_device__ int nocaseCollatingFunc(void *notUsed, int key1Length, const void *key1, int key2Length, const void *key2)
{
	int r = strnicmp((const char *)key1, (const char *)key2, key1Length < key2Length ? key1Length : key2Length);
	UNUSED_SYMBOL(notUsed);
	if (!r) 
		r = key1Length - key2Length;
	return r;
}

/* Return the ROWID of the most recent insert */
__host_device__ int64_t taglastInsertRowid(tagbase_t *tag) //: sqlite3_last_insert_rowid
{
#ifdef ENABLE_API_ARMOR
	if (!tagSafetyCheckOk(tag)) { (void)RC_MISUSE_BKPT; return 0; }
#endif
	return tag->lastRowid;
}

/* Set the value returned by the sqlite3_last_insert_rowid() API function. */
__host_device__ void tagsetLastInsertRowid(tagbase_t *tag, int64_t rowid) //: sqlite3_set_last_insert_rowid
{
#ifdef ENABLE_API_ARMOR
	if (!tagSafetyCheckOk(tag)) { (void)RC_MISUSE_BKPT; return; }
#endif
	mutex_enter(tag->mutex);
	tag->lastRowid = rowid;
	mutex_leave(tag->mutex);
}

/* Return the number of changes in the most recent call to sqlite3_exec(). */
__host_device__ int tagchanges(tagbase_t *tag) //: sqlite3_changes
{
#ifdef ENABLE_API_ARMOR
	if (!tagSafetyCheckOk(tag)) { (void)RC_MISUSE_BKPT; return 0; }
#endif
	return tag->changes;
}

/* Return the number of changes since the database handle was opened. */
__host_device__ int tagtotalChanges(tagbase_t *tag) //: sqlite3_total_changes
{
#ifdef ENABLE_API_ARMOR
	if (!tagSafetyCheckOk(tag)) { (void)RC_MISUSE_BKPT; return 0; }
#endif
	return tag->totalChanges;
}

/* Close all open savepoints. This function only manipulates fields of the database handle object, it does not close any savepoints that may be open
** at the b-tree/pager level.
*/
__host_device__ void tagcloseSavepoints(tagbase_t *tag) //: sqlite3CloseSavepoints
{
	while (tag->savepoint) {
		Savepoint *tmp = tag->savepoint;
		tag->savepoint = tmp->next;
		tagfree(db, tmp);
	}
	tag->savepoints = 0;
	tag->statements = 0;
	tag->isTransactionSavepoint = 0;
}

#endif

/* Return a static string containing the name corresponding to the error code specified in the argument. */
#if defined(LIBCU_NEED_ERR_NAME)
__host_device__ const char *libcuErrName(int rc) //: sqlite3ErrName
{
	int origRc = rc;
	const char *name = nullptr;
	for (int i = 0; i < 2 && name == nullptr; i++, rc &= 0xff)
		switch (rc) {
		case RC_OK:                 name = "RC_OK";                break;
		case RC_ERROR:              name = "RC_ERROR";             break;
		case RC_INTERNAL:           name = "RC_INTERNAL";          break;
		case RC_PERM:               name = "RC_PERM";              break;
		case RC_ABORT:              name = "RC_ABORT";             break;
		case RC_ABORT_ROLLBACK:     name = "RC_ABORT_ROLLBACK";    break;
		case RC_BUSY:               name = "RC_BUSY";              break;
		case RC_BUSY_RECOVERY:      name = "RC_BUSY_RECOVERY";     break;
		case RC_BUSY_SNAPSHOT:      name = "RC_BUSY_SNAPSHOT";     break;
		case RC_LOCKED:             name = "RC_LOCKED";            break;
		case RC_LOCKED_SHAREDCACHE: name = "RC_LOCKED_SHAREDCACHE";break;
		case RC_NOMEM:              name = "RC_NOMEM";             break;
		case RC_READONLY:           name = "RC_READONLY";          break;
		case RC_READONLY_RECOVERY:  name = "RC_READONLY_RECOVERY"; break;
		case RC_READONLY_CANTLOCK:  name = "RC_READONLY_CANTLOCK"; break;
		case RC_READONLY_ROLLBACK:  name = "RC_READONLY_ROLLBACK"; break;
		case RC_READONLY_DBMOVED:   name = "RC_READONLY_DBMOVED";  break;
		case RC_INTERRUPT:          name = "RC_INTERRUPT";         break;
		case RC_IOERR:              name = "RC_IOERR";             break;
		case RC_IOERR_READ:         name = "RC_IOERR_READ";        break;
		case RC_IOERR_SHORT_READ:   name = "RC_IOERR_SHORT_READ";  break;
		case RC_IOERR_WRITE:        name = "RC_IOERR_WRITE";       break;
		case RC_IOERR_FSYNC:        name = "RC_IOERR_FSYNC";       break;
		case RC_IOERR_DIR_FSYNC:    name = "RC_IOERR_DIR_FSYNC";   break;
		case RC_IOERR_TRUNCATE:     name = "RC_IOERR_TRUNCATE";    break;
		case RC_IOERR_FSTAT:        name = "RC_IOERR_FSTAT";       break;
		case RC_IOERR_UNLOCK:       name = "RC_IOERR_UNLOCK";      break;
		case RC_IOERR_RDLOCK:       name = "RC_IOERR_RDLOCK";      break;
		case RC_IOERR_DELETE:       name = "RC_IOERR_DELETE";      break;
		case RC_IOERR_NOMEM:        name = "RC_IOERR_NOMEM";       break;
		case RC_IOERR_ACCESS:       name = "RC_IOERR_ACCESS";      break;
		case RC_IOERR_CHECKRESERVEDLOCK: name = "RC_IOERR_CHECKRESERVEDLOCK"; break;
		case RC_IOERR_LOCK:         name = "RC_IOERR_LOCK";        break;
		case RC_IOERR_CLOSE:        name = "RC_IOERR_CLOSE";       break;
		case RC_IOERR_DIR_CLOSE:    name = "RC_IOERR_DIR_CLOSE";   break;
		case RC_IOERR_SHMOPEN:      name = "RC_IOERR_SHMOPEN";     break;
		case RC_IOERR_SHMSIZE:      name = "RC_IOERR_SHMSIZE";     break;
		case RC_IOERR_SHMLOCK:      name = "RC_IOERR_SHMLOCK";     break;
		case RC_IOERR_SHMMAP:       name = "RC_IOERR_SHMMAP";      break;
		case RC_IOERR_SEEK:         name = "RC_IOERR_SEEK";        break;
		case RC_IOERR_DELETE_NOENT: name = "RC_IOERR_DELETE_NOENT";break;
		case RC_IOERR_MMAP:         name = "RC_IOERR_MMAP";        break;
		case RC_IOERR_GETTEMPPATH:  name = "RC_IOERR_GETTEMPPATH"; break;
		case RC_IOERR_CONVPATH:     name = "RC_IOERR_CONVPATH";    break;
		case RC_CORRUPT:            name = "RC_CORRUPT";           break;
		case RC_CORRUPT_VTAB:       name = "RC_CORRUPT_VTAB";      break;
		case RC_NOTFOUND:           name = "RC_NOTFOUND";          break;
		case RC_FULL:               name = "RC_FULL";              break;
		case RC_CANTOPEN:           name = "RC_CANTOPEN";          break;
		case RC_CANTOPEN_NOTEMPDIR: name = "RC_CANTOPEN_NOTEMPDIR";break;
		case RC_CANTOPEN_ISDIR:     name = "RC_CANTOPEN_ISDIR";    break;
		case RC_CANTOPEN_FULLPATH:  name = "RC_CANTOPEN_FULLPATH"; break;
		case RC_CANTOPEN_CONVPATH:  name = "RC_CANTOPEN_CONVPATH"; break;
		case RC_PROTOCOL:           name = "RC_PROTOCOL";          break;
		case RC_EMPTY:              name = "RC_EMPTY";             break;
		case RC_SCHEMA:             name = "RC_SCHEMA";            break;
		case RC_TOOBIG:             name = "RC_TOOBIG";            break;
		case RC_CONSTRAINT:         name = "RC_CONSTRAINT";        break;
		case RC_CONSTRAINT_UNIQUE:  name = "RC_CONSTRAINT_UNIQUE"; break;
		case RC_CONSTRAINT_TRIGGER: name = "RC_CONSTRAINT_TRIGGER";break;
		case RC_CONSTRAINT_FOREIGNKEY: name = "RC_CONSTRAINT_FOREIGNKEY";   break;
		case RC_CONSTRAINT_CHECK:   name = "RC_CONSTRAINT_CHECK";  break;
		case RC_CONSTRAINT_PRIMARYKEY: name = "RC_CONSTRAINT_PRIMARYKEY";   break;
		case RC_CONSTRAINT_NOTNULL: name = "RC_CONSTRAINT_NOTNULL";break;
		case RC_CONSTRAINT_COMMITHOOK: name = "RC_CONSTRAINT_COMMITHOOK";   break;
		case RC_CONSTRAINT_VTAB:    name = "RC_CONSTRAINT_VTAB";   break;
		case RC_CONSTRAINT_FUNCTION: name = "RC_CONSTRAINT_FUNCTION";     break;
		case RC_CONSTRAINT_ROWID:   name = "RC_CONSTRAINT_ROWID";  break;
		case RC_MISMATCH:           name = "RC_MISMATCH";          break;
		case RC_MISUSE:             name = "RC_MISUSE";            break;
		case RC_NOLFS:              name = "RC_NOLFS";             break;
		case RC_AUTH:               name = "RC_AUTH";              break;
		case RC_FORMAT:             name = "RC_FORMAT";            break;
		case RC_RANGE:              name = "RC_RANGE";             break;
		case RC_NOTADB:             name = "RC_NOTADB";            break;
		case RC_ROW:                name = "RC_ROW";               break;
		case RC_NOTICE:             name = "RC_NOTICE";            break;
		case RC_NOTICE_RECOVER_WAL: name = "RC_NOTICE_RECOVER_WAL";break;
		case RC_NOTICE_RECOVER_ROLLBACK: name = "RC_NOTICE_RECOVER_ROLLBACK"; break;
		case RC_WARNING:            name = "RC_WARNING";           break;
		case RC_WARNING_AUTOINDEX:  name = "RC_WARNING_AUTOINDEX"; break;
		case RC_DONE:               name = "RC_DONE";              break;
	}
	if (!name) {
		static char buf[50];
		snprintf(buf, sizeof(buf), "RC_UNKNOWN(%d)", origRc);
		name = buf;
	}
	return name;
}
#endif

#if 0

/* Return a static string that describes the kind of error specified in the argument. */
__host_device__ const char *errStr(int rc) //: sqlite3ErrStr
{
	static const char *const msgs[] = {
		/* RC_OK          */ "not an error",
		/* RC_ERROR       */ "SQL logic error",
		/* RC_INTERNAL    */ 0,
		/* RC_PERM        */ "access permission denied",
		/* RC_ABORT       */ "query aborted",
		/* RC_BUSY        */ "database is locked",
		/* RC_LOCKED      */ "database table is locked",
		/* RC_NOMEM       */ "out of memory",
		/* RC_READONLY    */ "attempt to write a readonly database",
		/* RC_INTERRUPT   */ "interrupted",
		/* RC_IOERR       */ "disk I/O error",
		/* RC_CORRUPT     */ "database disk image is malformed",
		/* RC_NOTFOUND    */ "unknown operation",
		/* RC_FULL        */ "database or disk is full",
		/* RC_CANTOPEN    */ "unable to open database file",
		/* RC_PROTOCOL    */ "locking protocol",
		/* RC_EMPTY       */ 0,
		/* RC_SCHEMA      */ "database schema has changed",
		/* RC_TOOBIG      */ "string or blob too big",
		/* RC_CONSTRAINT  */ "constraint failed",
		/* RC_MISMATCH    */ "datatype mismatch",
		/* RC_MISUSE      */ "bad parameter or other API misuse",
#ifdef LIBCU_DISABLE_LFS
		/* RC_NOLFS       */ "large file support is disabled",
#else
		/* RC_NOLFS       */ 0,
#endif
		/* RC_AUTH        */ "authorization denied",
		/* RC_FORMAT      */ 0,
		/* RC_RANGE       */ "column index out of range",
		/* RC_NOTADB      */ "file is not a database",
	};
	const char *err = "unknown error";
	switch (rc) {
	case LIBCU_ABORT_ROLLBACK: err = "abort due to ROLLBACK"; break;
	default:
		rc &= 0xff;
		if (_ALWAYS(rc >= 0) && rc < _ARRAYSIZE(msgs) && msgs[rc] != 0) err = msgs[rc];
		break;
	}
	return err;
}

/* This routine implements a busy callback that sleeps and tries again until a timeout value is reached.  The timeout value is
** an integer number of milliseconds passed in as the first argument.
*/
static __host_device__ int libcuDefaultBusyCallback(void *p, int count)
{
	tagbase_t *tag = (tagbase_t *)p;
	int timeout = tag->busyTimeout;
#if __OS_WIN || HAVE_USLEEP
	static const uint8_t delays[] = { 1, 2, 5, 10, 15, 20, 25, 25,  25,  50,  50, 100 };
	static const uint8_t totals[] = { 0, 1, 3,  8, 18, 33, 53, 78, 103, 128, 178, 228 };
#define NDELAY _ARRAYSIZE(delays)
	int delay, prior;
	assert(count >= 0);
	if (count < NDELAY) {
		delay = delays[count];
		prior = totals[count];
	}
	else {
		delay = delays[NDELAY - 1];
		prior = totals[NDELAY - 1] + delay * (count - (NDELAY - 1));
	}
	if (prior + delay > timeout) {
		delay = timeout - prior;
		if (delay <= 0) return 0;
	}
	vsys_sleep(tag->vsys, delay * 1000);
	return 1;
#else
	if ((count + 1) * 1000 > timeout)
		return 0;
	vsys_sleep(tag->vsys, 1000000);
	return 1;
#endif
}

/* Invoke the given busy handler.
**
** This routine is called when an operation failed with a lock. If this routine returns non-zero, the lock is retried.  If it
** returns 0, the operation aborts with an SQLITE_BUSY error.
*/
__host_device__ int sqlite3InvokeBusyHandler(BusyHandler *p) //: sqlite3InvokeBusyHandler
{
	if (_NEVER(!p) || !p->func || p->busys < 0) return 0;
	int rc = p->func(p->arg, p->busys);
	if (rc == 0) p->busys = -1;
	else p->busys++;
	return rc; 
}

/*
/* This routine sets the busy callback for an Sqlite database to the given callback function with the given argument. */
__host_device__ int sqlite3_busy_handler(tagbase_t *tag, int (*busy)(void*,int), void *arg) //: sqlite3_busy_handler
{
#ifdef ENABLE_API_ARMOR
	if (!tagSafetyCheckOk(db) ) return RC_MISUSE_BKPT;
#endif
	mutex_enter(tag->mutex);
	tag->busyHandler.func = busy;
	tag->busyHandler.arg = arg;
	tag->busyHandler.busys = 0;
	tag->busyTimeout = 0;
	mutex_leave(tag->mutex);
	return RC_OK;
}

#ifndef OMIT_PROGRESS_CALLBACK
/* This routine sets the progress callback for an Sqlite database to the given callback function with the given argument. The progress callback will
** be invoked every nOps opcodes.
*/
__host_device__ void sqlite3_progress_handler(tagbase_t *tag, int ops, int (*progress)(void*), void *arg) //: sqlite3_progress_handler
{
#ifdef ENABLE_API_ARMOR
	if (!tagSafetyCheckOk(db)) { (void)RC_MISUSE_BKPT; return; }
#endif
	mutex_enter(tag->mutex);
	if (ops > 0) {
		tag->progress = progress;
		tag->progressOps = (unsigned)ops;
		tag->progressArg = arg;
	}
	else {
		tag->progress = nullptr;
		tag->progressOps = 0;
		tag->progressArg = nullptr;
	}
	mutex_leave(tag->mutex);
}
#endif /* OMIT_PROGRESS_CALLBACK */

/* This routine installs a default busy handler that waits for the specified number of milliseconds before returning 0. */
__host_device__ int sqlite3_busy_timeout(tagbase_t *tag, int ms)
{
#ifdef ENABLE_API_ARMOR
	if (!tagSafetyCheckOk(tag)) return RC_MISUSE_BKPT;
#endif
	if (ms > 0) { sqlite3_busy_handler(tag, libcuDefaultBusyCallback, (void *)tag); tag->busyTimeout = ms; }
	else sqlite3_busy_handler(tag, nullptr, nullptr);
	return RC_OK;
}

/* Cause any pending operation to stop at its earliest opportunity. */
__host_device__ void sqlite3_interrupt(tagbase_t *tag) //: sqlite3_interrupt
{
#ifdef ENABLE_API_ARMOR
	if (!tagSafetyCheckOk(tag) && (!tag || tag->magic != TAG_MAGIC_ZOMBIE)) { (void)RC_MISUSE_BKPT; return; }
#endif
	tag->u1.isInterrupted = 1;
}

#ifndef OMIT_TRACE
/* Register a trace callback using the version-2 interface. */
__host_device__ int sqlite3_trace_v2(tagbase_t *tag, unsigned traceMask, int (*trace)(unsigned,void*,void*,void*), void *arg) //: sqlite3_trace_v2
{
#ifdef ENABLE_API_ARMOR
	if (!tagSafetyCheckOk(tag)) return RC_MISUSE_BKPT;
#endif
	mutex_enter(db->mutex);
	if (!traceMask) trace = nullptr;
	if (!trace) traceMask = 0;
	tag->traceMask = traceMask;
	tag->trace = trace;
	tag->traceArg = arg;
	mutex_leave(tag->mutex);
	return RC_OK;
}
#endif /* OMIT_TRACE */

/*
/* Register a function to be invoked when a transaction commits. If the invoked function returns non-zero, then the commit becomes a rollback. */
__host_device__ void *sqlite3_commit_hook(tagbase_t *tag, int (*callback)(void*), void *arg) //: sqlite3_commit_hook
{
#ifdef ENABLE_API_ARMOR
	if (!tagSafetyCheckOk(tag)) { (void)RC_MISUSE_BKPT; return 0; }
#endif
	mutex_enter(tag->mutex);
	void *old = tag->commitArg;
	tag->commitCallback = callback;
	tag->commitArg = arg;
	mutex_leave(tag->mutex);
	return old;
}

/* Register a callback to be invoked each time a row is updated, inserted or deleted using this database connection. */
__host_device__ void *sqlite3_update_hook(tagbase_t *tag, void (*callback)(void*,int,char const*,char const*,int64_t), void *arg) //: sqlite3_update_hook
{
#ifdef ENABLE_API_ARMOR
	if (!tagSafetyCheckOk(tag)) { (void)RC_MISUSE_BKPT; return 0; }
#endif
	mutex_enter(tag->mutex);
	void *r = tag->updateArg;
	tag->updateCallback = callback;
	tag->updateArg = arg;
	mutex_leave(tag->mutex);
	return r;
}

/* Register a callback to be invoked each time a transaction is rolled back by this database connection. */
__host_device__ void *sqlite3_rollback_hook(tagbase_t *tag, void (*callback)(void*), void *arg) //: sqlite3_rollback_hook
{
#ifdef ENABLE_API_ARMOR
	if (!tagSafetyCheckOk(tag)) { (void)RC_MISUSE_BKPT; return 0; }
#endif
	mutex_enter(tag->mutex);
	void *r = tag->rollbackArg;
	tag->rollbackCallback = callback;
	tag->rollbackArg = arg;
	mutex_leave(tag->mutex);
	return r;
}

#ifdef ENABLE_PREUPDATE_HOOK
/* Register a callback to be invoked each time a row is updated, inserted or deleted using this database connection. */
__host_device__ void *sqlite3_preupdate_hook(tagbase_t *tag, void (*callback)(void*,tagbase_t*,int,char const*,char const*,int64_t,int64_t), void *arg) //: sqlite3_preupdate_hook
{
	mutex_enter(tag->mutex);
	void *r = tag->preUpdateArg;
	tag->preUpdateCallback = callback;
	tag->preUpdateArg = arg;
	mutex_leave(db->mutex);
	return r;
}
#endif /* ENABLE_PREUPDATE_HOOK */

#ifndef OMIT_WAL
/* The sqlite3_wal_hook() callback registered by sqlite3_wal_autocheckpoint(). Invoke sqlite3_wal_checkpoint if the number of frames in the log file
** is greater than sqlite3.pWalArg cast to an integer (the value configured by wal_autocheckpoint()).
*/ 
__host_device__ int sqlite3WalDefaultHook(void *clientData, tagbase_t *tag, const char *db, int frames) //: sqlite3WalDefaultHook
{
	if (frames >= _PTR_TO_INT(clientData)) {
		allocBenignBegin();
		sqlite3_wal_checkpoint(tag, db);
		allocBenignEnd();
	}
	return RC_OK;
}
#endif /* OMIT_WAL */

/* Configure an sqlite3_wal_hook() callback to automatically checkpoint a database after committing a transaction if there are nFrame or
** more frames in the log file. Passing zero or a negative value as the nFrame parameter disables automatic checkpoints entirely.
**
** The callback registered by this function replaces any existing callback registered using sqlite3_wal_hook(). Likewise, registering a callback
** using sqlite3_wal_hook() disables the automatic checkpoint mechanism configured by this function.
*/
__host_device__ RC sqlite3_wal_autocheckpoint(tagbase_t *tag, int frames) //: sqlite3_wal_autocheckpoint
{
#ifdef OMIT_WAL
	UNUSED_SYMBOL(db);
	UNUSED_SYMBOL(nFrame);
#else
#ifdef ENABLE_API_ARMOR
	if (!tagSafetyCheckOk(tag)) return RC_MISUSE_BKPT;
#endif
	if (frames > 0) sqlite3_wal_hook(tag, libcuWalDefaultHook, _INT_TO_PTR(frames));
	else sqlite3_wal_hook(tag, nullptr, 0);
#endif
	return RC_OK;
}

/* Register a callback to be invoked each time a transaction is written into the write-ahead-log by this database connection. */
__host_device__ void *sqlite3_wal_hook(tagbase_t *tag, int (*callback)(void*,tagbase_t*,const char*,int), void *arg) //: sqlite3_wal_hook
{
#ifndef OMIT_WAL
#ifdef ENABLE_API_ARMOR
	if (!tagSafetyCheckOk(tag)) { (void)RC_MISUSE_BKPT; return 0; }
#endif
	mutex_enter(tag->mutex);
	void *r = tag->walArg;
	tag->walCallback = callback;
	tag->walArg = arg;
	mutex_leave(tag->mutex);
	return r;
#else
	return nullptr;
#endif
}

/* Checkpoint database zDb. */
__host_device__ RC sqlite3_wal_checkpoint_v2(tagbase_t *tag, const char *db, int mode, int *logs, int *ckpts)
{
#ifdef OMIT_WAL
	return RC_OK;
#else
#ifdef ENABLE_API_ARMOR
	if (!tagSafetyCheckOk(tag)) return RC_MISUSE_BKPT;
#endif
	// Initialize the output variables to -1 in case an error occurs.
	if (logs) *logs = -1;
	if (ckpts) *ckpts = -1;
	assert(CHECKPOINT_PASSIVE == 0);
	assert(CHECKPOINT_FULL == 1);
	assert(CHECKPOINT_RESTART == 2);
	assert(CHECKPOINT_TRUNCATE == 3);
	// EVIDENCE-OF: R-03996-12088 The M parameter must be a valid checkpoint mode:
	if (mode < CHECKPOINT_PASSIVE || mode > CHECKPOINT_TRUNCATE) return RC_MISUSE;
	mutex_enter(tag->mutex);
	int dbId = LIBCU_MAX_ATTACHED; // sqlite3.aDb[] index of db to checkpoint
	if (db && db[0]) dbId = sqlite3FindDbName(db, zDb);
	RC rc; if (dbId < 0) { rc = RC_ERROR; tagErrorWithMsg(tag, RC_ERROR, "unknown database: %s", db); }
	else { tag->busyHandler.busys = 0; rc = tagCheckpoint(tag, dbId, mode, logs, ckpts); tagError(tag, rc); }
	rc = tagApiExit(tag, rc);
	// If there are no active statements, clear the interrupt flag at this point.
	if (!tag->vdbeActive) tag->u1.isInterrupted = 0;
	mutex_leave(tag->mutex);
	return rc;
#endif
}

/* Checkpoint database zDb. If zDb is NULL, or if the buffer zDb points to contains a zero-length string, all attached databases are checkpointed. */
__host_device__ int sqlite3_wal_checkpoint(tagbase_t *tag, const char *db) //: sqlite3_wal_checkpoint
{
	// EVIDENCE-OF: R-41613-20553 The sqlite3_wal_checkpoint(D,X) is equivalent to sqlite3_wal_checkpoint_v2(D,X,SQLITE_CHECKPOINT_PASSIVE,0,0).
	return sqlite3_wal_checkpoint_v2(tag, db, CHECKPOINT_PASSIVE, nullptr, nullptr);
}


/* This function returns true if main-memory should be used instead of a temporary file for transient pager files and statement journals.
** The value returned depends on the value of db->temp_store (runtime parameter) and the compile time value of SQLITE_TEMP_STORE. The
** following table describes the relationship between these two values and this functions return value.
**
**   SQLITE_TEMP_STORE     db->temp_store     Location of temporary database
**   -----------------     --------------     ------------------------------
**   0                     any                file      (return 0)
**   1                     1                  file      (return 0)
**   1                     2                  memory    (return 1)
**   1                     0                  file      (return 0)
**   2                     1                  file      (return 0)
**   2                     2                  memory    (return 1)
**   2                     0                  memory    (return 1)
**   3                     any                memory    (return 1)
*/
__host_device__ int sqlite3TempInMemory(const tagbase_t *tag) //: sqlite3TempInMemory
{
#if LIBCU_TEMPSTORE == 1
	return tag->temp_store == 2;
#endif
#if LIBCU_TEMPSTORE == 2
	return tag->temp_store != 1;
#endif
#if LIBCU_TEMPSTORE == 3
	UNUSED_SYMBOL(tag);
	return 1;
#endif
#if LIBCU_TEMPSTORE < 1 || LIBCU_TEMPSTORE > 3
	UNUSED_SYMBOL(tag);
	return 0;
#endif
}

/* Return UTF-8 encoded English language explanation of the most recent error. */
__host_device__ const char *tagerrmsg(tagbase_t *tag) //: sqlite3_errmsg
{
	if (!tag) return tagErrStr(RC_NOMEM_BKPT);
	if (!tagSafetyCheckSickOrOk(tag)) return tagErrStr(RC_MISUSE_BKPT);
	mutex_enter(tag->mutex);
	const char *z;
	if (tag->mallocFailed) z = sqlite3ErrStr(SQLITE_NOMEM_BKPT);
	else {
		TESTCASE(!tag->err);
		z = (char *)sqlite3_value_text(tag->err);
		assert(!tag->mallocFailed);
		if (!z) z = tagErrStr(tag->errCode);
	}
	mutex_leave(tag->mutex);
	return z;
}

#ifndef OMIT_UTF16
/* Return UTF-16 encoded English language explanation of the most recent error. */
__host__device__ const void *tagerrmsg16(tagbase_t *tag) //: sqlite3_errmsg16
{
	static const uint16_t outOfMem[] = { 'o', 'u', 't', ' ', 'o', 'f', ' ', 'm', 'e', 'm', 'o', 'r', 'y', 0 };
	static const uint16_t misuse[] = { 'b', 'a', 'd', ' ', 'p', 'a', 'r', 'a', 'm', 'e', 't', 'e', 'r', ' ', 'o', 'r', ' ', 'o', 't', 'h', 'e', 'r', ' ', 'A', 'P', 'I', ' ', 'm', 'i', 's', 'u', 's', 'e', 0 };
	const void *z;
	if (!tag) return (void *)outOfMem;
	if (!tagSafetyCheckSickOrOk(tag)) return (void *)misuse;
	mutex_enter(tag->mutex);
	if (tag->mallocFailed) z = (void *)outOfMem;
	else {
		z = sqlite3_value_text16(db->err);
		if (!z){
			tagErrorWithMsg(tag, tag->errCode, tagErrStr(tag->errCode));
			z = sqlite3_value_text16(tag->err);
		}
		// A malloc() may have failed within the call to sqlite3_value_text16() above. If this is the case, then the db->mallocFailed flag needs to
		// be cleared before returning. Do this directly, instead of via sqlite3ApiExit(), to avoid setting the database handle error message.
		tagOomClear(tag);
	}
	mutex_leave(tag->mutex);
	return z;
}
#endif /* OMIT_UTF16 */

/* Return the most recent error code generated by an SQLite routine. If NULL is passed to this function, we assume a malloc() failed during sqlite3_open(). */
__host_device__ int tagerrcode(tagbase_t *tag) //: sqlite3_errcode
{
	if (tag && !tagSafetyCheckSickOrOk(tag)) return RC_MISUSE_BKPT;
	if (!tag || tag->mallocFailed) return RC_NOMEM_BKPT;
	return tag->errCode & tag->errMask;
}

__host_device__ int tagextendedErrcode(tagbase_t *tag) //: sqlite3_extended_errcode
{
	if (tag && !tagSafetyCheckSickOrOk(tag)) return RC_MISUSE_BKPT;
	if (!tag || tag->mallocFailed) return RC_NOMEM_BKPT;
	return tag->errCode;
}

__host_device__ int tagsystemErrno(tagbase_t *tag) //: sqlite3_system_errno
{
	return tag ? tag->sysErrno : 0;
}

/* Return a string that describes the kind of error specified in the argument.  For now, this simply calls the internal sqlite3ErrStr() function. */
__host_device__ const char *errstr(int rc) { return errStr(rc); } //: sqlite3_errstr

#endif

/* The following routines are substitutes for constants SQLITE_CORRUPT, SQLITE_MISUSE, SQLITE_CANTOPEN, SQLITE_NOMEM and possibly other error
** constants.  They serve two purposes:
**
**   1.  Serve as a convenient place to set a breakpoint in a debugger to detect when version error conditions occurs.
**
**   2.  Invoke sqlite3_log() to provide the source code location where a low-level error is first detected.
*/
static __host_device__ int reportError(int err, int lineno, const char *type)
{
	_log(err, "%s at line %d of [%.10s]", type, lineno, 20 + libcu_sourceid());
	return err;
}
__host_device__ int libcuCorruptError(int lineno) { TESTCASE(_runtimeConfig.log); return reportError(RC_CORRUPT, lineno, "database corruption"); } //: sqlite3CorruptError
__host_device__ int libcuMisuseError(int lineno) { TESTCASE(_runtimeConfig.log); return reportError(RC_MISUSE, lineno, "misuse"); } //: sqlite3MisuseError
__host_device__ int libcuCantopenError(int lineno) { TESTCASE(_runtimeConfig.log); return reportError(RC_CANTOPEN, lineno, "cannot open file"); } //: sqlite3CantopenError
#ifdef _DEBUG
__host_device__ int libcuCorruptPgnoError(int lineno, Pgno pgno) { char msg[100]; snprintf(msg, sizeof(msg), "database corruption page %d", pgno); TESTCASE(_runtimeConfig.log); return reportError(RC_CORRUPT, lineno, msg); } //: sqlite3CorruptPgnoError
__host_device__ int libcuNomemError(int lineno) { TESTCASE(_runtimeConfig.log); return reportError(RC_NOMEM, lineno, "OOM"); } //: sqlite3NomemError
__host_device__ int libcuIoerrnomemError(int lineno){ TESTCASE(_runtimeConfig.log); return reportError(RC_IOERR_NOMEM, lineno, "I/O OOM error"); } //: sqlite3IoerrnomemError
#endif

#ifndef OMIT_COMPILEOPTION_DIAGS
/* Given the name of a compile-time option, return true if that option was used and false if not.
**
** The name can optionally begin with "SQLITE_" but the "SQLITE_" prefix is not required for a match.
*/
__host_device__ int sqlite3_compileoption_used(const char *optName) //: sqlite3_compileoption_used
{
#if ENABLE_API_ARMOR
	if (!optName) { (void)RC_MISUSE_BKPT; return 0; }
#endif
	int opts; const char **compileOpts = sqlite3CompileOptions(&opts);
	if (!strnicmp(optName, "SQLITE_", 7)) optName += 7;
	int n = strlen(optName);
	// Since nOpt is normally in single digits, a linear search is adequate. No need for a binary search.
	for (int i = 0; i < opts; i++)
		if (!strnicmp(optName, compileOpts[i], n) && !isidchar((unsigned char)compileOpt[i][n])) return 1;
	return 0;
}

/* Return the N-th compile-time option string.  If N is out of range, return a NULL pointer. */
__host_device__ const char *sqlite3_compileoption_get(int n) //: sqlite3_compileoption_get
{
	int opts; const char **compileOpts = sqlite3CompileOptions(&opts);
	return n >= 0 && n < opts ? compileOpts[n] : nullptr;
}
#endif /* OMIT_COMPILEOPTION_DIAGS */
