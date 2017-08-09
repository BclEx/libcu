#include <ext/global.h>
#include <assert.h>

//#if defined(_TEST) || defined(_DEBUG)
//bool OsTrace = false;
//#define OSTRACE(X, ...) if (OsTrace) { _dprintf("OS: "X, __VA_ARGS__); }
//#else
//#define OSTRACE(X, ...)
//#endif

/*
** If we compile with the SQLITE_TEST macro set, then the following block of code will give us the ability to simulate a disk I/O error.  This
** is used for testing the I/O recovery logic.
*/
#ifdef _TEST
__device__ int libcu_io_error_hit = 0;            // Total number of I/O Errors
__device__ int libcu_io_error_hardhit = 0;        // Number of non-benign errors
__device__ int libcu_io_error_pending = 0;        // Count down to first I/O error
__device__ int libcu_io_error_persist = 0;        // True if I/O errors persist
__device__ int libcu_io_error_benign = 0;         // True if errors are benign
__device__ int libcu_diskfull_pending = 0;
__device__ int libcu_diskfull = 0;
#endif

/* When testing, also keep a count of the number of open files. */
#if defined(_TEST)
int libcu_open_file_count = 0;
#endif

/*
** The default SQLite sqlite3_vfs implementations do not allocate memory (actually, os_unix.c allocates a small amount of memory
** from within OsOpen()), but some third-party implementations may. So we test the effects of a malloc() failing and the sqlite3OsXXX()
** function returning SQLITE_IOERR_NOMEM using the DO_OS_MALLOC_TEST macro.
**
** The following functions are instrumented for malloc() failure testing:
**
**     sqlite3OsRead()
**     sqlite3OsWrite()
**     sqlite3OsSync()
**     sqlite3OsFileSize()
**     sqlite3OsLock()
**     sqlite3OsCheckReservedLock()
**     sqlite3OsFileControl()
**     sqlite3OsShmMap()
**     sqlite3OsOpen()
**     sqlite3OsDelete()
**     sqlite3OsAccess()
**     sqlite3OsFullPathname()
**
*/
#if defined(_TEST)
int libcu_memdebug_vfs_oom_test = 1;
#define DO_OS_MALLOC_TEST(x) \
	if (libcu_memdebug_vfs_oom_test && (!x || !sqlite3JournalIsInMemory(x))) { \
	void *tstAlloc = alloc(10); \
	if (!tstAlloc) return RC_IOERR_NOMEM_BKPT; \
	mfree(tstAlloc); } \
#else
#define DO_OS_MALLOC_TEST(x)
#endif

/*
** The following routines are convenience wrappers around methods of the sqlite3_file object.  This is mostly just syntactic sugar. All
** of this would be completely automatic if SQLite were coded using C++ instead of plain old C.
*/
__host_device__ void sqlite3OsClose(sqlite3_file *pId) { if (pId->pMethods){ pId->pMethods->xClose(pId); pId->pMethods = nullptr; } }
__host_device__ RC sqlite3OsRead(sqlite3_file *id, void *pBuf, int amt, i64 offset) { DO_OS_MALLOC_TEST(id); return id->pMethods->xRead(id, pBuf, amt, offset); }
__host_device__ RC sqlite3OsWrite(sqlite3_file *id, const void *pBuf, int amt, i64 offset){ DO_OS_MALLOC_TEST(id); return id->pMethods->xWrite(id, pBuf, amt, offset); }
__host_device__ RC sqlite3OsTruncate(sqlite3_file *id, i64 size) { return id->pMethods->xTruncate(id, size); }
__host_device__ RC sqlite3OsSync(sqlite3_file *id, int flags) { DO_OS_MALLOC_TEST(id); return id->pMethods->xSync(id, flags); }
__host_device__ RC sqlite3OsFileSize(sqlite3_file *id, i64 *pSize) { DO_OS_MALLOC_TEST(id); return id->pMethods->xFileSize(id, pSize); }
__host_device__ RC sqlite3OsLock(sqlite3_file *id, int lockType) { DO_OS_MALLOC_TEST(id); return id->pMethods->xLock(id, lockType); }
__host_device__ RC sqlite3OsUnlock(sqlite3_file *id, int lockType) { return id->pMethods->xUnlock(id, lockType); }
__host_device__ RC sqlite3OsCheckReservedLock(sqlite3_file *id, int *pResOut) { DO_OS_MALLOC_TEST(id); return id->pMethods->xCheckReservedLock(id, pResOut); }

/*
** Use sqlite3OsFileControl() when we are doing something that might fail and we need to know about the failures.  Use sqlite3OsFileControlHint()
** when simply tossing information over the wall to the VFS and we do not really care if the VFS receives and understands the information since it
** is only a hint and can be safely ignored.  The sqlite3OsFileControlHint() routine has no return value since the return value would be meaningless.
*/
int sqlite3OsFileControl(sqlite3_file *id, int op, void *pArg)
{
#ifdef _TEST
	if (op != SQLITE_FCNTL_COMMIT_PHASETWO) {
		/* Faults are not injected into COMMIT_PHASETWO because, assuming SQLite is using a regular VFS, it is called after the corresponding
		** transaction has been committed. Injecting a fault at this point confuses the test scripts - the COMMIT comand returns SQLITE_NOMEM
		** but the transaction is committed anyway.
		**
		** The core must call OsFileControl() though, not OsFileControlHint(), as if a custom VFS (e.g. zipvfs) returns an error here, it probably
		** means the commit really has failed and an error should be returned to the user.  */
		DO_OS_MALLOC_TEST(id);
	}
#endif
	return id->pMethods->xFileControl(id, op, pArg);
}
void sqlite3OsFileControlHint(sqlite3_file *id, int op, void *pArg){ (void)id->pMethods->xFileControl(id, op, pArg); }

int sqlite3OsSectorSize(sqlite3_file *id){ int (*xSectorSize)(sqlite3_file*) = id->pMethods->xSectorSize; return (xSectorSize ? xSectorSize(id) : SQLITE_DEFAULT_SECTOR_SIZE); }
int sqlite3OsDeviceCharacteristics(sqlite3_file *id){ return id->pMethods->xDeviceCharacteristics(id); }
int sqlite3OsShmLock(sqlite3_file *id, int offset, int n, int flags){ return id->pMethods->xShmLock(id, offset, n, flags); }
void sqlite3OsShmBarrier(sqlite3_file *id){ id->pMethods->xShmBarrier(id); }
int sqlite3OsShmUnmap(sqlite3_file *id, int deleteFlag){ return id->pMethods->xShmUnmap(id, deleteFlag); }
int sqlite3OsShmMap(sqlite3_file *id, int iPage, int pgsz, int bExtend, void volatile **pp) { DO_OS_MALLOC_TEST(id); return id->pMethods->xShmMap(id, iPage, pgsz, bExtend, pp); }

#if SQLITE_MAX_MMAP_SIZE > 0
/* The real implementation of xFetch and xUnfetch */
int sqlite3OsFetch(sqlite3_file *id, i64 iOff, int iAmt, void **pp){ DO_OS_MALLOC_TEST(id); return id->pMethods->xFetch(id, iOff, iAmt, pp); }
int sqlite3OsUnfetch(sqlite3_file *id, i64 iOff, void *p){ return id->pMethods->xUnfetch(id, iOff, p); }
#else
/* No-op stubs to use when memory-mapped I/O is disabled */
int sqlite3OsFetch(sqlite3_file *id, i64 iOff, int iAmt, void **pp){ *pp = 0; return SQLITE_OK; }
int sqlite3OsUnfetch(sqlite3_file *id, i64 iOff, void *p){ return SQLITE_OK; }
#endif

/*
** The next group of routines are convenience wrappers around the VFS methods.
*/
int sqlite3OsOpen(sqlite3_vfs *pVfs, const char *zPath, sqlite3_file *pFile, int flags, int *pFlagsOut ){
	int rc;
	DO_OS_MALLOC_TEST(0);
	/* 0x87f7f is a mask of SQLITE_OPEN_ flags that are valid to be passed
	** down into the VFS layer.  Some SQLITE_OPEN_ flags (for example,
	** SQLITE_OPEN_FULLMUTEX or SQLITE_OPEN_SHAREDCACHE) are blocked before
	** reaching the VFS. */
	rc = pVfs->xOpen(pVfs, zPath, pFile, flags & 0x87f7f, pFlagsOut);
	assert( rc==SQLITE_OK || pFile->pMethods==0 );
	return rc;
}
int sqlite3OsDelete(sqlite3_vfs *pVfs, const char *zPath, int dirSync){
	DO_OS_MALLOC_TEST(0);
	assert( dirSync==0 || dirSync==1 );
	return pVfs->xDelete(pVfs, zPath, dirSync);
}
int sqlite3OsAccess(sqlite3_vfs *pVfs, const char *zPath, int flags, int *pResOut){ DO_OS_MALLOC_TEST(0); return pVfs->xAccess(pVfs, zPath, flags, pResOut); }
int sqlite3OsFullPathname(sqlite3_vfs *pVfs, const char *zPath, int nPathOut, char *zPathOut ){ DO_OS_MALLOC_TEST(0); zPathOut[0] = 0; return pVfs->xFullPathname(pVfs, zPath, nPathOut, zPathOut); }
#ifndef SQLITE_OMIT_LOAD_EXTENSION
void *sqlite3OsDlOpen(sqlite3_vfs *pVfs, const char *zPath){ return pVfs->xDlOpen(pVfs, zPath); }
void sqlite3OsDlError(sqlite3_vfs *pVfs, int nByte, char *zBufOut){ pVfs->xDlError(pVfs, nByte, zBufOut); }
void (*sqlite3OsDlSym(sqlite3_vfs *pVfs, void *pHdle, const char *zSym))(void){ return pVfs->xDlSym(pVfs, pHdle, zSym); }
void sqlite3OsDlClose(sqlite3_vfs *pVfs, void *pHandle){ pVfs->xDlClose(pVfs, pHandle); }
#endif
int sqlite3OsRandomness(sqlite3_vfs *pVfs, int nByte, char *zBufOut){ return pVfs->xRandomness(pVfs, nByte, zBufOut); }
int sqlite3OsSleep(sqlite3_vfs *pVfs, int nMicro){ return pVfs->xSleep(pVfs, nMicro); }
int sqlite3OsGetLastError(sqlite3_vfs *pVfs){ return pVfs->xGetLastError ? pVfs->xGetLastError(pVfs, 0, 0) : 0; }
int sqlite3OsCurrentTimeInt64(sqlite3_vfs *pVfs, sqlite3_int64 *pTimeOut){
	int rc;
	/* IMPLEMENTATION-OF: R-49045-42493 SQLite will use the xCurrentTimeInt64()
	** method to get the current date and time if that method is available
	** (if iVersion is 2 or greater and the function pointer is not NULL) and
	** will fall back to xCurrentTime() if xCurrentTimeInt64() is
	** unavailable.
	*/
	if( pVfs->iVersion>=2 && pVfs->xCurrentTimeInt64 ){
		rc = pVfs->xCurrentTimeInt64(pVfs, pTimeOut);
	}else{
		double r;
		rc = pVfs->xCurrentTime(pVfs, &r);
		*pTimeOut = (sqlite3_int64)(r*86400000.0);
	}
	return rc;
}

int sqlite3OsOpenMalloc(
	sqlite3_vfs *pVfs,
	const char *zFile,
	sqlite3_file **ppFile,
	int flags,
	int *pOutFlags
	){
		int rc;
		sqlite3_file *pFile;
		pFile = (sqlite3_file *)sqlite3MallocZero(pVfs->szOsFile);
		if( pFile ){
			rc = sqlite3OsOpen(pVfs, zFile, pFile, flags, pOutFlags);
			if( rc!=SQLITE_OK ){
				sqlite3_free(pFile);
			}else{
				*ppFile = pFile;
			}
		}else{
			rc = SQLITE_NOMEM_BKPT;
		}
		return rc;
}
void sqlite3OsCloseFree(sqlite3_file *pFile){
	assert( pFile );
	sqlite3OsClose(pFile);
	sqlite3_free(pFile);
}

/*
** This function is a wrapper around the OS specific implementation of
** sqlite3_os_init(). The purpose of the wrapper is to provide the
** ability to simulate a malloc failure, so that the handling of an
** error in sqlite3_os_init() by the upper layers can be tested.
*/
int sqlite3OsInit(void){
	void *p = sqlite3_malloc(10);
	if( p==0 ) return SQLITE_NOMEM_BKPT;
	sqlite3_free(p);
	return sqlite3_os_init();
}

/*
** The list of all registered VFS implementations.
*/
static sqlite3_vfs * SQLITE_WSD vfsList = 0;
#define vfsList GLOBAL(sqlite3_vfs *, vfsList)

/*
** Locate a VFS by name.  If no name is given, simply return the
** first VFS on the list.
*/
sqlite3_vfs *sqlite3_vfs_find(const char *zVfs){
	sqlite3_vfs *pVfs = 0;
#if SQLITE_THREADSAFE
	sqlite3_mutex *mutex;
#endif
#ifndef SQLITE_OMIT_AUTOINIT
	int rc = sqlite3_initialize();
	if( rc ) return 0;
#endif
#if SQLITE_THREADSAFE
	mutex = sqlite3MutexAlloc(SQLITE_MUTEX_STATIC_MASTER);
#endif
	sqlite3_mutex_enter(mutex);
	for(pVfs = vfsList; pVfs; pVfs=pVfs->pNext){
		if( zVfs==0 ) break;
		if( strcmp(zVfs, pVfs->zName)==0 ) break;
	}
	sqlite3_mutex_leave(mutex);
	return pVfs;
}

/*
** Unlink a VFS from the linked list
*/
static void vfsUnlink(sqlite3_vfs *pVfs){
	assert( sqlite3_mutex_held(sqlite3MutexAlloc(SQLITE_MUTEX_STATIC_MASTER)) );
	if( pVfs==0 ){
		/* No-op */
	}else if( vfsList==pVfs ){
		vfsList = pVfs->pNext;
	}else if( vfsList ){
		sqlite3_vfs *p = vfsList;
		while( p->pNext && p->pNext!=pVfs ){
			p = p->pNext;
		}
		if( p->pNext==pVfs ){
			p->pNext = pVfs->pNext;
		}
	}
}

/*
** Register a VFS with the system.  It is harmless to register the same
** VFS multiple times.  The new VFS becomes the default if makeDflt is
** true.
*/
int sqlite3_vfs_register(sqlite3_vfs *pVfs, int makeDflt){
	MUTEX_LOGIC(sqlite3_mutex *mutex;)
#ifndef SQLITE_OMIT_AUTOINIT
		int rc = sqlite3_initialize();
	if( rc ) return rc;
#endif
#ifdef SQLITE_ENABLE_API_ARMOR
	if( pVfs==0 ) return SQLITE_MISUSE_BKPT;
#endif

	MUTEX_LOGIC( mutex = sqlite3MutexAlloc(SQLITE_MUTEX_STATIC_MASTER); )
		sqlite3_mutex_enter(mutex);
	vfsUnlink(pVfs);
	if( makeDflt || vfsList==0 ){
		pVfs->pNext = vfsList;
		vfsList = pVfs;
	}else{
		pVfs->pNext = vfsList->pNext;
		vfsList->pNext = pVfs;
	}
	assert(vfsList);
	sqlite3_mutex_leave(mutex);
	return SQLITE_OK;
}

/*
** Unregister a VFS so that it is no longer accessible.
*/
int sqlite3_vfs_unregister(sqlite3_vfs *pVfs){
#if SQLITE_THREADSAFE
	sqlite3_mutex *mutex = sqlite3MutexAlloc(SQLITE_MUTEX_STATIC_MASTER);
#endif
	sqlite3_mutex_enter(mutex);
	vfsUnlink(pVfs);
	sqlite3_mutex_leave(mutex);
	return SQLITE_OK;
}





















































#ifdef _TEST
#define SimulateIOErrorBenign(X) g_io_error_benign=(X)
#define SimulateIOError(CODE) \
	if ((g_io_error_persist && g_io_error_hit) || g_io_error_pending-- == 1) { local_ioerr(); CODE; }
__device__ static void local_ioerr() { OSTRACE("IOERR\n"); g_io_error_hit++; if (!g_io_error_benign) g_io_error_hardhit++; }
#define SimulateDiskfullError(CODE) \
	if (g_diskfull_pending) { if (g_diskfull_pending == 1) { \
	local_ioerr(); g_diskfull = 1; g_io_error_hit = 1; CODE; \
	} else g_diskfull_pending--; }
#else
#define SimulateIOErrorBenign(X)
#define SimulateIOError(A)
#define SimulateDiskfullError(A)
#endif

#ifdef _TESTX
__device__ static int _saved_cnt;
__device__ void DisableSimulatedIOErrors(int *pending, int *hit) { if (!pending) pending = &_saved_cnt; *pending = g_io_error_pending; g_io_error_pending = -1; if (hit) { *hit = g_io_error_hit; g_io_error_hit = 0; } }
__device__ void EnableSimulatedIOErrors(int *pending, int *hit) { if (!pending) pending = &_saved_cnt; g_io_error_pending = *pending; if (hit) g_io_error_hit = *hit; }
#endif

#if OS_WIN // This file is used for Windows only
#define POWERSAFE_OVERWRITE 1

// When testing, keep a count of the number of open files.
#ifdef _TEST
__device__ int g_open_file_count = 0;
#define OpenCounter(X) g_open_file_count += (X)
#else
#define OpenCounter(X)
#endif

#if OS_WIN && !defined(OS_WINNT)
#define OS_WINNT 1
#endif
#if defined(_WIN32_WCE)
#define OS_WINCE 1
#else
#define OS_WINCE 0
#endif
#if !defined(OS_WINRT)
#define OS_WINRT 0
#endif

#pragma endregion


{
	__device__ VFile *VSystem::_AttachFile(void *buffer) { return nullptr; }
	__device__ RC VSystem::Open(const char *path, VFile *file, OPEN flags, OPEN *outFlags) { return RC_OK; }
	__device__ RC VSystem::Delete(const char *path, bool syncDirectory) { return RC_OK; }
	__device__ RC VSystem::Access(const char *path, ACCESS flags, int *outRC) { return RC_OK; }
	__device__ RC VSystem::FullPathname(const char *path, int pathOutLength, char *pathOut) { return RC_OK; }

	__device__ void *VSystem::DlOpen(const char *filename) { return nullptr; }
	__device__ void VSystem::DlError(int bufLength, char *buf) { }
	__device__ void (*VSystem::DlSym(void *handle, const char *symbol))() { return nullptr; }
	__device__ void VSystem::DlClose(void *handle) { }

	__device__ int VSystem::Randomness(int bufLength, char *buf) { return 0; }
	__device__ int VSystem::Sleep(int microseconds) { return 0; }
	__device__ RC VSystem::CurrentTimeInt64(int64 *now) { return RC_OK; }
	__device__ RC VSystem::CurrentTime(double *now) { return RC_OK; }
	__device__ RC VSystem::GetLastError(int bufLength, char *buf) { return RC_OK; }

	__device__ RC VSystem::SetSystemCall(const char *name, syscall_ptr newFunc) { return RC_OK; }
	__device__ syscall_ptr VSystem::GetSystemCall(const char *name) { return nullptr; }
	__device__ const char *VSystem::NextSystemCall(const char *name) { return nullptr; }


	// VfsList
#pragma region VfsList

	__device__ static VSystem *_WSD g_vfsList = nullptr;
#define _vfsList _GLOBAL(VSystem *, g_vfsList)

	__device__ VSystem *VSystem::FindVfs(const char *name)
	{
#ifndef OMIT_AUTOINIT
		RC rc = SysEx::AutoInitialize();
		if (rc) return nullptr;
#endif
		VSystem *vfs = nullptr;
		MUTEX_LOGIC(MutexEx mutex = )_mutex_alloc(MUTEX_STATIC_MASTER);
		_mutex_enter(mutex);
		for (vfs = _vfsList; vfs && name && _strcmp(name, vfs->Name); vfs = vfs->Next) { }
		_mutex_leave(mutex);
		return vfs;
	}

	__device__ static void UnlinkVfs(VSystem *vfs)
	{
		_assert(_mutex_held(_mutex_alloc(MUTEX_STATIC_MASTER)));
		if (!vfs) { }
		else if (_vfsList == vfs)
			_vfsList = vfs->Next;
		else if (_vfsList)
		{
			VSystem *p = _vfsList;
			while (p->Next && p->Next != vfs)
				p = p->Next;
			if (p->Next == vfs)
				p->Next = vfs->Next;
		}
	}

	__device__ RC VSystem::RegisterVfs(VSystem *vfs, bool default_)
	{
		MUTEX_LOGIC(MutexEx mutex = )_mutex_alloc(MUTEX_STATIC_MASTER);
		_mutex_enter(mutex);
		UnlinkVfs(vfs);
		if (default_ || !_vfsList)
		{
			vfs->Next = _vfsList;
			_vfsList = vfs;
		}
		else
		{
			vfs->Next = _vfsList->Next;
			_vfsList->Next = vfs;
		}
		_assert(_vfsList != nullptr);
		_mutex_leave(mutex);
		return RC_OK;
	}

	__device__ RC VSystem::UnregisterVfs(VSystem *vfs)
	{
		MUTEX_LOGIC(MutexEx mutex = )_mutex_alloc(MUTEX_STATIC_MASTER);
		_mutex_enter(mutex);
		UnlinkVfs(vfs);
		_mutex_leave(mutex);
		return RC_OK;
	}

#pragma endregion

	// from main_c
#pragma region File

#ifdef ENABLE_8_3_NAMES
	__device__ void SysEx::FileSuffix3(const char *baseFilename, char *z)
	{
#if ENABLE_8_3_NAMES<2
		if (!UriBoolean(baseFilename, "8_3_names", 0)) return;
#endif
		int size = _strlen(z);
		int i;
		for (i = size-1; i > 0 && z[i] != '/' && z[i] !='.'; i--) { }
		if (z[i] == '.' && _ALWAYS(size > i+4)) _memmove(&z[i+1], &z[size-3], 4);
	}
#endif

	struct OpenMode
	{
		const char *Z;
		VSystem::OPEN Mode;
	};

	__constant__ static OpenMode _cacheModes[] =
	{
		{ "shared",  VSystem::OPEN_SHAREDCACHE },
		{ "private", VSystem::OPEN_PRIVATECACHE },
		{ nullptr, (VSystem::OPEN)0 }
	};

	__constant__ static OpenMode _openModes[] =
	{
		{ "ro",  VSystem::OPEN_READONLY },
		{ "rw",  VSystem::OPEN_READWRITE }, 
		{ "rwc", (VSystem::OPEN)((int)VSystem::OPEN_READWRITE | (int)VSystem::OPEN_CREATE) },
		{ "memory", VSystem::OPEN_MEMORY },
		{ nullptr, (VSystem::OPEN)0 }
	};

	__device__ RC VSystem::ParseUri(const char *defaultVfsName, const char *uri, VSystem::OPEN *flagsRef, VSystem **vfsOut, char **fileNameOut, char **errMsgOut)
	{
		_assert(*errMsgOut == nullptr);

		VSystem::OPEN flags = *flagsRef;
		const char *vfsName = defaultVfsName;
		int uriLength = _strlen(uri);

		RC rc = RC_OK;
		char *fileName;
		if (((flags & VSystem::OPEN_URI) || SysEx_GlobalStatics.OpenUri) && uriLength >= 5 && !_memcmp(uri, "file:", 5))
		{
			// Make sure the SQLITE_OPEN_URI flag is set to indicate to the VFS xOpen method that there may be extra parameters following the file-name.
			flags |= VSystem::OPEN_URI;

			int bytes = uriLength+2; // Bytes of space to allocate
			int uriIdx; // Input character index
			for (uriIdx = 0; uriIdx < uriLength; uriIdx++) bytes += (uri[uriIdx] == '&');
			fileName = (char *)_alloc(bytes);
			if (!fileName) return RC_NOMEM;

			// Discard the scheme and authority segments of the URI.
			if (uri[5] == '/' && uri[6] == '/')
			{
				uriIdx = 7;
				while (uri[uriIdx] && uri[uriIdx] != '/') uriIdx++;
				if (uriIdx != 7 && (uriIdx != 16 || _memcmp("localhost", &uri[7], 9)))
				{
					*errMsgOut = _mprintf("invalid uri authority: %.*s", uriIdx-7, &uri[7]);
					rc = RC_ERROR;
					goto parse_uri_out;
				}
			}
			else
				uriIdx = 5;

			// Copy the filename and any query parameters into the zFile buffer. Decode %HH escape codes along the way. 
			//
			// Within this loop, variable eState may be set to 0, 1 or 2, depending on the parsing context. As follows:
			//
			//   0: Parsing file-name.
			//   1: Parsing name section of a name=value query parameter.
			//   2: Parsing value section of a name=value query parameter.
			int state = 0; // Parser state when parsing URI
			char c;
			int fileNameIdx = 0; // Output character index
			while ((c = uri[uriIdx]) != 0 && c != '#')
			{
				uriIdx++;
				if (c == '%' && _isxdigit(uri[uriIdx]) && _isxdigit(uri[uriIdx+1]))
				{
					int octet = (_hextobyte(uri[uriIdx++]) << 4);
					octet += _hextobyte(uri[uriIdx++]);
					_assert(octet >= 0 && octet < 256);
					if (octet == 0)
					{
						// This branch is taken when "%00" appears within the URI. In this case we ignore all text in the remainder of the path, name or
						// value currently being parsed. So ignore the current character and skip to the next "?", "=" or "&", as appropriate.
						while ((c = uri[uriIdx]) != 0 && c !='#' && 
							(state != 0 || c != '?') && 
							(state != 1 || (c != '=' && c != '&')) && 
							(state != 2 || c != '&'))
							uriIdx++;
						continue;
					}
					c = octet;
				}
				else if (state == 1 && (c == '&' || c == '='))
				{
					if (fileName[fileNameIdx-1] == 0)
					{
						// An empty option name. Ignore this option altogether.
						while (uri[uriIdx] && uri[uriIdx] != '#' && uri[uriIdx-1] != '&') uriIdx++;
						continue;
					}
					if (c == '&')
						fileName[fileNameIdx++] = '\0';
					else
						state = 2;
					c = 0;
				}
				else if ((state == 0 && c == '?') || (state == 2 && c == '&'))
				{
					c = 0;
					state = 1;
				}
				fileName[fileNameIdx++] = c;
			}
			if (state == 1) fileName[fileNameIdx++] = '\0';
			fileName[fileNameIdx++] = '\0';
			fileName[fileNameIdx++] = '\0';

			// Check if there were any options specified that should be interpreted here. Options that are interpreted here include "vfs" and those that
			// correspond to flags that may be passed to the sqlite3_open_v2() method.
			char *opt = &fileName[_strlen(fileName)+1];
			while (opt[0])
			{
				int optLength = _strlen(opt);
				char *val = &opt[optLength+1];
				int valLength = _strlen(val);
				if (optLength == 3 && !_memcmp("vfs", opt, 3))
					vfsName = val;
				else
				{
					OpenMode *modes = nullptr;
					char *modeType = nullptr;
					VSystem::OPEN mask = (VSystem::OPEN)0;
					VSystem::OPEN limit = (VSystem::OPEN)0;
					if (optLength == 5 && !_memcmp("cache", opt, 5))
					{
						mask = (VSystem::OPEN)(VSystem::OPEN_SHAREDCACHE|VSystem::OPEN_PRIVATECACHE);
						modes = _cacheModes;
						limit = mask;
						modeType = "cache";
					}
					if (optLength == 4 && !_memcmp("mode", opt, 4))
					{
						mask = (VSystem::OPEN)(VSystem::OPEN_READONLY|VSystem::OPEN_READWRITE|VSystem::OPEN_CREATE|VSystem::OPEN_MEMORY);
						modes = _openModes;
						limit = (VSystem::OPEN)(mask & flags);
						modeType = "access";
					}
					if (modes)
					{
						VSystem::OPEN mode = (VSystem::OPEN)0;
						for (int i = 0; modes[i].Z; i++)
						{
							const char *z = modes[i].Z;
							if (valLength == _strlen(z) && !_memcmp(val, z, valLength))
							{
								mode = modes[i].Mode;
								break;
							}
						}
						if (mode == 0)
						{
							*errMsgOut = _mprintf("no such %s mode: %s", modeType, val);
							rc = RC_ERROR;
							goto parse_uri_out;
						}
						if ((mode & ~VSystem::OPEN_MEMORY) > limit)
						{
							*errMsgOut = _mprintf("%s mode not allowed: %s", modeType, val);
							rc = RC_PERM;
							goto parse_uri_out;
						}
						flags = (VSystem::OPEN)((flags & ~mask) | mode);
					}
				}
				opt = &val[valLength+1];
			}
		}
		else
		{
			fileName = (char *)_alloc(uriLength+2);
			if (!fileName) return RC_NOMEM;
			_memcpy(fileName, uri, uriLength);
			fileName[uriLength] = '\0';
			fileName[uriLength+1] = '\0';
			flags &= ~VSystem::OPEN_URI;
		}

		*vfsOut = FindVfs(vfsName);
		if (!*vfsOut)
		{
			*errMsgOut = _mprintf("no such vfs: %s", vfsName);
			rc = RC_ERROR;
		}

parse_uri_out:
		if (rc != RC_OK)
		{
			_free(fileName);
			fileName = nullptr;
		}
		*flagsRef = flags;
		*fileNameOut = fileName;
		return rc;
	}

	__device__ const char *VSystem::UriParameter(const char *filename, const char *param)
	{
		if (!filename) return nullptr;
		filename += _strlen(filename) + 1;
		while (filename[0])
		{
			int x = _strcmp(filename, param);
			filename += _strlen(filename) + 1;
			if (x == 0) return filename;
			filename += _strlen(filename) + 1;
		}
		return nullptr;
	}

	__device__ bool VSystem::UriBoolean(const char *filename, const char *param, bool dflt)
	{
		const char *z = UriParameter(filename, param);
		return (z ? __atob(z, dflt) : dflt);
	}

	__device__ int64 VSystem::UriInt64(const char *filename, const char *param, int64 dflt)
	{
		const char *z = UriParameter(filename, param);
		int64 v;
		return (z && __atoi64(z, &v, _strlen(z), TEXTENCODE_UTF8) == RC_OK ? v : dflt);
	}
