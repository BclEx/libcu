#include <stringcu.h> //: os.c
#include <ext/global.h>
#include <assert.h>

//#if defined(_TEST) || defined(_DEBUG)
//bool OsTrace = false;
//#define OSTRACE(X, ...) if (OsTrace) { _dprintf("OS: "X, __VA_ARGS__); }
//#else
//#define OSTRACE(X, ...)
//#endif

/* If we compile with the SQLITE_TEST macro set, then the following block of code will give us the ability to simulate a disk I/O error.  This
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

/* The default SQLite sqlite3_vfs implementations do not allocate memory (actually, os_unix.c allocates a small amount of memory
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
	mfree(tstAlloc); }
#else
#define DO_OS_MALLOC_TEST(x)
#endif

/* The following routines are convenience wrappers around methods of the vsysfile object.  This is mostly just syntactic sugar. All
** of this would be completely automatic if SQLite were coded using C++ instead of plain old C.
*/
__host_device__ void vsys_close(vsysfile *p) { if (p->methods) { p->methods->close(p); p->methods = nullptr; } } //: sqlite3OsClose
__host_device__ RC vsys_read(vsysfile *p, void *buf, int amount, int64_t offset) { DO_OS_MALLOC_TEST(p); return p->methods->read(p, buf, amount, offset); } //: sqlite3OsRead
__host_device__ RC vsys_write(vsysfile *p, const void *buf, int amount, int64_t offset){ DO_OS_MALLOC_TEST(p); return p->methods->write(p, buf, amount, offset); } //: sqlite3OsWrite
__host_device__ RC vsys_truncate(vsysfile *p, int64_t size) { return p->methods->truncate(p, size); } //: sqlite3OsTruncate
__host_device__ RC vsys_sync(vsysfile *p, int flags) { DO_OS_MALLOC_TEST(id); return flags ? p->methods->sync(p, flags) : RC_OK; } //: sqlite3OsSync
__host_device__ RC vsys_fileSize(vsysfile *p, int64_t *size) { DO_OS_MALLOC_TEST(p); return p->methods->fileSize(p, size); } //: sqlite3OsFileSize
__host_device__ RC vsys_lock(vsysfile *p, int lockType) { DO_OS_MALLOC_TEST(p); return p->methods->lock(p, lockType); } //: sqlite3OsLock
__host_device__ RC vsys_unlock(vsysfile *p, int lockType) { return p->methods->unlock(p, lockType); } //: sqlite3OsUnlock
__host_device__ RC vsys_checkReservedLock(vsysfile *p, int *res) { DO_OS_MALLOC_TEST(id); return p->methods->checkReservedLock(p, res); } //: sqlite3OsCheckReservedLock

/* Use sqlite3OsFileControl() when we are doing something that might fail and we need to know about the failures.  Use sqlite3OsFileControlHint()
** when simply tossing information over the wall to the VFS and we do not really care if the VFS receives and understands the information since it
** is only a hint and can be safely ignored.  The sqlite3OsFileControlHint() routine has no return value since the return value would be meaningless.
*/
__host_device__ RC vsys_fileControl(vsysfile *p, int op, void *arg) //: sqlite3OsFileControl
{
#ifdef _TEST
	if (op != VSYS_FCNTL_COMMIT_PHASETWO) {
		// Faults are not injected into COMMIT_PHASETWO because, assuming SQLite is using a regular VFS, it is called after the corresponding
		// transaction has been committed. Injecting a fault at this point confuses the test scripts - the COMMIT comand returns SQLITE_NOMEM
		// but the transaction is committed anyway.
		//
		// The core must call OsFileControl() though, not OsFileControlHint(), as if a custom VFS (e.g. zipvfs) returns an error here, it probably
		// means the commit really has failed and an error should be returned to the user.  */
		DO_OS_MALLOC_TEST(id);
	}
#endif
	return p->methods->fileControl(p, op, arg);
}
__host_device__ void vsys_fileControlHint(vsysfile *p, int op, void *arg) { p->methods->fileControl(p, op, arg); } //: sqlite3OsFileControlHint
__host_device__ int vsys_sectorSize(vsysfile *p) { int (*sectorSize)(vsysfile *) = p->methods->sectorSize; return sectorSize ? sectorSize(p) : LIBCU_DEFAULT_SECTOR_SIZE; } //: sqlite3OsSectorSize
__host_device__ int vsys_deviceCharacteristics(vsysfile *p) { return p->methods->deviceCharacteristics(p); } //: sqlite3OsDeviceCharacteristics
#ifndef OMIT_WAL
__host_device__ RC vsys_shmLock(vsysfile *p, int offset, int n, int flags){ return p->methods->shmLock(p, offset, n, flags); } //: sqlite3OsShmLock
__host_device__ void vsys_shmBarrier(vsysfile *p) { p->methods->shmBarrier(p); } //: sqlite3OsShmBarrier
__host_device__ RC vsys_shmUnmap(vsysfile *p, int deleteFlag) { return p->methods->shmUnmap(p, deleteFlag); } //: sqlite3OsShmUnmap
__host_device__ RC vsys_shmMap(vsysfile *p, int pageId, int pageSize, int extend, void volatile **pages) { DO_OS_MALLOC_TEST(p); return p->methods->shmMap(p, pageId, pageSize, extend, pages); } //: sqlite3OsShmMap
#endif

#if LIBCU_MAX_MMAP_SIZE > 0
/* The real implementation of xFetch and xUnfetch */
__host_device__ RC vsys_fetch(vsysfile *p, int64_t offset, int amount, void **pages) { DO_OS_MALLOC_TEST(p); return p->methods->fetch(p, offset, amount, pages); } //: sqlite3OsFetch
__host_device__ RC vsys_unfetch(vsysfile *p, int64_t offset, void *pages) { return p->methods->unfetch(p, offset, pages); } //: sqlite3OsUnfetch
#else
/* No-op stubs to use when memory-mapped I/O is disabled */
__host_device__ RC vsys_fetch(vsysfile *p, int64_t offset, int amount, void **pages){ *pages = nullptr; return RC_OK; } //: sqlite3OsFetch
__host_device__ RC vsys_unfetch(vsysfile *p, int64_t offset, void *pages) { return RC_OK; } //: sqlite3OsUnfetch
#endif

/* The next group of routines are convenience wrappers around the VFS methods. */
__host_device__ RC vsys_open(vsystem *p, const char *path, vsysfile *file, int flags, int *flagsOut) { DO_OS_MALLOC_TEST(nullptr); RC rc = p->open(p, path, file, flags & 0x87f7f, flagsOut); assert(rc == RC_OK || !file->methods); return rc; } //: sqlite3OsOpen
/* 0x87f7f is a mask of SQLITE_OPEN_ flags that are valid to be passed down into the VFS layer.  Some SQLITE_OPEN_ flags (for example,
** SQLITE_OPEN_FULLMUTEX or SQLITE_OPEN_SHAREDCACHE) are blocked before reaching the VFS. */
__host_device__ RC vsys_delete(vsystem *p, const char *path, int dirSync) { DO_OS_MALLOC_TEST(nullptr); assert(dirSync == 0 || dirSync == 1); return p->delete_(p, path, dirSync); } //: sqlite3OsDelete
__host_device__ RC vsys_access(vsystem *p, const char *path, int flags, int *resOut) { DO_OS_MALLOC_TEST(nullptr); return p->access(p, path, flags, resOut); } //: sqlite3OsAccess
__host_device__ RC vsys_fullPathname(vsystem *p, const char *path, int pathOutSize, char *pathOut) { DO_OS_MALLOC_TEST(0); pathOut[0] = 0; return p->fullPathname(p, path, pathOutSize, pathOut); } //: sqlite3OsFullPathname
#ifndef NO_LOAD_EXTENSION
__host_device__ void *vsys_dlOpen(vsystem *p, const char *path) { return p->dlOpen(p, path); } //: sqlite3OsDlOpen
__host_device__ void vsys_dlError(vsystem *p, int bytes, char *bufOut) { p->dlError(p, bytes, bufOut); } //: sqlite3OsDlError
__host_device__ void (*vsys_dlSym(vsystem *p, void *handle, const char *sym))() { return p->dlSym(p, handle, sym); } //: sqlite3OsDlSym
__host_device__ void vsys_dlClose(vsystem *p, void *handle) { p->dlClose(p, handle); } //: sqlite3OsDlClose
#endif
__host_device__ RC vsys_randomness(vsystem *p, int bytes, char *bufOut) { return p->randomness(p, bytes, bufOut); } //: sqlite3OsRandomness
__host_device__ RC vsys_sleep(vsystem *p, int microseconds) { return p->sleep(p, microseconds); } //: sqlite3OsSleep
__host_device__ int vsys_getLastError(vsystem *p) { return p->getLastError ? p->getLastError(p, 0, nullptr) : 0; } //: sqlite3OsGetLastError
__host_device__ RC vsys_currentTimeInt64(vsystem *p, int64_t *timeOut) { if (p->version >= 2 && p->currentTimeInt64) return p->currentTimeInt64(p, timeOut); double r; RC rc = p->currentTime(p, &r); *timeOut = (int64_t)(r*86400000.0); return rc; } //: sqlite3OsCurrentTimeInt64
/* IMPLEMENTATION-OF: R-49045-42493 SQLite will use the xCurrentTimeInt64() method to get the current date and time if that method is available
** (if iVersion is 2 or greater and the function pointer is not NULL) and will fall back to xCurrentTime() if xCurrentTimeInt64() is unavailable.
*/
__host_device__ RC vsys_openMalloc(vsystem *p, const char *fileName, vsysfile **file, int flags, int *flagsOut) //: sqlite3OsOpenMalloc
{
	vsysfile *newFile = (vsysfile *)allocZero(p->sizeOsFile);
	if (newFile) {
		RC rc = vsys_open(p, fileName, newFile, flags, flagsOut);
		if (rc != RC_OK) mfree(newFile);
		else *file = newFile;
		return rc;
	}
	return RC_NOMEM_BKPT;
}
__host_device__ void vsys_closeAndFree(vsysfile *p) { assert(p); vsys_close(p); mfree(p); } // : sqlite3OsCloseFree

/* This function is a wrapper around the OS specific implementation of sqlite3_os_init(). The purpose of the wrapper is to provide the
** ability to simulate a malloc failure, so that the handling of an error in sqlite3_os_init() by the upper layers can be tested.
*/
__host_device__ RC vsystemFakeInit() //: sqlite3OsInit
{
	//void *p = alloc(10);
	//if (!p) return RC_NOMEM_BKPT;
	//mfree(p);
	//return sqlite3_os_init();
	return 0;
}

/* The list of all registered VFS implementations. */
static __hostb_device__ vsystem * WSD_ _vfsGlobal = nullptr;
#define vfs GLOBAL_(vsystem *, _vfsGlobal)

/* Locate a VFS by name.  If no name is given, simply return the first VFS on the list. */
__host_device__ vsystem *vsystemFind(const char *name) //: sqlite3_vfs_find
{
#ifndef OMIT_AUTOINIT
	RC rc = runtimeInitialize();
	if (rc) return nullptr;
#endif
#if LIBCU_THREADSAFE
	mutex *mutex = mutex_alloc(MUTEX_STATIC_MASTER);
#endif
	mutex_enter(mutex);
	vsystem *p = nullptr;
	for (p = vfs; p; p = p->next){
		if (!p) break;
		if (!strcmp(name, p->name)) break;
	}
	mutex_leave(mutex);
	return p;
}

/* Unlink a VFS from the linked list */
static __host_device__ void vsystemUnlink(vsystem *p)
{
	assert(mutex_held(mutexAlloc(MUTEX_STATIC_MASTER)));
	if (!p) { } /* No-op */
	else if (vfs == p) vfs = p->next;
	else if (vfs) {
		vsystem *p2 = vfs;
		while (p2->next && p2->next != p) p2 = p2->next;
		if (p2->next == p) p2->next = p->next;
	}
}

/* Register a VFS with the system.  It is harmless to register the same VFS multiple times.  The new VFS becomes the default if makeDflt is true. */
__host_device__ RC vsystemRegister(vsystem *p, bool makeDefault) //: sqlite3_vfs_register
{
#ifndef OMIT_AUTOINIT
	RC rc = runtimeInitialize();
	if (rc) return rc;
#endif
#ifdef LIBCU_ENABLE_API_ARMOR
	if (!p) return RC_MISUSE_BKPT;
#endif
#if LIBCU_THREADSAFE
	mutex *mutex = mutex_alloc(MUTEX_STATIC_MASTER); 
#endif
	mutex_enter(mutex);
	vsystemUnlink(p);
	if (makeDefault || !vfs) { p->next = vfs; vfs = p; }
	else { p->next = vfs->next; vfs->next = p; }
	assert(vfs);
	mutex_leave(mutex);
	return RC_OK;
}

/* Unregister a VFS so that it is no longer accessible. */
__host_device__ RC vsystemUnregister(vsystem *p) //: sqlite3_vfs_unregister
{
#if LIBCU_THREADSAFE
	mutex *mutex = mutex_alloc(MUTEX_STATIC_MASTER);
#endif
	mutex_enter(mutex);
	vsystemUnlink(p);
	mutex_leave(mutex);
	return RC_OK;
}


//
//// from main_c
//#pragma region File
//
//#ifdef ENABLE_8_3_NAMES
//__device__ void SysEx::FileSuffix3(const char *baseFilename, char *z)
//{
//#if ENABLE_8_3_NAMES<2
//	if (!UriBoolean(baseFilename, "8_3_names", 0)) return;
//#endif
//	int size = _strlen(z);
//	int i;
//	for (i = size-1; i > 0 && z[i] != '/' && z[i] !='.'; i--) { }
//	if (z[i] == '.' && ALWAYS_(size > i+4)) _memmove(&z[i+1], &z[size-3], 4);
//}
//#endif
//
//struct OpenMode
//{
//	const char *Z;
//	VSystem::OPEN Mode;
//};
//
//__constant__ static OpenMode _cacheModes[] =
//{
//	{ "shared",  VSystem::OPEN_SHAREDCACHE },
//	{ "private", VSystem::OPEN_PRIVATECACHE },
//	{ nullptr, (VSystem::OPEN)0 }
//};
//
//__constant__ static OpenMode _openModes[] =
//{
//	{ "ro",  VSystem::OPEN_READONLY },
//	{ "rw",  VSystem::OPEN_READWRITE }, 
//	{ "rwc", (VSystem::OPEN)((int)VSystem::OPEN_READWRITE | (int)VSystem::OPEN_CREATE) },
//	{ "memory", VSystem::OPEN_MEMORY },
//	{ nullptr, (VSystem::OPEN)0 }
//};
//
//__device__ RC VSystem::ParseUri(const char *defaultVfsName, const char *uri, VSystem::OPEN *flagsRef, VSystem **vfsOut, char **fileNameOut, char **errMsgOut)
//{
//	_assert(*errMsgOut == nullptr);
//
//	VSystem::OPEN flags = *flagsRef;
//	const char *vfsName = defaultVfsName;
//	int uriLength = _strlen(uri);
//
//	RC rc = RC_OK;
//	char *fileName;
//	if (((flags & VSystem::OPEN_URI) || SysExGLOBAL_Statics.OpenUri) && uriLength >= 5 && !_memcmp(uri, "file:", 5))
//	{
//		// Make sure the SQLITE_OPEN_URI flag is set to indicate to the VFS xOpen method that there may be extra parameters following the file-name.
//		flags |= VSystem::OPEN_URI;
//
//		int bytes = uriLength+2; // Bytes of space to allocate
//		int uriIdx; // Input character index
//		for (uriIdx = 0; uriIdx < uriLength; uriIdx++) bytes += (uri[uriIdx] == '&');
//		fileName = (char *)_alloc(bytes);
//		if (!fileName) return RC_NOMEM;
//
//		// Discard the scheme and authority segments of the URI.
//		if (uri[5] == '/' && uri[6] == '/')
//		{
//			uriIdx = 7;
//			while (uri[uriIdx] && uri[uriIdx] != '/') uriIdx++;
//			if (uriIdx != 7 && (uriIdx != 16 || _memcmp("localhost", &uri[7], 9)))
//			{
//				*errMsgOut = _mprintf("invalid uri authority: %.*s", uriIdx-7, &uri[7]);
//				rc = RC_ERROR;
//				goto parse_uri_out;
//			}
//		}
//		else
//			uriIdx = 5;
//
//		// Copy the filename and any query parameters into the zFile buffer. Decode %HH escape codes along the way. 
//		//
//		// Within this loop, variable eState may be set to 0, 1 or 2, depending on the parsing context. As follows:
//		//
//		//   0: Parsing file-name.
//		//   1: Parsing name section of a name=value query parameter.
//		//   2: Parsing value section of a name=value query parameter.
//		int state = 0; // Parser state when parsing URI
//		char c;
//		int fileNameIdx = 0; // Output character index
//		while ((c = uri[uriIdx]) != 0 && c != '#')
//		{
//			uriIdx++;
//			if (c == '%' && _isxdigit(uri[uriIdx]) && _isxdigit(uri[uriIdx+1]))
//			{
//				int octet = (_hextobyte(uri[uriIdx++]) << 4);
//				octet += _hextobyte(uri[uriIdx++]);
//				_assert(octet >= 0 && octet < 256);
//				if (octet == 0)
//				{
//					// This branch is taken when "%00" appears within the URI. In this case we ignore all text in the remainder of the path, name or
//					// value currently being parsed. So ignore the current character and skip to the next "?", "=" or "&", as appropriate.
//					while ((c = uri[uriIdx]) != 0 && c !='#' && 
//						(state != 0 || c != '?') && 
//						(state != 1 || (c != '=' && c != '&')) && 
//						(state != 2 || c != '&'))
//						uriIdx++;
//					continue;
//				}
//				c = octet;
//			}
//			else if (state == 1 && (c == '&' || c == '='))
//			{
//				if (fileName[fileNameIdx-1] == 0)
//				{
//					// An empty option name. Ignore this option altogether.
//					while (uri[uriIdx] && uri[uriIdx] != '#' && uri[uriIdx-1] != '&') uriIdx++;
//					continue;
//				}
//				if (c == '&')
//					fileName[fileNameIdx++] = '\0';
//				else
//					state = 2;
//				c = 0;
//			}
//			else if ((state == 0 && c == '?') || (state == 2 && c == '&'))
//			{
//				c = 0;
//				state = 1;
//			}
//			fileName[fileNameIdx++] = c;
//		}
//		if (state == 1) fileName[fileNameIdx++] = '\0';
//		fileName[fileNameIdx++] = '\0';
//		fileName[fileNameIdx++] = '\0';
//
//		// Check if there were any options specified that should be interpreted here. Options that are interpreted here include "vfs" and those that
//		// correspond to flags that may be passed to the sqlite3_open_v2() method.
//		char *opt = &fileName[_strlen(fileName)+1];
//		while (opt[0])
//		{
//			int optLength = _strlen(opt);
//			char *val = &opt[optLength+1];
//			int valLength = _strlen(val);
//			if (optLength == 3 && !_memcmp("vfs", opt, 3))
//				vfsName = val;
//			else
//			{
//				OpenMode *modes = nullptr;
//				char *modeType = nullptr;
//				VSystem::OPEN mask = (VSystem::OPEN)0;
//				VSystem::OPEN limit = (VSystem::OPEN)0;
//				if (optLength == 5 && !_memcmp("cache", opt, 5))
//				{
//					mask = (VSystem::OPEN)(VSystem::OPEN_SHAREDCACHE|VSystem::OPEN_PRIVATECACHE);
//					modes = _cacheModes;
//					limit = mask;
//					modeType = "cache";
//				}
//				if (optLength == 4 && !_memcmp("mode", opt, 4))
//				{
//					mask = (VSystem::OPEN)(VSystem::OPEN_READONLY|VSystem::OPEN_READWRITE|VSystem::OPEN_CREATE|VSystem::OPEN_MEMORY);
//					modes = _openModes;
//					limit = (VSystem::OPEN)(mask & flags);
//					modeType = "access";
//				}
//				if (modes)
//				{
//					VSystem::OPEN mode = (VSystem::OPEN)0;
//					for (int i = 0; modes[i].Z; i++)
//					{
//						const char *z = modes[i].Z;
//						if (valLength == _strlen(z) && !_memcmp(val, z, valLength))
//						{
//							mode = modes[i].Mode;
//							break;
//						}
//					}
//					if (mode == 0)
//					{
//						*errMsgOut = _mprintf("no such %s mode: %s", modeType, val);
//						rc = RC_ERROR;
//						goto parse_uri_out;
//					}
//					if ((mode & ~VSystem::OPEN_MEMORY) > limit)
//					{
//						*errMsgOut = _mprintf("%s mode not allowed: %s", modeType, val);
//						rc = RC_PERM;
//						goto parse_uri_out;
//					}
//					flags = (VSystem::OPEN)((flags & ~mask) | mode);
//				}
//			}
//			opt = &val[valLength+1];
//		}
//	}
//	else
//	{
//		fileName = (char *)_alloc(uriLength+2);
//		if (!fileName) return RC_NOMEM;
//		_memcpy(fileName, uri, uriLength);
//		fileName[uriLength] = '\0';
//		fileName[uriLength+1] = '\0';
//		flags &= ~VSystem::OPEN_URI;
//	}
//
//	*vfsOut = FindVfs(vfsName);
//	if (!*vfsOut)
//	{
//		*errMsgOut = _mprintf("no such vfs: %s", vfsName);
//		rc = RC_ERROR;
//	}
//
//parse_uri_out:
//	if (rc != RC_OK)
//	{
//		_free(fileName);
//		fileName = nullptr;
//	}
//	*flagsRef = flags;
//	*fileNameOut = fileName;
//	return rc;
//}
//
//__device__ const char *VSystem::UriParameter(const char *filename, const char *param)
//{
//	if (!filename) return nullptr;
//	filename += _strlen(filename) + 1;
//	while (filename[0])
//	{
//		int x = _strcmp(filename, param);
//		filename += _strlen(filename) + 1;
//		if (x == 0) return filename;
//		filename += _strlen(filename) + 1;
//	}
//	return nullptr;
//}
//
//__device__ bool VSystem::UriBoolean(const char *filename, const char *param, bool dflt)
//{
//	const char *z = UriParameter(filename, param);
//	return (z ? __atob(z, dflt) : dflt);
//}
//
//__device__ int64 VSystem::UriInt64(const char *filename, const char *param, int64 dflt)
//{
//	const char *z = UriParameter(filename, param);
//	int64 v;
//	return (z && __atoi64(z, &v, _strlen(z), TEXTENCODE_UTF8) == RC_OK ? v : dflt);
//}
