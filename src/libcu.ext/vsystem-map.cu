#include <ext/global.h> //: new.c
#include <stdiocu.h>
#include <assert.h>
#include "sentinel-vsystem.h"

/* The mapFile structure is a subclass of sqlite3_file * specific to the map portability layer. */
typedef struct mapsysfile mapsysfile;
struct mapsysfile {
	const vsysfile_methods *methods; // Must be first
	vsystem *vsys;	// The VFS used to open this file
	vsysfile *f;	// The native VFile
};

__device__ int mapClose(vsysfile *f)
{
	mapsysfile *file = (mapsysfile *)f;
	vsysfile_close msg(f);
	file->methods = nullptr;
	OSTRACE(("CLOSE %d %s\n", f, !msg.RC ? "ok" : "failed"));
	return msg.RC;
}

__device__ int mapRead(vsysfile *f, void *buf, int amount, int64_t offset)
{
	OSTRACE(("READ %d - ", f));
	vsysfile_read msg(f, buf, amount, offset);
	OSTRACE(("%s\n", !msg.RC ? "ok" : "failed"));
	return msg.RC;
}

__device__ int mapWrite(vsysfile *f, const void *buf, int amount, int64_t offset)
{
	OSTRACE(("WRITE %d - ", f));
	vsysfile_write msg(f, buf, amount, offset);
	OSTRACE(("%s\n", !msg.RC ? "ok" : "failed"));
	return msg.RC;
}

__device__ int mapTruncate(vsysfile *f, int64_t size)
{
	vsysfile_truncate msg(f, size);
	OSTRACE(("TRUC %d %s\n", f, !msg.RC ? "ok" : "failed"));
	return msg.RC;
}

__device__ int mapSync(vsysfile *f, int flags)
{
	vsysfile_sync msg(f, flags);
	OSTRACE(("SYNC %d %s\n", f, !msg.RC ? "ok" : "failed"));
	return msg.RC;
}

__device__ int mapFileSize(vsysfile *f, int64_t *size)
{
	vsysfile_fileSize msg(f);
	*size = msg.Size;
	OSTRACE(("FILESIZE %d %s\n", f, !msg.RC ? "ok" : "failed"));
	return msg.RC;
}

__device__ int mapLock(vsysfile *f, int lock)
{
	vsysfile_lock msg(f, lock);
	OSTRACE(("LOCK %d %s\n", f, !msg.RC ? "ok" : "failed"));
	return msg.RC;
}

__device__ int mapUnlock(vsysfile *f, int lock)
{
	vsysfile_unlock msg(f, lock);
	OSTRACE(("UNLOCK %d %s\n", f, !msg.RC ? "ok" : "failed"));
	return msg.RC;
}

__device__ int mapCheckReservedLock(vsysfile *f, int *lock)
{
	vsysfile_checkReservedLock msg(f);
	*lock = msg.Lock;
	OSTRACE(("CHECKRESLK %d %s\n", f, !msg.RC ? "ok" : "failed"));
	return msg.RC;
}

__device__ int mapFileControl(vsysfile *f, int op, void *arg)
{
	return RC_NOTFOUND;
}

__device__ int mapSectorSize(vsysfile *f)
{
	return 512;
}

__device__ int mapDeviceCharacteristics(vsysfile *f)
{
	return 0;
}

__device__ int mapShmMap(vsysfile *f, int page, int pageSize, int a, void volatile **b)
{
	return 0;
}

__device__ int mapShmLock(vsysfile *f, int offset, int n, int flags)
{
	return 0;
}

__device__ void mapShmBarrier(vsysfile *f)
{
}

__device__ int mapShmUnmap(vsysfile *f, int deleteFlag)
{
	return 0;
}

__device__ int mapFetch(vsysfile *f, int64_t offset, int amount, void **p)
{
	return 0;
}

__device__ int mapUnfetch(vsysfile *f, int64_t offset, void *p)
{
	return 0;
}

/* This vector defines all the methods that can operate on an sqlite3_file for win32. */
static __constant__ const vsysfile_methods _mapFileMethods = {
	3,                              // iVersion */
	mapClose,                       // xClose */
	mapRead,                        // xRead */
	mapWrite,                       // xWrite */
	mapTruncate,                    // xTruncate */
	mapSync,                        // xSync */
	mapFileSize,                    // xFileSize */
	mapLock,                        // xLock */
	mapUnlock,                      // xUnlock */
	mapCheckReservedLock,           // xCheckReservedLock */
	mapFileControl,                 // xFileControl */
	mapSectorSize,                  // xSectorSize */
	mapDeviceCharacteristics,       // xDeviceCharacteristics */
	mapShmMap,                      // xShmMap */
	mapShmLock,                     // xShmLock */
	mapShmBarrier,                  // xShmBarrier */
	mapShmUnmap,                    // xShmUnmap */
	mapFetch,                       // xFetch
	mapUnfetch                      // xUnfetch
};

#ifndef MAP_DATA_DIRECTORY_TYPE // The value used with sqlite3_win32_set_directory() to specify that the data directory should be changed.
#define MAP_DATA_DIRECTORY_TYPE (1)
#endif
#ifndef MAP_TEMP_DIRECTORY_TYPE // The value used with sqlite3_win32_set_directory() to specify that the temporary directory should be changed.
#define MAP_TEMP_DIRECTORY_TYPE (2) 
#endif

__device__ char *g_libcu_dataDirectory;
__device__ char *g_libcu_tempDirectory;

__device__ int mapSetDirectory(int type, void *value)
{
#ifndef OMIT_AUTOINIT
	int rc = runtimeInitialize();
	if (rc) return rc;
#endif
	char **directory = nullptr;
	if (type == MAP_DATA_DIRECTORY_TYPE) directory = &g_libcu_dataDirectory;
	else if (type == MAP_TEMP_DIRECTORY_TYPE) directory = &g_libcu_tempDirectory;
	assert(!directory || type == MAP_DATA_DIRECTORY_TYPE || type == MAP_TEMP_DIRECTORY_TYPE);
	assert(!directory || memdbg_hastype(*directory, MEMTYPE_HEAP));
	if (directory) {
		mfree(*directory);
		*directory = (char *)value;
		return RC_OK;
	}
	return RC_ERROR;
}

__device__ int mapOpen(vsystem *t, const char *name, vsysfile *f, int flags, int *outFlags)
{
	mapsysfile *file = (mapsysfile *)f;
	assert(file);
	memset(file, 0, sizeof(mapsysfile));
	vsystem_open msg(name, flags);
	if (outFlags)
		*outFlags = msg.OutFlags;
	file->methods = &_mapFileMethods;
	file->vsys = t;
	file->f = msg.F;
	OSTRACE(("OPEN %s %s\n", name, !msg.RC ? "ok" : "failed"));
	return msg.RC;
}

__device__ int mapDelete(vsystem *t, const char *filename, int syncDir)
{
	vsystem_delete msg(filename, syncDir);
	OSTRACE(("DELETE %s %s\n", filename, !msg.RC ? "ok" : "failed"));
	return msg.RC;
}

__device__ int mapAccess(vsystem *t, const char *filename, int flags, int *resOut)
{
	vsystem_access msg(filename, flags);
	*resOut = msg.ResOut;
	OSTRACE(("ACCESS %s %s\n", filename, !msg.RC ? "ok" : "failed"));
	return msg.RC;
}

__device__ int mapFullPathname(vsystem *t, const char *relative, int fullLength, char *full)
{
	vsystem_fullPathname msg(relative, fullLength, full);
	OSTRACE(("FULLPATHNAME %s %s\n", msg.Full, !msg.RC ? "ok" : "failed"));
	return msg.RC;
}

#ifndef OMIT_LOAD_EXTENSION
__device__ void *mapDlOpen(vsystem *t, const char *filename)
{
	return nullptr;
}

__device__ void mapDlError(vsystem *t, int bufLength, char *buf)
{
}

__device__ void (*mapDlSym(vsystem *t, void *handle, const char *symbol))()
{
	return nullptr;
}

__device__ void mapDlClose(vsystem *t, void *handle)
{
}
#else
#define winDlOpen  nullptr
#define winDlError nullptr
#define winDlSym   nullptr
#define winDlClose nullptr
#endif

__device__ int mapRandomness(vsystem *t, int bufLength, char *buf)
{
	int n = 0;
	//#if _TEST
	//		n = bufLength;
	//		_memset(buf, 0, bufLength);
	//#else
	//		if (sizeof(DWORD) <= bufLength - n)
	//		{
	//			DWORD cnt = clock();
	//			memcpy(&buf[n], &cnt, sizeof(cnt));
	//			n += sizeof(cnt);
	//		}
	//		if (sizeof(DWORD) <= bufLength - n)
	//		{
	//			DWORD cnt = clock();
	//			memcpy(&buf[n], &cnt, sizeof(cnt));
	//			n += sizeof(cnt);
	//		}
	//#endif
	return n;
}

__device__ int mapSleep(vsystem *t, int milliseconds)
{
#ifdef __CUDACC__
	clock_t start = clock();
	clock_t end = milliseconds * 10;
	for (;;) {
		clock_t now = clock();
		clock_t cycles = (now > start ? now - start : now + (0xffffffff - start));
		if (cycles >= end) break;
	}
	return ((milliseconds+999)/1000)*1000;
#else
	return 0;
#endif
}

__device__ int mapCurrentTimeInt64(vsystem *t, int64_t *now)
{
#ifdef __CUDACC__
	*now = clock();
#endif
	return RC_OK;
}

__device__ int mapCurrentTime(vsystem *t, double *now)
{
#ifdef __CUDACC__
	int64_t i; int rc = mapCurrentTimeInt64(t, &i);
	if (rc == RC_OK)
		*now = i/86400000.0;
	return rc;
#else
	return RC_OK;
#endif
}

__device__ int mapGetLastError(vsystem *t, int bufLength, char *buf)
{
	vsystem_getLastError msg(bufLength);
	buf = mprintf("%", msg.Buf);
	OSTRACE(("GETLASTERROR %s %s\n", buf, !msg.RC ? "ok" : "failed"));
	return msg.RC;
}

__device__ int mapSetSystemCall(vsystem *t, const char *name, vsystemcall_ptr newFunc)
{
	return RC_ERROR;
}
__device__ vsystemcall_ptr mapGetSystemCall(vsystem *t, const char *name)
{
	return nullptr;
}
__device__ const char *mapNextSystemCall(vsystem *t, const char *name)
{
	return nullptr;
}

static __constant__ vsystem _mapsystem = {
	3,						/* iVersion */
	sizeof(mapsysfile),     /* szOsFile */
	260,					/* mxPathname */
	0,						/* pNext */
	"map",					/* zName */
	nullptr,				/* pAppData */
	mapOpen,				/* xOpen */
	mapDelete,				/* xDelete */
	mapAccess,				/* xAccess */
	mapFullPathname,		/* xFullPathname */
	mapDlOpen,				/* xDlOpen */
	mapDlError,				/* xDlError */
	mapDlSym,				/* xDlSym */
	mapDlClose,				/* xDlClose */
	mapRandomness,			/* xRandomness */
	mapSleep,				/* xSleep */
	mapCurrentTime,			/* xCurrentTime */
	mapGetLastError,		/* xGetLastError */
	mapCurrentTimeInt64,	/* xCurrentTimeInt64 */
	mapSetSystemCall,		/* xSetSystemCall */
	mapGetSystemCall,		/* xGetSystemCall */
	mapNextSystemCall,		/* xNextSystemCall */
};

__device__ int vsystemInitialize()
{
#if defined(__CUDA_ARCH__)
	vsystemRegister(&_mapsystem, true);
	return RC_OK;
#else
	extern int sqlite3_os_init();
	return sqlite3_os_init();
#endif
}

__device__ int vsystemShutdown()
{
#if defined(__CUDA_ARCH__)
	vsystemRegister(&_mapsystem, true);
	return RC_OK;
#else
	extern int sqlite3_os_end();
	return sqlite3_os_end();
#endif
}
