/*
vsystem.h - xxx
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
#ifndef _EXT_VSYSTEM_H
#define _EXT_VSYSTEM_H
__BEGIN_DECLS;

// CAPI3REF: Flags For File Open Operations
#define VSYS_OPEN_READONLY         0x00000001  // Ok for sqlite3_open_v2()
#define VSYS_OPEN_READWRITE        0x00000002  // Ok for sqlite3_open_v2()
#define VSYS_OPEN_CREATE           0x00000004  // Ok for sqlite3_open_v2()
#define VSYS_OPEN_DELETEONCLOSE    0x00000008  // VFS only
#define VSYS_OPEN_EXCLUSIVE        0x00000010  // VFS only
#define VSYS_OPEN_AUTOPROXY        0x00000020  // VFS only
#define VSYS_OPEN_URI              0x00000040  // Ok for sqlite3_open_v2()
#define VSYS_OPEN_MEMORY           0x00000080  // Ok for sqlite3_open_v2()
#define VSYS_OPEN_MAIN_DB          0x00000100  // VFS only
#define VSYS_OPEN_TEMP_DB          0x00000200  // VFS only
#define VSYS_OPEN_TRANSIENT_DB     0x00000400  // VFS only
#define VSYS_OPEN_MAIN_JOURNAL     0x00000800  // VFS only
#define VSYS_OPEN_TEMP_JOURNAL     0x00001000  // VFS only
#define VSYS_OPEN_SUBJOURNAL       0x00002000  // VFS only
#define VSYS_OPEN_MASTER_JOURNAL   0x00004000  // VFS only
#define VSYS_OPEN_NOMUTEX          0x00008000  // Ok for sqlite3_open_v2()
#define VSYS_OPEN_FULLMUTEX        0x00010000  // Ok for sqlite3_open_v2()
#define VSYS_OPEN_SHAREDCACHE      0x00020000  // Ok for sqlite3_open_v2()
#define VSYS_OPEN_PRIVATECACHE     0x00040000  // Ok for sqlite3_open_v2()
#define VSYS_OPEN_WAL              0x00080000  // VFS only
/* Reserved:                       0x00F00000 */

// CAPI3REF: Device Characteristics
#define VSYS_IOCAP_ATOMIC                 0x00000001
#define VSYS_IOCAP_ATOMIC512              0x00000002
#define VSYS_IOCAP_ATOMIC1K               0x00000004
#define VSYS_IOCAP_ATOMIC2K               0x00000008
#define VSYS_IOCAP_ATOMIC4K               0x00000010
#define VSYS_IOCAP_ATOMIC8K               0x00000020
#define VSYS_IOCAP_ATOMIC16K              0x00000040
#define VSYS_IOCAP_ATOMIC32K              0x00000080
#define VSYS_IOCAP_ATOMIC64K              0x00000100
#define VSYS_IOCAP_SAFE_APPEND            0x00000200
#define VSYS_IOCAP_SEQUENTIAL             0x00000400
#define VSYS_IOCAP_UNDELETABLE_WHEN_OPEN  0x00000800
#define VSYS_IOCAP_POWERSAFE_OVERWRITE    0x00001000
#define VSYS_IOCAP_IMMUTABLE              0x00002000

// CAPI3REF: File Locking Levels
#define VSYS_LOCK_NONE          0
#define VSYS_LOCK_SHARED        1
#define VSYS_LOCK_RESERVED      2
#define VSYS_LOCK_PENDING       3
#define VSYS_LOCK_EXCLUSIVE     4

// CAPI3REF: Synchronization Type Flags
#define VSYS_SYNC_NORMAL        0x00002
#define VSYS_SYNC_FULL          0x00003
#define VSYS_SYNC_DATAONLY      0x00010

// CAPI3REF: OS Interface Open File Handle
typedef struct vsysfile vsysfile;
struct vsysfile {
	const struct vsysfile_methods *methods;  // Methods for an open file
};

// CAPI3REF: OS Interface File Virtual Methods Object
typedef struct vsysfile_methods vsysfile_methods;
struct vsysfile_methods {
	int version;
	int (*close)(vsysfile *);
	int (*read)(vsysfile *, void *, int amount, int64_t offset);
	int (*write)(vsysfile *, const void *, int amount, int64_t offset);
	int (*truncate)(vsysfile *, int64_t size);
	int (*sync)(vsysfile *, int flags);
	int (*fileSize)(vsysfile *, int64_t *size);
	int (*lock)(vsysfile *, int);
	int (*unlock)(vsysfile *, int);
	int (*checkReservedLock)(vsysfile *, int *resOut);
	int (*fileControl)(vsysfile *, int op, void *args);
	int (*sectorSize)(vsysfile *);
	int (*deviceCharacteristics)(vsysfile *);
	int (*shmMap)(vsysfile *, int page, int pageSize, int, void volatile **);
	int (*shmLock)(vsysfile *, int offset, int n, int flags);
	void (*shmBarrier)(vsysfile *);
	int (*shmUnmap)(vsysfile *, int deleteFlag);
	int (*fetch)(vsysfile *, int64_t offset, int amount, void **p);
	int (*unfetch)(vsysfile *, int64_t offset, void *p);
};

// CAPI3REF: Standard File Control Opcodes
#define VSYS_FCNTL_LOCKSTATE               1
#define VSYS_FCNTL_GET_LOCKPROXYFILE       2
#define VSYS_FCNTL_SET_LOCKPROXYFILE       3
#define VSYS_FCNTL_LAST_ERRNO              4
#define VSYS_FCNTL_SIZE_HINT               5
#define VSYS_FCNTL_CHUNK_SIZE              6
#define VSYS_FCNTL_FILE_POINTER            7
#define VSYS_FCNTL_SYNC_OMITTED            8
#define VSYS_FCNTL_WIN32_AV_RETRY          9
#define VSYS_FCNTL_PERSIST_WAL            10
#define VSYS_FCNTL_OVERWRITE              11
#define VSYS_FCNTL_VFSNAME                12
#define VSYS_FCNTL_POWERSAFE_OVERWRITE    13
#define VSYS_FCNTL_PRAGMA                 14
#define VSYS_FCNTL_BUSYHANDLER            15
#define VSYS_FCNTL_TEMPFILENAME           16
#define VSYS_FCNTL_MMAP_SIZE              18
#define VSYS_FCNTL_TRACE                  19
#define VSYS_FCNTL_HAS_MOVED              20
#define VSYS_FCNTL_SYNC                   21
#define VSYS_FCNTL_COMMIT_PHASETWO        22
#define VSYS_FCNTL_WIN32_SET_HANDLE       23
#define VSYS_FCNTL_WAL_BLOCK              24
#define VSYS_FCNTL_ZIPVFS                 25
#define VSYS_FCNTL_RBU                    26
#define VSYS_FCNTL_VFS_POINTER            27
#define VSYS_FCNTL_JOURNAL_POINTER        28
#define VSYS_FCNTL_WIN32_GET_HANDLE       29
#define VSYS_FCNTL_PDB                    30

// CAPI3REF: Mutex Handle
typedef struct mutex mutex;
//
//// CAPI3REF: Loadable Extension Thunk
//typedef struct sqlite3_api_routines sqlite3_api_routines;

// CAPI3REF: OS Interface Object
typedef struct vsystem vsystem;
typedef void (*vsystemcall_ptr)();
struct vsystem {
	int version;			// Structure version number (currently 3)
	int sizeOsFile;			// Size of subclassed vsysfile
	int maxPathname;		// Maximum file pathname length
	vsystem *next;			// Next registered VFS
	const char *name;		// Name of this virtual file system
	void *appData;			// Pointer to application-specific data
	int (*open)(vsystem *, const char *name, vsysfile *, int flags, int *outFlags);
	int (*delete_)(vsystem *, const char *name, int syncDir);
	int (*access)(vsystem *, const char *name, int flags, int *resOut);
	int (*fullPathname)(vsystem *, const char *name, int outLength, char *out);
	void *(*dlOpen)(vsystem *, const char *filename);
	void (*dlError)(vsystem *, int bytes, char *errMsg);
	void (*(*dlSym)(vsystem *, void *, const char *symbol))();
	void (*dlClose)(vsystem *, void *);
	int (*randomness)(vsystem *, int bytes, char *out);
	int (*sleep)(vsystem *, int microseconds);
	int (*currentTime)(vsystem *, double *);
	int (*getLastError)(vsystem *, int, char *);
	int (*currentTimeInt64)(vsystem *, int64_t *);
	int (*setSystemCall)(vsystem *, const char *name, vsystemcall_ptr);
	vsystemcall_ptr (*getSystemCall)(vsystem *, const char *name);
	const char *(*nextSystemCall)(vsystem *, const char *name);
};

// CAPI3REF: Flags for the xAccess VFS method
#define VSYS_ACCESS_EXISTS    0
#define VSYS_ACCESS_READWRITE 1   // Used by PRAGMA temp_store_directory
#define VSYS_ACCESS_READ      2   // Unused

// CAPI3REF: Flags for the xShmLock VFS method
#define VSYS_SHM_UNLOCK       1
#define VSYS_SHM_LOCK         2
#define VSYS_SHM_SHARED       4
#define VSYS_SHM_EXCLUSIVE    8

// CAPI3REF: Maximum xShmLock index
#define VSYS_SHM_NLOCK        8

/* If the SET_FULLSYNC macro is not defined above, then make it a no-op */
#ifndef SET_FULLSYNC
#define SET_FULLSYNC(x, y)
#endif

/* The default size of a disk sector */
#ifndef LIBCU_DEFAULT_SECTOR_SIZE
#define LIBCU_DEFAULT_SECTOR_SIZE 4096
#endif

/* Wrapper around OS specific sqlite3_os_init() function. */
//int vsystemFakeInit(); //: sqlite3OsInit

/* Functions for accessing vsysfile methods */
extern __host_device__ void vsys_close(vsysfile *); //: sqlite3OsClose 
extern __host_device__ RC vsys_read(vsysfile *, void *, int amount, int64_t offset); //: sqlite3OsRead
extern __host_device__ RC vsys_write(vsysfile *, const void *, int amount, int64_t offset); //: sqlite3OsWrite
extern __host_device__ RC vsys_truncate(vsysfile *, int64_t size); //: sqlite3OsTruncate
extern __host_device__ RC vsys_sync(vsysfile *, int); //: sqlite3OsSync
extern __host_device__ RC vsys_fileSize(vsysfile *, int64_t *size); //: sqlite3OsFileSize
extern __host_device__ RC vsys_lock(vsysfile *, int); //: sqlite3OsLock
extern __host_device__ RC vsys_unlock(vsysfile *, int); //: sqlite3OsUnlock
extern __host_device__ RC vsys_checkReservedLock(vsysfile *, int *); //: sqlite3OsCheckReservedLock
extern __host_device__ RC vsys_fileControl(vsysfile *, int, void *); //: sqlite3OsFileControl
extern __host_device__ void vsys_fileControlHint(vsysfile *, int, void *); //: sqlite3OsFileControlHint
#define FCNTL_DB_UNCHANGED 0xca093fa0
extern __host_device__ int vsys_sectorSize(vsysfile *); //: sqlite3OsSectorSize
extern __host_device__ int vsys_deviceCharacteristics(vsysfile *); //: sqlite3OsDeviceCharacteristics
#ifndef NO_WAL
extern __host_device__ RC vsys_shmMap(vsysfile *, int, int, int, void volatile **); //: sqlite3OsShmMap
extern __host_device__ RC vsys_shmLock(vsysfile *, int, int, int); //: sqlite3OsShmLock
extern __host_device__ void vsys_shmBarrier(vsysfile *); //: sqlite3OsShmBarrier
extern __host_device__ RC vsys_shmUnmap(vsysfile *, int); //: sqlite3OsShmUnmap
#endif
extern __host_device__ RC vsys_fetch(vsysfile *, int64_t, int, void **); //: sqlite3OsFetch
extern __host_device__ RC vsys_unfetch(vsysfile *, int64_t, void *); //: sqlite3OsUnfetch

/* Functions for accessing vsystem methods */
extern __host_device__ RC vsys_open(vsystem *, const char *, vsysfile*, int, int *); //: sqlite3OsOpen
extern __host_device__ RC vsys_delete(vsystem *, const char *, int); //: sqlite3OsDelete
extern __host_device__ RC vsys_access(vsystem *, const char *, int, int *pResOut); //: sqlite3OsAccess
extern __host_device__ RC vsys_fullPathname(vsystem *, const char *, int, char *); //: sqlite3OsFullPathname
#ifndef NO_LOAD_EXTENSION
extern __host_device__ void *vsys_dlOpen(vsystem *, const char *); //: sqlite3OsDlOpen
extern __host_device__ void vsys_dlError(vsystem *, int, char *); //: sqlite3OsDlError
extern __host_device__ void (*vsys_dlSym(vsystem *, void *, const char *))(void); //: sqlite3OsDlSym
extern __host_device__ void vsys_dlClose(vsystem *, void *); //: sqlite3OsDlClose
#endif
extern __host_device__ RC vsys_randomness(vsystem *, int, char *);
extern __host_device__ RC vsys_sleep(vsystem *, int);
extern __host_device__ int vsys_getLastError(vsystem *);
extern __host_device__ RC vsys_currentTimeInt64(vsystem *, int64_t *);

/* Convenience functions for opening and closing files using alloc() to obtain space for the file-handle structure. */
extern __host_device__ RC vsys_openMalloc(vsystem *, const char *, vsysfile **, int, int *);
extern __host_device__ void vsys_closeAndFree(vsysfile *);

/* Locate a VFS by name.  If no name is given, simply return the first VFS on the list. */
extern __host_device__ vsystem *vsystemFind(const char *name);
/* Register a VFS with the system.  It is harmless to register the same VFS multiple times.  The new VFS becomes the default if makeDflt is true. */
extern __host_device__ RC vsystemRegister(vsystem *p, bool makeDefault);
/* Unregister a VFS so that it is no longer accessible. */
extern __host_device__ RC vsystemUnregister(vsystem *p);

__END_DECLS;
#endif	/* _EXT_VSYSTEM_H */