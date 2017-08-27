/*
** 2015 January 12
**
** The author disclaims copyright to this source code.  In place of
** a legal notice, here is a blessing:
**
**    May you do good and not evil.
**    May you find forgiveness for yourself and forgive others.
**    May you share freely, never taking more than you give.
**
******************************************************************************
**
** This file contains code that is specific to LIBCU.
*/
#ifndef SQLITE_LIBCU_H
#define SQLITE_LIBCU_H

// CAPI3REF: Result Codes
#define SQLITE_OK RC_OK	// Successful result
/* beginning-of-error-codes */
#define SQLITE_ERROR RC_ERROR			// SQL error or missing database
#define SQLITE_INTERNAL RC_INTERNAL		// Internal logic error in SQLite
#define SQLITE_PERM RC_PERM				// Access permission denied
#define SQLITE_ABORT RC_ABORT			// Callback routine requested an abort
#define SQLITE_BUSY RC_BUSY				// The database file is locked
#define SQLITE_LOCKED RC_LOCKED			// A table in the database is locked
#define SQLITE_NOMEM RC_NOMEM			// A malloc() failed
#define SQLITE_READONLY RC_READONLY		// Attempt to write a readonly database
#define SQLITE_INTERRUPT RC_INTERRUPT   // Operation terminated by sqlite3_interrupt()
#define SQLITE_IOERR RC_IOERR			// Some kind of disk I/O error occurred
#define SQLITE_CORRUPT RC_CORRUPT		// The database disk image is malformed
#define SQLITE_NOTFOUND RC_NOTFOUND		// Unknown opcode in sqlite3_file_control()
#define SQLITE_FULL RC_FULL				// Insertion failed because database is full
#define SQLITE_CANTOPEN RC_CANTOPEN		// Unable to open the database file
#define SQLITE_PROTOCOL RC_PROTOCOL		// Database lock protocol error
#define SQLITE_EMPTY RC_EMPTY			// Database is empty
#define SQLITE_SCHEMA RC_SCHEMA			// The database schema changed
#define SQLITE_TOOBIG RC_TOOBIG			// String or BLOB exceeds size limit
#define SQLITE_CONSTRAINT RC_CONSTRAINT // Abort due to constraint violation
#define SQLITE_MISMATCH RC_MISMATCH		// Data type mismatch
#define SQLITE_MISUSE RC_MISUSE			// Library used incorrectly
#define SQLITE_NOLFS RC_NOLFS			// Uses OS features not supported on host
#define SQLITE_AUTH RC_AUTH				// Authorization denied
#define SQLITE_FORMAT RC_FORMAT			// Auxiliary database format error
#define SQLITE_RANGE RC_RANGE			// 2nd parameter to sqlite3_bind out of range
#define SQLITE_NOTADB RC_NOTADB			// File opened that is not a database file
#define SQLITE_NOTICE RC_NOTICE			// Notifications from sqlite3_log()
#define SQLITE_WARNING RC_WARNING		// Warnings from sqlite3_log()
#define SQLITE_ROW RC_ROW				// sqlite3_step() has another row ready
#define SQLITE_DONE RC_DONE				// sqlite3_step() has finished executing

// CAPI3REF: Extended Result Codes
#define SQLITE_IOERR_READ RC_IOERR_READ
#define SQLITE_IOERR_SHORT_READ RC_IOERR_SHORT_READ
#define SQLITE_IOERR_WRITE RC_IOERR_WRITE
#define SQLITE_IOERR_FSYNC RC_IOERR_FSYNC
#define SQLITE_IOERR_DIR_FSYNC RC_IOERR_DIR_FSYNC
#define SQLITE_IOERR_TRUNCATE RC_IOERR_TRUNCATE
#define SQLITE_IOERR_FSTAT RC_IOERR_FSTAT
#define SQLITE_IOERR_UNLOCK RC_IOERR_UNLOCK
#define SQLITE_IOERR_RDLOCK RC_IOERR_RDLOCK
#define SQLITE_IOERR_DELETE RC_IOERR_DELETE
#define SQLITE_IOERR_BLOCKED RC_IOERR_BLOCKED
#define SQLITE_IOERR_NOMEM RC_IOERR_NOMEM
#define SQLITE_IOERR_ACCESS RC_IOERR_ACCESS
#define SQLITE_IOERR_CHECKRESERVEDLOCK RC_IOERR_CHECKRESERVEDLOCK
#define SQLITE_IOERR_LOCK RC_IOERR_LOCK
#define SQLITE_IOERR_CLOSE RC_IOERR_CLOSE
#define SQLITE_IOERR_DIR_CLOSE RC_IOERR_DIR_CLOSE
#define SQLITE_IOERR_SHMOPEN RC_IOERR_SHMOPEN
#define SQLITE_IOERR_SHMSIZE RC_IOERR_SHMSIZE
#define SQLITE_IOERR_SHMLOCK RC_IOERR_SHMLOCK
#define SQLITE_IOERR_SHMMAP RC_IOERR_SHMMAP
#define SQLITE_IOERR_SEEK RC_IOERR_SEEK
#define SQLITE_IOERR_DELETE_NOENT RC_IOERR_DELETE_NOENT
#define SQLITE_IOERR_MMAPRC_IOERR_MMAP
#define SQLITE_IOERR_GETTEMPPATH RC_IOERR_GETTEMPPATH
#define SQLITE_IOERR_CONVPATHRC_IOERR_CONVPATH
#define SQLITE_IOERR_VNODE RC_IOERR_VNODE
#define SQLITE_IOERR_AUTH RC_IOERR_AUTH
#define SQLITE_LOCKED_SHAREDCACHE RC_LOCKED_SHAREDCACHE
#define SQLITE_BUSY_RECOVERY RC_BUSY_RECOVERY
#define SQLITE_BUSY_SNAPSHOT RC_BUSY_SNAPSHOT
#define SQLITE_CANTOPEN_NOTEMPDIRRC_CANTOPEN_NOTEMPDIR
#define SQLITE_CANTOPEN_ISDIR RC_CANTOPEN_ISDIR
#define SQLITE_CANTOPEN_FULLPATH RC_CANTOPEN_FULLPATH
#define SQLITE_CANTOPEN_CONVPATH RC_CANTOPEN_CONVPATH
#define SQLITE_CORRUPT_VTAB RC_CORRUPT_VTAB
#define SQLITE_READONLY_RECOVERY RC_READONLY_RECOVERY
#define SQLITE_READONLY_CANTLOCK RC_READONLY_CANTLOCK
#define SQLITE_READONLY_ROLLBACK RC_READONLY_ROLLBACK
#define SQLITE_READONLY_DBMOVED RC_READONLY_DBMOVED
#define SQLITE_ABORT_ROLLBACK RC_ABORT_ROLLBACK
#define SQLITE_CONSTRAINT_CHECK RC_CONSTRAINT_CHECK
#define SQLITE_CONSTRAINT_COMMITHOOK RC_CONSTRAINT_COMMITHOOK
#define SQLITE_CONSTRAINT_FOREIGNKEY RC_CONSTRAINT_FOREIGNKEY
#define SQLITE_CONSTRAINT_FUNCTION RC_CONSTRAINT_FUNCTION
#define SQLITE_CONSTRAINT_NOTNULL RC_CONSTRAINT_NOTNULL
#define SQLITE_CONSTRAINT_PRIMARYKEY RC_CONSTRAINT_PRIMARYKEY
#define SQLITE_CONSTRAINT_TRIGGER RC_CONSTRAINT_TRIGGER
#define SQLITE_CONSTRAINT_UNIQUE RC_CONSTRAINT_UNIQUE
#define SQLITE_CONSTRAINT_VTAB RC_CONSTRAINT_VTAB
#define SQLITE_CONSTRAINT_ROWID RC_CONSTRAINT_ROWID
#define SQLITE_NOTICE_RECOVER_WAL RC_NOTICE_RECOVER_WAL
#define SQLITE_NOTICE_RECOVER_ROLLBACK RC_NOTICE_RECOVER_ROLLBACK
#define SQLITE_WARNING_AUTOINDEX RC_WARNING_AUTOINDEX
#define SQLITE_AUTH_USER RC_AUTH_USER
#define SQLITE_OK_LOAD_PERMANENTLY RC_OK_LOAD_PERMANENTLY

#pragma region vsystem

// CAPI3REF: Flags For File Open Operations
#define SQLITE_OPEN_READONLY VSYS_OPEN_READONLY
#define SQLITE_OPEN_READWRITE VSYS_OPEN_READWRITE
#define SQLITE_OPEN_CREATE VSYS_OPEN_CREATE
#define SQLITE_OPEN_DELETEONCLOSE VSYS_OPEN_DELETEONCLOSE
#define SQLITE_OPEN_EXCLUSIVE VSYS_OPEN_EXCLUSIVE
#define SQLITE_OPEN_AUTOPROXY VSYS_OPEN_AUTOPROXY
#define SQLITE_OPEN_URI VSYS_OPEN_URI
#define SQLITE_OPEN_MEMORY VSYS_OPEN_MEMORY
#define SQLITE_OPEN_MAIN_DB VSYS_OPEN_MAIN_DB
#define SQLITE_OPEN_TEMP_DB VSYS_OPEN_TEMP_DB
#define SQLITE_OPEN_TRANSIENT_DB VSYS_OPEN_TRANSIENT_DB
#define SQLITE_OPEN_MAIN_JOURNAL VSYS_OPEN_MAIN_JOURNAL
#define SQLITE_OPEN_TEMP_JOURNAL VSYS_OPEN_TEMP_JOURNAL
#define SQLITE_OPEN_SUBJOURNAL VSYS_OPEN_SUBJOURNAL
#define SQLITE_OPEN_MASTER_JOURNAL VSYS_OPEN_MASTER_JOURNAL
#define SQLITE_OPEN_NOMUTEX VSYS_OPEN_NOMUTEX
#define SQLITE_OPEN_FULLMUTEX VSYS_OPEN_FULLMUTEX
#define SQLITE_OPEN_SHAREDCACHE VSYS_OPEN_SHAREDCACHE
#define SQLITE_OPEN_PRIVATECACHE VSYS_OPEN_PRIVATECACHE
#define SQLITE_OPEN_WAL VSYS_OPEN_WAL
/* Reserved */

// CAPI3REF: Device Characteristics
#define SQLITE_IOCAP_ATOMIC VSYS_IOCAP_ATOMIC
#define SQLITE_IOCAP_ATOMIC512 VSYS_IOCAP_ATOMIC512
#define SQLITE_IOCAP_ATOMIC1K VSYS_IOCAP_ATOMIC1K
#define SQLITE_IOCAP_ATOMIC2K VSYS_IOCAP_ATOMIC2K
#define SQLITE_IOCAP_ATOMIC4K VSYS_IOCAP_ATOMIC4K
#define SQLITE_IOCAP_ATOMIC8K VSYS_IOCAP_ATOMIC8K
#define SQLITE_IOCAP_ATOMIC16K VSYS_IOCAP_ATOMIC16K
#define SQLITE_IOCAP_ATOMIC32K VSYS_IOCAP_ATOMIC32K
#define SQLITE_IOCAP_ATOMIC64K VSYS_IOCAP_ATOMIC64K
#define SQLITE_IOCAP_SAFE_APPEND VSYS_IOCAP_SAFE_APPEND
#define SQLITE_IOCAP_SEQUENTIAL VSYS_IOCAP_SEQUENTIAL
#define SQLITE_IOCAP_UNDELETABLE_WHEN_OPEN VSYS_IOCAP_UNDELETABLE_WHEN_OPEN
#define SQLITE_IOCAP_POWERSAFE_OVERWRITE VSYS_IOCAP_POWERSAFE_OVERWRITE
#define SQLITE_IOCAP_IMMUTABLE VSYS_IOCAP_IMMUTABLE

// CAPI3REF: File Locking Levels
#define SQLITE_LOCK_NONE VSYS_LOCK_NONE
#define SQLITE_LOCK_SHARED VSYS_LOCK_SHARED
#define SQLITE_LOCK_RESERVED VSYS_LOCK_RESERVED
#define SQLITE_LOCK_PENDING VSYS_LOCK_PENDING
#define SQLITE_LOCK_EXCLUSIVE VSYS_LOCK_EXCLUSIVE

// CAPI3REF: Synchronization Type Flags
#define SQLITE_SYNC_NORMAL VSYS_SYNC_NORMAL
#define SQLITE_SYNC_FULL VSYS_SYNC_FULL
#define SQLITE_SYNC_DATAONLY VSYS_SYNC_DATAONLY

// CAPI3REF: OS Interface Open File Handle
#define vsystemfile sqlite3_file
#	define pMethods methods

// CAPI3REF: OS Interface File Virtual Methods Object
#define vsystemfile_methods sqlite3_io_methods;

// CAPI3REF: Standard File Control Opcodes
#define SQLITE_FCNTL_LOCKSTATE VSYS_FCNTL_LOCKSTATE
#define SQLITE_FCNTL_GET_LOCKPROXYFILE VSYS_FCNTL_GET_LOCKPROXYFILE
#define SQLITE_FCNTL_SET_LOCKPROXYFILE VSYS_FCNTL_SET_LOCKPROXYFILE
#define SQLITE_FCNTL_LAST_ERRNO VSYS_FCNTL_LAST_ERRNO
#define SQLITE_FCNTL_SIZE_HINT VSYS_FCNTL_SIZE_HINT
#define SQLITE_FCNTL_CHUNK_SIZE VSYS_FCNTL_CHUNK_SIZE
#define SQLITE_FCNTL_FILE_POINTER VSYS_FCNTL_FILE_POINTER
#define SQLITE_FCNTL_SYNC_OMITTED VSYS_FCNTL_SYNC_OMITTED
#define SQLITE_FCNTL_WIN32_AV_RETRY VSYS_FCNTL_WIN32_AV_RETRY
#define SQLITE_FCNTL_PERSIST_WAL VSYS_FCNTL_PERSIST_WAL
#define SQLITE_FCNTL_OVERWRITE VSYS_FCNTL_OVERWRITE
#define SQLITE_FCNTL_VFSNAME VSYS_FCNTL_VFSNAME
#define SQLITE_FCNTL_POWERSAFE_OVERWRITE VSYS_FCNTL_POWERSAFE_OVERWRITE
#define SQLITE_FCNTL_PRAGMA VSYS_FCNTL_PRAGMA
#define SQLITE_FCNTL_BUSYHANDLER VSYS_FCNTL_BUSYHANDLER
#define SQLITE_FCNTL_TEMPFILENAME VSYS_FCNTL_TEMPFILENAME
#define SQLITE_FCNTL_MMAP_SIZE VSYS_FCNTL_MMAP_SIZE
#define SQLITE_FCNTL_TRACE VSYS_FCNTL_TRACE
#define SQLITE_FCNTL_HAS_MOVED VSYS_FCNTL_HAS_MOVED
#define SQLITE_FCNTL_SYNC VSYS_FCNTL_SYNC
#define SQLITE_FCNTL_COMMIT_PHASETWO VSYS_FCNTL_COMMIT_PHASETWO
#define SQLITE_FCNTL_WIN32_SET_HANDLE VSYS_FCNTL_WIN32_SET_HANDLE
#define SQLITE_FCNTL_WAL_BLOCK VSYS_FCNTL_WAL_BLOCK
#define SQLITE_FCNTL_ZIPVFS VSYS_FCNTL_ZIPVFS
#define SQLITE_FCNTL_RBU VSYS_FCNTL_RBU
#define SQLITE_FCNTL_VFS_POINTER VSYS_FCNTL_VFS_POINTER
#define SQLITE_FCNTL_JOURNAL_POINTER VSYS_FCNTL_JOURNAL_POINTER
#define SQLITE_FCNTL_WIN32_GET_HANDLE VSYS_FCNTL_WIN32_GET_HANDLE
#define SQLITE_FCNTL_PDB VSYS_FCNTL_PDB

// CAPI3REF: Mutex Handle
#define sqlite3_mutex mutex

//// CAPI3REF: Loadable Extension Thunk
//typedef struct sqlite3_api_routines sqlite3_api_routines;

// CAPI3REF: OS Interface Object
#define sqlite3_syscall_ptr sqlite3_syscall_ptr
#define sqlite3_vfs vsystem 
#	define iVersion version
#	define szOsFile sizeOsFile
#	define mxPathname maxPathname
#	define pNext next
#	define zName name
#	define pAppData appData

// CAPI3REF: Flags for the xAccess VFS method
#define SQLITE_ACCESS_EXISTS VSYS_ACCESS_EXISTS
#define SQLITE_ACCESS_READWRITE VSYS_ACCESS_READWRITE
#define SQLITE_ACCESS_READ VSYS_ACCESS_READ

// CAPI3REF: Flags for the xShmLock VFS method
#define SQLITE_SHM_UNLOCK VSYS_SHM_UNLOCK
#define SQLITE_SHM_LOCK VSYS_SHM_LOCK
#define SQLITE_SHM_SHARED VSYS_SHM_SHARED
#define SQLITE_SHM_EXCLUSIVE VSYS_SHM_EXCLUSIVE

// CAPI3REF: Maximum xShmLock index
#define SQLITE_SHM_NLOCK VSYS_SHM_NLOCK

// CAPI3REF: Initialize The SQLite Library
int sqlite3_initialize();
int sqlite3_shutdown();
int sqlite3_os_init();
int sqlite3_os_end();


#pragma endregion

#endif /* SQLITE_LIBCU_H */