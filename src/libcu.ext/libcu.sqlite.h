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
#include <ext/global.h>

#pragma region from x

#define SQLITE_PTRSIZE _PTRSIZE
#define SQLITE_WITHIN _WITHIN

#define SQLITE_BYTEORDER LIBCU_BYTEORDER
#define SQLITE_BIGENDIAN LIBCU_BIGENDIAN 
#define SQLITE_LITTLEENDIAN LIBCU_LITTLEENDIAN
#define SQLITE_UTF16NATIVE LIBCU_UTF16NATIVE

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

// CAPI3REF: Text Encodings
#define SQLITE_UTF8 TEXTENCODE_UTF8
#define SQLITE_UTF16LE TEXTENCODE_UTF16LE
#define SQLITE_UTF16BE TEXTENCODE_UTF16BE
#define SQLITE_UTF16 TEXTENCODE_UTF16
#define SQLITE_ANY TEXTENCODE_ANY
#define SQLITE_UTF16_ALIGNED TEXTENCODE_UTF16_ALIGNED

#pragma endregion

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
#define sqlite3_file vsysfile 
//#	define pMethods methods

// CAPI3REF: OS Interface File Virtual Methods Object
#define sqlite3_io_methods vsystemfile_methods

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
RC runtimeInitialize();
RC runtimeShutdown();
RC vsystemInitialize();
RC vsystemShutdown();

#pragma endregion

#pragma region From: bitvec.c
#define Bitvec bitvec_t
#define sqlite3BitvecCreate(iSize) bitvecNew(iSize)
#define sqlite3BitvecTestNotNull(p, i) bitvecGet(p, i)
#define sqlite3BitvecTest(p, i) !bitvecGet(p, i)
#define sqlite3BitvecSet(p, i) bitvecSet(p. i)
#define sqlite3BitvecClear(p, i, pBuf) bitvecClear(p, i, pBuf)
#define sqlite3BitvecDestroy(p) bitvecDestroy(p)
#define sqlite3BitvecSize(p) bitvecSize(p)
#define sqlite3BitvecBuiltinTest(sz, aOp) bitvecBuiltinTest(sz, aOp)
#pragma endregion

#pragma region From: fault.c
#define BenignMallocHooks BenignMallocHooks
#define sqlite3BenignMallocHooks(xBenignBegin, xBenignEnd) allocBenignHook(xBenignBegin, xBenignEnd)
#define sqlite3BeginBenignMalloc() allocBenignBegin()
#define sqlite3EndBenignMalloc() allocBenignEnd()
#pragma endregion

#pragma region From: global.c
#define Sqlite3Config RuntimeConfig
#pragma endregion

#pragma region From: hash.c
#define Hash hash_t
#define sqlite3HashInit(pNew) hashInit(pNew)
#define sqlite3HashClear(pH) hashClear(ph)
#define sqlite3HashFind(pH, pKey) hashFind(ph, pKey)
#define sqlite3HashInsert(pH, pKey, data) hashInsert(ph, pKey, data)
#pragma endregion

#pragma region From: hwtime.c
#define sqlite3Hwtime() panic("NEED")
#pragma endregion

#pragma region From: main.c
#define sqlite3_version libcu_version
#define sqlite3_libversion() libcu_libversion()
#define sqlite3_sourceid() libcu_sourceid()
#define sqlite3_libversion_number() libcu_libversion_number()
#define sqlite3_threadsafe() libcu_threadsafe()
#define sqlite3IoTrace panic("NEED")
#define sqlite3_temp_directory libcu_tempDirectory
#define sqlite3_data_directory libcu_dataDirectory
#define sqlite3_initialize() runtimeInitialize()
#define sqlite3_shutdown() runtimeShutdown()
#define sqlite3_config(op, ...) runtimeConfig(op, __VA_ARGS__)
#define sqlite3_db_mutex(db)
#define sqlite3_db_release_memory(db)
#define sqlite3_db_cacheflush(db)
#define sqlite3_db_config(db, op, ...)
#define sqlite3_last_insert_rowid(db)
#define sqlite3_set_last_insert_rowid(db, iRowid)
#define sqlite3_changes(db)
#define sqlite3_total_changes(db)
#define sqlite3CloseSavepoints(db)
#define sqlite3_close(db)
#define sqlite3_close_v2(db)
#define sqlite3LeaveMutexAndCloseZombie(db)
#define sqlite3RollbackAll(db, tripCode)
#define sqlite3ErrName(rc)
#define sqlite3ErrStr(rc)
#define sqlite3InvokeBusyHandler(p)
#define sqlite3_busy_handler(db, xBusy, pArg)
#define sqlite3_progress_handler(db, nOps, xProgress, pArg)
#define sqlite3_busy_timeout(db, ms)
#define sqlite3_interrupt(db){
#define sqlite3CreateFunc(db, zFunctionName, nArg, enc, pUserData, xSFunc, xStep, xFinal, pDestructor)
#define sqlite3_create_function(db, zFunc, nArg, enc, p, xSFunc, xStep, xFinal)
#define sqlite3_create_function_v2(db, zFunc, nArg, enc, p, xSFunc, xStep, xFinal, xDestroy)
#define sqlite3_overload_function(db, zName, nArg)
#define sqlite3_trace(db, xTrace, pArg)
#define sqlite3_profile(db, xProfile, pArg)
#define sqlite3_commit_hook(db, xCallback, pArg)
#define sqlite3_update_hook(db, xCallback, pArg)
#define sqlite3_rollback_hook(db, xCallback, pArg)
#define sqlite3_preupdate_hook(db,xCallback, pArg )
#define sqlite3WalDefaultHook(pClientData, db, zDb, nFrame)
#define sqlite3_wal_autocheckpoint(db, nFrame)
#define sqlite3_wal_hook(db, xCallback, pArg)
#define sqlite3_wal_checkpoint_v2(db, zDb, eMode, pnLog, pnCkpt)
#define sqlite3_wal_checkpoint(db, zDb)
#define sqlite3Checkpoint(db, iDb, eMode, pnLog, pnCkpt)
#define sqlite3TempInMemory(db)
#define sqlite3_errmsg(db)
#define sqlite3_errmsg16(db)
#define sqlite3_errcode(db)
#define sqlite3_extended_errcode(db)
#define sqlite3_system_errno(db)
#define sqlite3_errstr(rc)
#define sqlite3_limit(db, limitId, newLimit)
#define sqlite3ParseUri(zDefaultVfs, zUri, pFlags, ppVfs, pzFile, pzErrMsg)
#define sqlite3_open(zFilename, ppDb)
#define sqlite3_open16(zFilename, ppDb)
#define sqlite3_create_collation(db, zName, enc, pCtx, xCompare)
#define sqlite3_create_collation_v2(db, zName, enc, pCtx, xCompare, xDel)
#define sqlite3_create_collation16(db, zName, enc, pCtx, xCompare)
#define sqlite3_collation_needed(db, pCollNeededArg, xCollNeeded)
#define sqlite3_collation_needed16(db, pCollNeededArg, xCollNeeded16)
#define sqlite3_global_recover()
#define sqlite3_get_autocommit(db)
#define sqlite3CorruptError(lineno)
#define sqlite3MisuseError(lineno)
#define sqlite3CantopenError(lineno)
#define sqlite3CorruptPgnoError(lineno, pgno)
#define sqlite3NomemError(lineno)
#define sqlite3IoerrnomemError(lineno)
#define sqlite3_thread_cleanup()
#define sqlite3_table_column_metadata(db, zDbName, zTableName, zColumnName, pzDataType, pzCollSeq, pNotNull, pPrimaryKey, pAutoinc)
#define sqlite3_sleep(ms)
#define sqlite3_extended_result_codes(db, onoff)
#define sqlite3_file_control(db, zDbName, op, pArg)
#define sqlite3_test_control(op, ...)
#define sqlite3_uri_parameter(zFilename, zParam)
#define sqlite3_uri_boolean(zFilename, zParam, bDflt)
#define sqlite3_uri_int64(zFilename, zParam, bDflt)
#define sqlite3DbNameToBtree(db, zDbName)
#define sqlite3_db_filename(db, zDbName)
#define sqlite3_db_readonly(db, zDbName)
#define sqlite3_snapshot_get(db, zDb, ppSnapshot)
#define sqlite3_snapshot_open(db, zDb, pSnapshot)
#define sqlite3_snapshot_recover(db, zDb)
#define sqlite3_snapshot_free(pSnapshot)
#define sqlite3_compileoption_used(zOptName)
#define sqlite3_compileoption_get(N)
#pragma endregion

#pragma region From: malloc.c
#define sqlite3_release_memory(n) alloc_releasememory(n)
#define sqlite3MallocMutex() allocMutex()
#define sqlite3_memory_alarm(xCallback, pArg, iThreshold) panic("DEPRECATED")
#define sqlite3_soft_heap_limit64(n) alloc_softheaplimit64()
#define sqlite3_soft_heap_limit(n) alloc_softheaplimit()
#define sqlite3MallocInit() allocInitialize()
#define sqlite3HeapNearlyFull() allocHeapNearlyFull()
#define sqlite3MallocEnd() allocShutdown()
#define sqlite3_memory_used() alloc_memoryused()
#define sqlite3_memory_highwater(resetFlag) alloc_memoryhighwater(resetFlag)
#define sqlite3Malloc(n) alloc(n)
#define sqlite3_malloc(n) alloc32(n)
#define sqlite3_malloc64(n) alloc64(n)
#define sqlite3MallocSize(p) allocSize(p)
#define sqlite3DbMallocSize(db, p) tagallocSize(db, p)
#define sqlite3_msize(p) alloc_msize(p)
#define sqlite3_free(p) mfree(p)
#define sqlite3DbFreeNN(db, p) tagfreeNN(db, p)
#define sqlite3DbFree(db, p) tagfree(db, p)
#define sqlite3Realloc(pOld, nBytes) allocRealloc(pOld, nBytes)
#define sqlite3_realloc(pOld, n) alloc_realloc32(pOld, n)
#define sqlite3_realloc64(pOld, n) alloc_realloc64(pOld, n)
#define sqlite3MallocZero(n) allocZero(n)
#define sqlite3DbMallocZero(db, n) tagallocZero(db, n)
#define sqlite3DbMallocRaw(db, n) tagallocRaw(db, n)
#define sqlite3DbMallocRawNN(db, n) tagallocRawNN(db, n)
#define sqlite3DbRealloc(db, p, n) tagrealloc(db, p, n)
#define sqlite3DbReallocOrFree(db, p, n) tagreallocOrFree(db, p, n)
#define sqlite3DbStrDup(db, z) tagstrdup(db, z)
#define sqlite3DbStrNDup(db, z, n) tagstrndup(db, z, n)
#define sqlite3SetString(pz, db, zNew) tagstrset(pz, db, zNew)
#define sqlite3OomFault(db) tagOomFault(db)
#define sqlite3OomClear(db) tagOomClear(db)
#define sqlite3ApiExit(db, rc) tagApiExit(db, rc)
#pragma endregion

#pragma region From: mem0.c, mem1.c
#define sqlite3MemSetDefault() __allocsystemSetDefault()
#pragma endregion

#pragma region From: memjournal.c
#define sqlite3_file memfile_t
#define sqlite3JournalOpen(pVfs, zName, pJfd, flags, nSpill)
#define sqlite3MemJournalOpen(pJfd) memfileOpen
#define sqlite3JournalCreate(pJfd)
#define sqlite3JournalIsInMemory(p)
#define sqlite3JournalSize(pVfs)
#pragma endregion

#pragma region From: mutex_noop.c, mutex_unix.c, mutex_w32.c
#define sqlite3DefaultMutex() __mutexsystemDefault()
#pragma endregion

#pragma region From: mutex.c
#define sqlite3_mutex mutex
#define sqlite3MutexInit() mutexInitialize()
#define sqlite3MutexEnd() mutexShutdown()
#define sqlite3_mutex_alloc(id) mutex_alloc(id)
#define sqlite3MutexAlloc(id) mutexAlloc(id)
#define sqlite3_mutex_free(p) mutex_free(p)
#define sqlite3_mutex_enter(p) mutex_enter(p)
#define sqlite3_mutex_try(p) mutex_tryenter(p)
#define sqlite3_mutex_leave(p) mutex_leave(p)
#define sqlite3_mutex_held(p) mutex_held(p)
#define sqlite3_mutex_notheld(p) mutex_notheld(p)
#pragma endregion

#pragma region From: notify.c
#pragma endregion

#pragma region From: os_unix.c, os_win.c 
#pragma endregion

#pragma region From: os.c
#define sqlite3_io_error_hit libcu_io_error_hit
#define sqlite3_io_error_hardhit libcu_io_error_hardhit
#define sqlite3_io_error_pending libcu_io_error_pending
#define sqlite3_io_error_persist libcu_io_error_persist
#define sqlite3_io_error_benign libcu_io_error_benign
#define sqlite3_diskfull_pending libcu_diskfull_pending
#define sqlite3_diskfull libcu_diskfull
#define sqlite3_open_file_count libcu_open_file_count
#define sqlite3_memdebug_vfs_oom_test libcu_memdebug_vfs_oom_test
#define sqlite3OsClose(pId) vsys_close(pId)
#define sqlite3OsRead(id, pBuf, amt, offset) vsys_read(id, pBuf, amt, offset)
#define sqlite3OsWrite(id, pBuf, amt, offset) vsys_write(id, pBuf, amt, offset)
#define sqlite3OsTruncate(id, size) vsys_truncate(id, size)
#define sqlite3OsSync(id, flags) vsys_sync(id, flags)
#define sqlite3OsFileSize(id, pSize) vsys_fileSize(id, pSize)
#define sqlite3OsLock(id, lockType) vsys_lock(id, lockType)
#define sqlite3OsUnlock(id, lockType) vsys_unlock(id, lockType)
#define sqlite3OsCheckReservedLock(id, pResOut) vsys_checkReservedLock(id, pResOut)
#define sqlite3OsFileControl(id, op, pArg) vsys_fileControl(id, op, pArg)
#define sqlite3OsFileControlHint(id, op, pArg) vsys_fileControlHint(id, op, pArg)
#define sqlite3OsSectorSize(id) vsys_sectorSize(id)
#define sqlite3OsDeviceCharacteristics(id) vsys_deviceCharacteristics(id)
#define sqlite3OsShmLock(id, offset, n, flags) vsys_shmLock(id, offset, n, flags)
#define sqlite3OsShmBarrier(id) vsys_shmBarrier(id)
#define sqlite3OsShmUnmap(id, deleteFlag) vsys_shmUnmap(id, deleteFlag)
#define sqlite3OsShmMap(id, iPage, pgsz, bExtend, pp) vsys_shmMap(id, iPage, pgsz, bExtend, pp)
#define sqlite3OsFetch(id, iOff, iAmt, pp) vsys_fetch(id, iOff, iAmt, pp)
#define sqlite3OsUnfetch(id, iOff, p) vsys_unfetch(id, iOff, p)
#define sqlite3OsOpen(pVfs, zPath, pFile, flags, pFlagsOut) vsys_open(pVfs, zPath, pFile, flags, pFlagsOut)
#define sqlite3OsDelete(pVfs, zPath, dirSync) vsys_delete(pVfs, zPath, dirSync)
#define sqlite3OsAccess(pVfs, zPath, flags, pResOut) vsys_access(pVfs, zPath, flags, pResOut)
#define sqlite3OsFullPathname(pVfs, zPath, nPathOut, zPathOut) vsys_fullPathname(pVfs, zPath, nPathOut, zPathOut)
#define sqlite3OsDlOpen(pVfs, zPath) vsys_dlOpen(pVfs, zPath)
#define sqlite3OsDlError(pVfs, nByte, zBufOut) vsys_dlError(pVfs, nByte, zBufOut)
#define sqlite3OsDlSym(pVfs, pHdle, zSym) vsys_dlSym(pVfs, pHdle, zSym)
#define sqlite3OsDlClose(pVfs, pHandle) vsys_dlClose(pVfs, pHandle)
#define sqlite3OsRandomness(pVfs, nByte, zBufOut) vsys_randomness(pVfs, nByte, zBufOut)
#define sqlite3OsSleep(pVfs, nMicro) vsys_sleep(pVfs, nMicro)
#define sqlite3OsGetLastError(pVfs) vsys_getLastError(pVfs)
#define sqlite3OsCurrentTimeInt64(pVfs, pTimeOut) vsys_currentTimeInt64(pVfs, pTimeOut)
#define sqlite3OsOpenMalloc(pVfs, zFile, ppFile, flags, pOutFlags) vsys_openMalloc(pVfs, zFile, ppFile, flags, pOutFlags)
#define sqlite3OsCloseFree(pFile) vsys_closeAndFree(pFile)
#define sqlite3OsInit() vsystemFakeInit()
#define sqlite3_vfs_find(zVfs) vsystemFind(zVfs)
#define sqlite3_vfs_register(pVfs, makeDflt) vsystemRegister(pVfs, makeDflt)
#define sqlite3_vfs_unregister(pVfs) vsystemUnregister(pVfs)
#pragma endregion

#pragma region From: pcache1.c
#define sqlite3PCacheBufferSetup(pBuf, sz, n) pcacheBufferSetup(pBuf, sz, n)
#define sqlite3PageMalloc(sz) sqlite3PageMalloc(sz)
#define sqlite3PageFree(p) sqlite3PageFree(p)
#define sqlite3PCacheSetDefault() __pcachesystemSetDefault()
#define sqlite3HeaderSizePcache1() pcache1HeaderSize()
#define sqlite3Pcache1Mutex() pcache1Mutex()
#define sqlite3PcacheReleaseMemory(nReq) pcacheReleaseMemory(nReq)
#define sqlite3PcacheStats(pnCurrent, pnMax, pnMin, pnRecyclable) pcacheStats(pnCurrent, pnMax, pnMin, pnRecyclable)
#pragma endregion

#pragma region From: printf.c
#define StrAccum strbuf_t
#define sqlite3VXPrintf(pAccum, fmt, ap) strbldAppendFormat(pAccum, fmt, ap)
#define sqlite3AppendChar(p, N, c) strbldAppendChar(p, N, c)
#define sqlite3StrAccumAppend(p, z, N) strbldAppend(p, z, N)
#define sqlite3StrAccumAppendAll(p, z) strbldAppendAll(p, z)
#define sqlite3StrAccumFinish(p) strbldToString(p)
#define sqlite3StrAccumReset(p) strbldReset(p)
#define sqlite3StrAccumInit(p, db, zBase, n, mx) strbldInit(p, db, zBase, n, mx)
#define sqlite3VMPrintf(db, zFormat, ap)
#define sqlite3MPrintf(db, zFormat, ...)
#define sqlite3_vmprintf(zFormat, ap)
#define sqlite3_mprintf(zFormat, ...)
#define sqlite3_vsnprintf(n, zBuf, zFormat, ap)
#define sqlite3_snprintf(n, zBuf, zFormat, ...)
#define sqlite3_log(iErrCode, zFormat, ...)
#define sqlite3DebugPrintf(zFormat, ...)
#define sqlite3XPrintf(p, zFormat, ...)
#pragma endregion

#pragma region From: random.c
#define sqlite3_randomness(N, pBuf) randomness_(N, pBuf)
#define sqlite3PrngSaveState() randomness_save()
#define sqlite3PrngRestoreState() randomness_restore()
#pragma endregion

#pragma region From: status.c
#define sqlite3StatusValue(op) status_now(op)
#define sqlite3StatusUp(op, N) status_inc(op, N)
#define sqlite3StatusDown(op, N) status_dec(op, N)
#define sqlite3StatusHighwater(op, X) status_max(op, X)
#define sqlite3_status64(op, pCurrent, pHighwater, resetFlag) status64(op, pCurrent, pHighwater, resetFlag)
#define sqlite3_status(op, pCurrent, pHighwater, resetFlag) status(op, pCurrent, pHighwater, resetFlag)
#define sqlite3LookasideUsed(db, pHighwater) taglookasideUsed(db, pHighwater)
#define sqlite3_db_status(db, op, pCurrent, pHighwater, resetFlag)
#pragma endregion

#pragma region From: threads.c
#define SQLiteThread thread_t
#define sqlite3ThreadCreate(ppThread, xTask, pIn) thread_create(ppThread, xTask, pIn)
#define sqlite3ThreadJoin(p, ppOut) thread_join(p, ppOut)
#pragma endregion

#pragma region From: utf.c
#define sqlite3Utf8Read(pz) utf8read(pz)
#define sqlite3VdbeMemTranslate(pMem, desiredEnc) panic("NEED")
#define sqlite3VdbeMemHandleBom(pMem) panic("NEED")
#define sqlite3Utf8CharLen(zIn, nByte) utf8charlen(zIn, nByte)
#define sqlite3Utf8To8(zIn) utf8to8(zIn)
#define sqlite3Utf16to8(db, z, nByte, enc) panic("NEED")
#define sqlite3Utf16ByteLen(zIn, nChar) utf16bytelen(zIn, nChar)
#define sqlite3UtfSelfTest() utfselftest()
#pragma endregion

#pragma region From: util.c
#define sqlite3Coverage(x) __coverage(x)
#define sqlite3FaultSim(iTest) sqlite3FaultSim(iTest)
#define sqlite3IsNaN(x) math_isNaN(x)
#define sqlite3Strlen30(z) strlen(z)
#define sqlite3ColumnType(pCol, zDflt) panic("NEED")
#define sqlite3Error(db, err_code) tagError(db, err_code)
#define sqlite3SystemError(db, rc) tagSystemError(db, rc)
#define sqlite3ErrorWithMsg(db, err_code, zFormat, ...) tagErrorWithMsg(db, err_code, zFormat, __VAR_ARGS__)
#define sqlite3ErrorMsg(pParse, zFormat, ...)
#define sqlite3Dequote(z) dequote(z)
#define sqlite3TokenInit(p, z) panic("NEED")
#define sqlite3_stricmp(zLeft, zRight) stricmp(zLeft, zRight)
#define sqlite3StrICmp(zLeft, zRight) stricmp(zLeft, zRight)
#define sqlite3_strnicmp(zLeft, zRight, N) strnicmp(zLeft, zRight, N)
#define sqlite3AtoF(z, pResult, length, enc) convert_atofe(z, pResult, length, enc)
#define sqlite3Atoi64(zNum, pNum, length, enc) convert_atoi64e(zNum, pNum, length, enc)
#define sqlite3DecOrHexToI64(z, pOut) convert_axtoi64e(z, pOut)
#define sqlite3GetInt32(zNum, pValue) convert_atoie(zNum, pValue)
#define sqlite3Atoi(z) convert_atoi(z)
#define sqlite3PutVarint(p, v) convert_putvarint(p, v)
#define sqlite3GetVarint(p, v) convert_getvarint(p, v)
#define sqlite3GetVarint32(p, v) convert_getvarint32(p, v)
#define sqlite3VarintLen(v) convert_getvarintLength(v)
#define sqlite3Get4byte(p) convert_get4(p)
#define sqlite3Put4byte(p, v) convert_put4(p, v)
#define sqlite3HexToInt(h) convert_xtoi(h)
#define sqlite3HexToBlob(db, z, n) convert_taghextoblob(db, z, n)
#define sqlite3SafetyCheckOk(db) tagSafetyCheckOk(db)
#define sqlite3SafetyCheckSickOrOk(db) sqlite3SafetyCheckSickOrOk(db)
#define sqlite3AddInt64(pA, iB) math_add64(pA, iB)
#define sqlite3SubInt64(pA, iB) math_sub64(pA, iB)
#define sqlite3MulInt64(pA, iB) math_mul64(pA, iB)
#define sqlite3AbsInt32(x) math_abs32(x)
#define sqlite3FileSuffix3(zBaseFilename, z) util_fileSuffix3(zBaseFilename, z)
#define sqlite3LogEstAdd(a, b) math_addLogest(a, b)
#define sqlite3LogEst(x) math_logest(x)
#define sqlite3LogEstFromDouble(x) math_logestFromDouble(x)
#define sqlite3LogEstToInt(x) panic("NEED")
#define sqlite3VListAdd(db, pIn, zName, nName, iVal) util_vlistadd(db, pIn, zName, nName, iVal)
#define sqlite3VListNumToName(pIn, iVal) util_vlistIdToName(pIn, iVal)
#define sqlite3VListNameToNum(pIn, zName, nName) util_vlistNameToId(pIn, zName, nName)
#pragma endregion

#endif /* SQLITE_LIBCU_H */