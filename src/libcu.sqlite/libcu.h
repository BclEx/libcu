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

// CUDA
#define METHOD __device__
#define FIELD __device__
#define CONSTFIELD __constant__
#define __error__ ERROR
#define __implement__ panic("__implement__") 

// CAPI3REF: OS Interface Open File Handle
#define sqlite3_file vsystemfile

// CAPI3REF: OS Interface File Virtual Methods Object
#define sqlite3_io_methods __error__

// CAPI3REF: Mutex Handle
#define sqlite3_mutex mutex

// CAPI3REF: OS Interface Object
#define sqlite3_vfs vsystem

// CAPI3REF: OS Interface Object
#define sqlite3_syscall_ptr __error__

// CAPI3REF: Initialize The SQLite Library
#define sqlite3_initialize() 0
#define sqlite3_shutdown() 0
#define sqlite3_os_init() 0
#define sqlite3_os_end() 0

// CAPI3REF: Configuring The SQLite Library
#define sqlite3_config(op, ...) __implement__

// CAPI3REF: Configure database connections
#define sqlite3_db_vconfig(tag, op, ...) __implement__

// CAPI3REF: Memory Allocation Routines
#define sqlite3_mem_methods alloc_methods

// CAPI3REF: Formatted String Printing Functions
//header: #define sqlite3_mprintf(f, ...) mprintf(f, __VA_ARGS__)
//header: #define sqlite3_vmprintf(f, va) vmprintf(f, va)
//header: #define sqlite3_snprintf(m, s, f, ...) msnprintf(s, m, f, __VA_ARGS__)
//header: #define sqlite3_vsnprintf(m, s, f, va) vmsnprintf(s, m, f, va)

// CAPI3REF: Memory Allocation Subsystem
#define sqlite3_malloc(s) alloc(s)
#define sqlite3_malloc64(s) alloc(s)
#define sqlite3_realloc(p, s) alloc_realloc(p, s)
#define sqlite3_realloc64(p, s) alloc_realloc(p, s)
#define sqlite3_free(p) alloc_free(p)
#define sqlite3_msize(p) (sqlite3_uint64)alloc_msize(p)
#define sqlite3_memory_used() alloc_memoryused()
#define sqlite3_memory_highwater(resetFlag) alloc_memoryhighwater(resetFlag)

// CAPI3REF: Pseudo-Random Number Generator
#define sqlite3_randomness(n, p) __implement__

// CAPI3REF: Obtain Values For URI Parameters
#define sqlite3_uri_parameter(zFilename, zParam) __implement__
#define sqlite3_uri_boolean(zFile, zParam, bDefault) __implement__
#define sqlite3_uri_int64(a, b, c) __implement__

// CAPI3REF: Error Codes And Messages
#define sqlite3_errcode(tag) __implement__
#define sqlite3_extended_errcode(tag) __implement__
#define sqlite3_errmsg(tag) __implement__
#define sqlite3_errmsg16(tag) __implement__
#define sqlite3_errstr(code) __implement__

// CAPI3REF: Suspend Execution For A Short Time
#define sqlite3_sleep(time) _sleep(time)

// CAPI3REF: Enable Or Disable Shared Pager Cache
#define sqlite3_enable_shared_cache(a) __implement__

// CAPI3REF: Attempt To Free Heap Memory
#define sqlite3_release_memory(a) __implement__

// CAPI3REF: Free Memory Used By A Database Connection
#define sqlite3_db_release_memory(tag) __implement__

// CAPI3REF: Impose A Limit On Heap Size
#define sqlite3_soft_heap_limit64(n) alloc_softheaplimit64(n)
#define sqlite3_soft_heap_limit(n) sqlite3_soft_heap_limit64(n) // SQLITE_DEPRECATED

// CAPI3REF: Virtual File System Objects
#define sqlite3_vfs_find(zVfsName) vsystem_find(zVfsName)
#define sqlite3_vfs_register(system, makeDflt) vsystem_register(system, makeDflt)
#define sqlite3_vfs_unregister(system) vsystem_unregister(system)

// CAPI3REF: Mutexes
#define sqlite3_mutex_alloc(n) mutex_alloc(m)
#define sqlite3_mutex_free(m) mutex_free(m)
#define sqlite3_mutex_enter(m) mutex_enter(m)
#define sqlite3_mutex_try(m) mutex_tryenter(m)
#define sqlite3_mutex_leave(m) mutex_leave(m)

// CAPI3REF: Mutex Methods Object
#define sqlite3_mutex_methods mutex_methods

// CAPI3REF: Mutex Verification Routines
#ifndef NDEBUG
#define sqlite3_mutex_held(m) mutex_held(m)
#define sqlite3_mutex_notheld(m) mutex_notheld(m)
#endif

// CAPI3REF: Retrieve the mutex for a database connection
#define sqlite3_db_mutex(tag) __implement__

// CAPI3REF: Low-Level Control Of Database Files
#define sqlite3_file_control(tag, zDbName, op, a) __implement__

// CAPI3REF: Testing Interface
#define sqlite3_test_vcontrol(op, ...) __implement__

// CAPI3REF: SQLite Runtime Status
#define sqlite3_status(op, pCurrent, pHighwater, resetFlag) status(op, pCurrent, pHighwater, resetFlag)
#define sqlite3_status64(op, pCurrent, pHighwater, resetFlag) status64(op, pCurrent, pHighwater, resetFlag)

// CAPI3REF: Database Connection Status
#define sqlite3_db_status(tag, op, pCur, pHiwtr, resetFlg) __implement__

// CAPI3REF: String Comparison
#define sqlite3_stricmp(a, b) stricmp(a, b)
#define sqlite3_strnicmp(a, b, n) strnicmp(a, b, n)

// CAPI3REF: String Globbing
#define sqlite3_strglob(zGlob, zStr) __implement__

// CAPI3REF: String LIKE Matching
#define sqlite3_strlike(zGlob, zStr, cEsc) __implement__

// CAPI3REF: Error Logging Interface
#define sqlite3_log(iErrCode, zFormat, ...) __implement__

// CAPI3REF: Low-level system error code
#define sqlite3_system_errno(tag) __implement__

#endif /* SQLITE_LIBCU_H */