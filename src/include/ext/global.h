/*
global.h - xxx
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

#define ENABLE_API_ARMOR 1
#define LIBCU_ENABLE_SQLLOG
#define OMIT_COMPILEOPTION_DIAGS
#define LIBCU_ENABLE_UNLOCK_NOTIFY

#ifndef _EXTGLOBAL__H
#define _EXTGLOBAL__H
#include <crtdefscu.h>
#include <stdargcu.h>
__BEGIN_DECLS;

/* Forward references to structures */
typedef struct mutex mutex;
typedef struct tagbase_t tagbase_t;
typedef struct Lookaside Lookaside;
typedef struct LookasideSlot LookasideSlot;

#pragma region From: sqlite.h

// CAPI3REF: Result Codes
#define RC int
#define RC_OK           0   // Successful result
/* beginning-of-error-codes */
#define RC_ERROR        1   // SQL error or missing database
#define RC_INTERNAL     2   // Internal logic error in SQLite
#define RC_PERM         3   // Access permission denied
#define RC_ABORT        4   // Callback routine requested an abort
#define RC_BUSY         5   // The database file is locked
#define RC_LOCKED       6   // A table in the database is locked
#define RC_NOMEM        7   // A malloc() failed
#define RC_READONLY     8   // Attempt to write a readonly database
#define RC_INTERRUPT    9   // Operation terminated by sqlite3_interrupt()
#define RC_IOERR       10   // Some kind of disk I/O error occurred
#define RC_CORRUPT     11   // The database disk image is malformed
#define RC_NOTFOUND    12   // Unknown opcode in sqlite3_file_control()
#define RC_FULL        13   // Insertion failed because database is full
#define RC_CANTOPEN    14   // Unable to open the database file
#define RC_PROTOCOL    15   // Database lock protocol error
#define RC_EMPTY       16   // Database is empty
#define RC_SCHEMA      17   // The database schema changed
#define RC_TOOBIG      18   // String or BLOB exceeds size limit
#define RC_CONSTRAINT  19   // Abort due to constraint violation
#define RC_MISMATCH    20   // Data type mismatch
#define RC_MISUSE      21   // Library used incorrectly
#define RC_NOLFS       22   // Uses OS features not supported on host
#define RC_AUTH        23   // Authorization denied
#define RC_FORMAT      24   // Auxiliary database format error
#define RC_RANGE       25   // 2nd parameter to sqlite3_bind out of range
#define RC_NOTADB      26   // File opened that is not a database file
#define RC_NOTICE      27   // Notifications from sqlite3_log()
#define RC_WARNING     28   // Warnings from sqlite3_log()
#define RC_ROW         100  // sqlite3_step() has another row ready
#define RC_DONE        101  // sqlite3_step() has finished executing

// CAPI3REF: Extended Result Codes
#define RC_IOERR_READ              (RC_IOERR | (1<<8))
#define RC_IOERR_SHORT_READ        (RC_IOERR | (2<<8))
#define RC_IOERR_WRITE             (RC_IOERR | (3<<8))
#define RC_IOERR_FSYNC             (RC_IOERR | (4<<8))
#define RC_IOERR_DIR_FSYNC         (RC_IOERR | (5<<8))
#define RC_IOERR_TRUNCATE          (RC_IOERR | (6<<8))
#define RC_IOERR_FSTAT             (RC_IOERR | (7<<8))
#define RC_IOERR_UNLOCK            (RC_IOERR | (8<<8))
#define RC_IOERR_RDLOCK            (RC_IOERR | (9<<8))
#define RC_IOERR_DELETE            (RC_IOERR | (10<<8))
#define RC_IOERR_BLOCKED           (RC_IOERR | (11<<8))
#define RC_IOERR_NOMEM             (RC_IOERR | (12<<8))
#define RC_IOERR_ACCESS            (RC_IOERR | (13<<8))
#define RC_IOERR_CHECKRESERVEDLOCK (RC_IOERR | (14<<8))
#define RC_IOERR_LOCK              (RC_IOERR | (15<<8))
#define RC_IOERR_CLOSE             (RC_IOERR | (16<<8))
#define RC_IOERR_DIR_CLOSE         (RC_IOERR | (17<<8))
#define RC_IOERR_SHMOPEN           (RC_IOERR | (18<<8))
#define RC_IOERR_SHMSIZE           (RC_IOERR | (19<<8))
#define RC_IOERR_SHMLOCK           (RC_IOERR | (20<<8))
#define RC_IOERR_SHMMAP            (RC_IOERR | (21<<8))
#define RC_IOERR_SEEK              (RC_IOERR | (22<<8))
#define RC_IOERR_DELETE_NOENT      (RC_IOERR | (23<<8))
#define RC_IOERR_MMAP              (RC_IOERR | (24<<8))
#define RC_IOERR_GETTEMPPATH       (RC_IOERR | (25<<8))
#define RC_IOERR_CONVPATH          (RC_IOERR | (26<<8))
#define RC_IOERR_VNODE             (RC_IOERR | (27<<8))
#define RC_IOERR_AUTH              (RC_IOERR | (28<<8))
#define RC_LOCKED_SHAREDCACHE      (RC_LOCKED |  (1<<8))
#define RC_BUSY_RECOVERY           (RC_BUSY   |  (1<<8))
#define RC_BUSY_SNAPSHOT           (RC_BUSY   |  (2<<8))
#define RC_CANTOPEN_NOTEMPDIR      (RC_CANTOPEN | (1<<8))
#define RC_CANTOPEN_ISDIR          (RC_CANTOPEN | (2<<8))
#define RC_CANTOPEN_FULLPATH       (RC_CANTOPEN | (3<<8))
#define RC_CANTOPEN_CONVPATH       (RC_CANTOPEN | (4<<8))
#define RC_CORRUPT_VTAB            (RC_CORRUPT | (1<<8))
#define RC_READONLY_RECOVERY       (RC_READONLY | (1<<8))
#define RC_READONLY_CANTLOCK       (RC_READONLY | (2<<8))
#define RC_READONLY_ROLLBACK       (RC_READONLY | (3<<8))
#define RC_READONLY_DBMOVED        (RC_READONLY | (4<<8))
#define RC_ABORT_ROLLBACK          (RC_ABORT | (2<<8))
#define RC_CONSTRAINT_CHECK        (RC_CONSTRAINT | (1<<8))
#define RC_CONSTRAINT_COMMITHOOK   (RC_CONSTRAINT | (2<<8))
#define RC_CONSTRAINT_FOREIGNKEY   (RC_CONSTRAINT | (3<<8))
#define RC_CONSTRAINT_FUNCTION     (RC_CONSTRAINT | (4<<8))
#define RC_CONSTRAINT_NOTNULL      (RC_CONSTRAINT | (5<<8))
#define RC_CONSTRAINT_PRIMARYKEY   (RC_CONSTRAINT | (6<<8))
#define RC_CONSTRAINT_TRIGGER      (RC_CONSTRAINT | (7<<8))
#define RC_CONSTRAINT_UNIQUE       (RC_CONSTRAINT | (8<<8))
#define RC_CONSTRAINT_VTAB         (RC_CONSTRAINT | (9<<8))
#define RC_CONSTRAINT_ROWID        (RC_CONSTRAINT |(10<<8))
#define RC_NOTICE_RECOVER_WAL      (RC_NOTICE | (1<<8))
#define RC_NOTICE_RECOVER_ROLLBACK (RC_NOTICE | (2<<8))
#define RC_WARNING_AUTOINDEX       (RC_WARNING | (1<<8))
#define RC_AUTH_USER               (RC_AUTH | (1<<8))
#define RC_OK_LOAD_PERMANENTLY     (RC_OK | (1<<8))

// CAPI3REF: Run-time Limits
extern __host_device__ int taglimit(tagbase_t *, int id, int newVal);

// CAPI3REF: Run-Time Limit Categories
#define TAG_LIMIT_LENGTH                    0
#define TAG_LIMIT_SQL_LENGTH                1
#define TAG_LIMIT_COLUMN                    2
#define TAG_LIMIT_EXPR_DEPTH                3
#define TAG_LIMIT_COMPOUND_SELECT           4
#define TAG_LIMIT_VDBE_OP                   5
#define TAG_LIMIT_FUNCTION_ARG              6
#define TAG_LIMIT_ATTACHED                  7
#define TAG_LIMIT_LIKE_PATTERN_LENGTH       8
#define TAG_LIMIT_VARIABLE_NUMBER           9
#define TAG_LIMIT_TRIGGER_DEPTH            10
#define TAG_LIMIT_WORKER_THREADS           11
#define TAG_LIMIT_MAX (TAG_LIMIT_WORKER_THREADS+1)

// CAPI3REF: Initialize The SQLite Library
extern __host_device__ RC runtimeInitialize(); //: sqlite3_initialize
extern __host_device__ RC runtimeShutdown(); //: sqlite3_shutdown
extern __host_device__ int vsystemInitialize(); //: sqlite3_os_init
extern __host_device__ int vsystemShutdown(); //: sqlite3_os_end

// CAPI3REF: Name Of The Folder Holding Temporary Files
extern char *libcu_tempDirectory;

// CAPI3REF: Name Of The Folder Holding Database Files
extern char *libcu_dataDirectory;

#ifndef OMIT_WSD
extern __hostb_device__ int _libcuPendingByte;
#endif

// CAPI3REF: Checkpoint a database
//extern __hostb_device__ int sqlite3_wal_checkpoint_v2(sqlite3 *db, const char *zDb, int eMode, int *pnLog, int *pnCkpt); //: sqlite3_wal_checkpoint_v2

// CAPI3REF: Checkpoint Mode Values
#define PCACHE_CHECKPOINT_PASSIVE  0  // Do as much as possible w/o blocking
#define PCACHE_CHECKPOINT_FULL     1  // Wait for writers, then checkpoint
#define PCACHE_CHECKPOINT_RESTART  2  // Like FULL but wait for for readers
#define PCACHE_CHECKPOINT_TRUNCATE 3  // Like RESTART but also truncate WAL

#pragma endregion

/* If the SQLITE_ENABLE IOTRACE exists then the global variable sqlite3IoTrace is a pointer to a printf-like routine used to
** print I/O tracing messages.
*/
#ifdef LIBCU_ENABLE_IOTRACE
#define IOTRACE(A)  if (_libcuIoTracev) { _libcuIoTrace A; }
  //__host_device__ void libcuVdbeIOTraceSql(Vdbe *);
__host_device__ void (*_libcuIoTracev)(const char*,va_list);
#else
#define IOTRACE(A)
#define libcuVdbeIOTraceSql(X)
#endif

/* Disable MMAP on platforms where it is known to not work */
#if defined(__OpenBSD__) || defined(__QNXNTO__)
#undef LIBCU_MAXMMAPSIZE
#define LIBCU_MAXMMAPSIZE 0
#endif

/* Default maximum size of memory used by memory-mapped I/O in the VFS */
#ifdef __APPLE__
#include <TargetConditionals.h>
#endif
#ifndef LIBCU_MAXMMAPSIZE
#if defined(__linux__) || defined(_WIN32) || (defined(__APPLE__) && defined(__MACH__)) || defined(__sun) || defined(__FreeBSD__) || defined(__DragonFly__)
# define LIBCU_MAXMMAPSIZE 0x7fff0000  // 2147418112
#else
# define LIBCU_MAXMMAPSIZE 0
#endif
#define LIBCU_MAXMMAPSIZE_xc 1 // exclude from ctime.c
#endif

/* The default MMAP_SIZE is zero on all platforms.  Or, even if a larger default MMAP_SIZE is specified at compile-time, make sure that it does
** not exceed the maximum mmap size.
*/
#ifndef LIBCU_DEFAULTMMAPSIZE
#define LIBCU_DEFAULTMMAPSIZE 0
#define LIBCU_DEFAULTMMAPSIZE_xc 1  // Exclude from ctime.c
#endif
#if LIBCU_DEFAULTMMAPSIZE > LIBCU_MAXMMAPSIZE
#undef LIBCU_DEFAULTMMAPSIZE
#define LIBCU_DEFAULTMMAPSIZE LIBCU_MAXMMAPSIZEc
#endif

/* Lookaside malloc is a set of fixed-size buffers that can be used to satisfy small transient memory allocation requests for objects
** associated with a particular database connection.  The use of lookaside malloc provides a significant performance enhancement
** (approx 10%) by avoiding numerous malloc/free requests while parsing SQL statements.
**
** The Lookaside structure holds configuration information about the lookaside malloc subsystem.  Each available memory allocation in
** the lookaside subsystem is stored on a linked list of LookasideSlot objects.
**
** Lookaside allocations are only allowed for objects that are associated with a particular database connection.  Hence, schema information cannot
** be stored in lookaside because in shared cache mode the schema information is shared by multiple database connections.  Therefore, while parsing
** schema information, the Lookaside.bEnabled flag is cleared so that lookaside allocations are not used to construct the schema objects.
*/
struct Lookaside {
	uint32_t disable;       // Only operate the lookaside when zero
	uint16_t size;          // Size of each buffer in bytes
	bool malloced;			// True if Start obtained from alloc32()
	int slots;				// Number of lookaside slots allocated
	int stats[3];			// 0: hits.  1: size misses.  2: full misses
	LookasideSlot *init;	// List of buffers not previously used
	LookasideSlot *free_;	// List of available buffers
	void *start;			// First byte of available memory space
	void *end;				// First byte past end of available space
};
struct LookasideSlot {
	LookasideSlot *next;   // Next buffer in the list of free buffers
};

/* Each Tag object is an instance of the following structure. */
typedef struct vsystem vsystem;
typedef struct tagbase_t tagbase_t;
struct tagbase_t {
	vsystem *vsys;          // OS Interface
	mutex *mutex;			// Connection mutex
	int errCode;            // Most recent error code (RC_*)
	int errMask;            // & result codes with this before returning
	int sysErrno;           // Errno value from last system error
	int vdbeExecs;			// Number of nested calls to VdbeExec()
	union {
		volatile int isInterrupted; // True if sqlite3_interrupt has been called
		double notUsed1;            // Spacer
	} u1;
	Lookaside lookaside;    // Lookaside malloc configuration
	bool mallocFailed;      // True if we have seen a malloc failure
	bool benignMalloc;      // Do not require OOMs if true
	int *bytesFreed;		// If not NULL, increment this in DbFree()
	uint8_t suppressErr;	// Do not issue error messages if true
	uint32_t magic;         // Magic number for detect library misuse
	int limits[TAG_LIMIT_MAX]; // Limits
	void *err;				// Most recent error message
#ifdef LIBCU_ENABLE_UNLOCK_NOTIFY
	/* The following variables are all protected by the STATIC_MASTER mutex, not by sqlite3.mutex. They are used by code in notify.c.
	** When X.pUnlockConnection==Y, that means that X is waiting for Y to unlock so that it can proceed.
	** When X.pBlockingConnection==Y, that means that something that X tried tried to do recently failed with an SQLITE_LOCKED error due to locks held by Y.
	*/
	tagbase_t *blockingConnection; // Connection that caused SQLITE_LOCKED
	tagbase_t *unlockConnection;           // Connection to watch for unlock
	void *unlockArg;                     // Argument to xUnlockNotify
	void (*unlockNotify)(void **, int);  // Unlock notify callback
	tagbase_t *nextBlocked;        // Next in list of all blocked connections
#endif
};

/* Possible values for the sqlite.magic field.
** The numbers are obtained at random and have no special meaning, other than being distinct from one another.
*/
#define TAG_MAGIC_OPEN     0xa029a697  // Database is open
#define TAG_MAGIC_CLOSED   0x9f3c2d33  // Database is closed
#define TAG_MAGIC_SICK     0x4b771290  // Error and awaiting close
#define TAG_MAGIC_BUSY     0xf03b7906  // Database currently in use
#define TAG_MAGIC_ERROR    0xb5357930  // An SQLITE_MISUSE error occurred
#define TAG_MAGIC_ZOMBIE   0x64cffc7f  // Close with last statement close

/* Pasebase structure */
typedef struct parsebase_t parsebase_t;
struct parsebase_t {
	tagbase_t *tag;			// The main database structure
	char *errMsg;			// An error message
	int rc;					// Return code from execution
	int errs;				// Number of errors seen
};

__END_DECLS;
#include "vsystem.h"
#include "convert.h"
#include "util.h"
#include "math.h"
#include "mutex.h"
#include "alloc.h"
#include "status.h"
#include "pcache.h"
__BEGIN_DECLS;

// CAPI3REF: Pseudo-Random Number Generator
extern __host_device__ void randomness_(int n, void *p);

// CAPI3REF: Configuration Options
#define CONFIG int
#define CONFIG_SINGLETHREAD  1  // nil
#define CONFIG_MULTITHREAD   2  // nil
#define CONFIG_SERIALIZED    3  // nil
#define CONFIG_MALLOC        4  // alloc_methods*
#define CONFIG_GETMALLOC     5  // alloc_methods*
#define CONFIG_SCRATCH       6  // void*, int size, int n
#define CONFIG_PAGECACHE     7  // void*, int size, int n
#define CONFIG_HEAP          8  // void*, int nByte, int min
#define CONFIG_MEMSTATUS     9  // boolean
#define CONFIG_MUTEX        10  // mutex_methods*
#define CONFIG_GETMUTEX     11  // mutex_methods*
// previously CONFIG_CHUNKALLOC 12 which is now unused.
#define CONFIG_LOOKASIDE    13  // int int
#define CONFIG_PCACHE       14  // no-op
#define CONFIG_GETPCACHE    15  // no-op
#define CONFIG_LOG          16  // xFunc, void*
#define CONFIG_URI          17  // int
#define CONFIG_PCACHE2      18  // pcache_methods2*
#define CONFIG_GETPCACHE2   19  // pcache_methods2*
#define CONFIG_COVERING_INDEX_SCAN 20  // int
#define CONFIG_SQLLOG       21  // xSqllog, void*
#define CONFIG_MMAP_SIZE    22  // int64_t, int64_t
#define CONFIG_WIN32_HEAPSIZE	23  // int nByte
#define CONFIG_PCACHE_HDRSZ		24  // int *psz
#define CONFIG_PMASZ			25  // unsigned int szPma
#define CONFIG_STMTJRNL_SPILL	26  // int nByte
#define CONFIG_SMALL_MALLOC		27  // boolean

// CAPI3REF: Database Connection Configuration Options
#define SQLITE_DBCONFIG_MAINDBNAME            1000 /* const char* */
#define SQLITE_DBCONFIG_LOOKASIDE             1001 /* void* int int */
#define SQLITE_DBCONFIG_ENABLE_FKEY           1002 /* int int* */
#define SQLITE_DBCONFIG_ENABLE_TRIGGER        1003 /* int int* */
#define SQLITE_DBCONFIG_ENABLE_FTS3_TOKENIZER 1004 /* int int* */
#define SQLITE_DBCONFIG_ENABLE_LOAD_EXTENSION 1005 /* int int* */
#define SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE      1006 /* int int* */
#define SQLITE_DBCONFIG_ENABLE_QPSG           1007 /* int int* */

// CAPI3REF: Enable Or Disable Extended Result Codes
extern __host_device__ int tagextendedResultcodes(tagbase_t *, int onoff); //: sqlite3_extended_result_codes

// CAPI3REF: Last Insert Rowid
extern __host_device__ int64_t taglastInsertRowid(tagbase_t *); //: sqlite3_last_insert_rowid

// CAPI3REF: Set the Last Insert Rowid value.
extern __host_device__ void tagsetLastInsertRowid(tagbase_t *, int64_t); //: sqlite3_set_last_insert_rowid

// CAPI3REF: Count The Number Of Rows Modified
extern __host_device__ int tagchanges(tagbase_t *); //: sqlite3_changes

// CAPI3REF: Total Number Of Rows Modified
extern __host_device__ int tagtotalchanges(tagbase_t *); //: sqlite3_total_changes

// CAPI3REF: Interrupt A Long-Running Query
extern __host_device__ void taginterrupt(tagbase_t *); //: sqlite3_interrupt

// CAPI3REF: Register A Callback To Handle SQLITE_BUSY Errors
extern __host_device__ int tagbusyhandler(tagbase_t *,int(*)(void*,int),void *); //: sqlite3_busy_handler

// CAPI3REF: Set A Busy Timeout
extern __host_device__ int tagbusytimeout(tagbase_t *, int ms); //: sqlite3_busy_timeout

// CAPI3REF: SQL Trace Event Codes
#define LIBCU_TRACE_STMT       0x01
#define LIBCU_TRACE_PROFILE    0x02
#define LIBCU_TRACE_ROW        0x04
#define LIBCU_TRACE_CLOSE      0x08

// CAPI3REF: SQL Trace Hook
extern __host_device__ int tagtrace(tagbase_t *, unsigned mask, int (*callback)(unsigned,void*,void*,void*), void *ctx); //: sqlite3_trace_v2

// CAPI3REF: Query Progress Callbacks
extern __host_device__ void tagprogresshandler(tagbase_t *, int, int(*)(void*), void*); //: sqlite3_progress_handler

// CAPI3REF: Error Codes And Messages
extern __host_device__ int tagerrcode(tagbase_t *); //: sqlite3_errcode
extern __host_device__ int tagextendedErrcode(tagbase_t *); //: sqlite3_extended_errcode
extern __host_device__ const char *tagerrmsg(tagbase_t *); //: sqlite3_errmsg
extern __host_device__ const void *tagerrmsg16(tagbase_t *); //: sqlite3_errmsg16
extern __host_device__ const char *errstr(int); //: sqlite3_errstr


/* Structure containing global configuration data for the Lib library.
**
** This structure also contains some state information.
*/
typedef struct RuntimeConfig RuntimeConfig;
struct RuntimeConfig {
	void (*appendFormat[2])(void *b, va_list va); // Formatter
	bool memstat;                   // True to enable memory status
	bool coreMutex;                 // True to enable core mutexing
	bool fullMutex;                 // True to enable full mutexing
	bool openUri;                   // True to interpret filenames as URIs
	bool useCis;                    // Use covering indices for full-scans
	bool smallMalloc;               // Avoid large memory allocations if true
	int maxStrlen;                  // Maximum string length
	bool neverCorrupt;              // Database is always well-formed
	int lookasideSize;              // Default lookaside buffer size
	int lookasides;					// Default lookaside buffer count
	int stmtSpills;                 // Stmt-journal spill-to-disk threshold
	alloc_methods allocSystem;		// Low-level memory allocation interface
	mutex_methods mutexSystem;		// Low-level mutex interface
	pcache_methods pcache2System;	// Low-level page-cache interface
	void *heap;						// Heap storage space
	int heapSize;                   // Size of heap[]
	int minHeapSize, maxHeapSize;	// Min and max heap requests sizes
	int64_t sizeMmap;				// mmap() space per open file
	int64_t maxMmap;				// Maximum value for szMmap
	void *page;                     // Page cache memory
	int pageSize;                   // Size of each page in page[]
	int pages;                      // Number of pages in page[]
	int maxParserStack;             // Maximum depth of the parser stack
	bool sharedCacheEnabled;        // True if shared-cache mode enabled
	uint32_t sizePma;               // Maximum Sorter PMA size
	/* The above might be initialized to non-zero.  The following need to always initially be zero, however. */
	bool isInit;					// True after initialization has finished
	bool inProgress;				// True while initialization in progress
	bool isMutexInit;				// True after mutexes are initialized
	bool isMallocInit;				// True after malloc is initialized
	bool isPcacheInit;              // True after malloc is initialized
	int initMutexRefs;				// Number of users of initMutex
	mutex *initMutex;				// Mutex used by runtimeInitialize()
	void (*log)(void*,int,const char*); // Function for logging
	void *logArg;					// First argument to xLog()
#ifdef LIBCU_ENABLE_SQLLOG
	void (*sqllog)(void*,tagbase_t*,const char*,int);
	void *sqllogArg;
#endif
#ifdef LIBCU_VDBE_COVERAGE
	// The following callback (if not NULL) is invoked on every VDBE branch operation.  Set the callback using SQLITE_TESTCTRL_VDBE_COVERAGE.
	void (*vdbeBranch)(void*,int,uint8_t,uint8_t); // Callback
	void *vdbeBranchArg;			// 1st argument
#endif
#ifndef LIBCU_UNTESTABLE
	int (*testCallback)(int);       // Invoked by sqlite3FaultSim()
#endif
	int localtimeFault;				// True to fail localtime() calls
	int onceResetThreshold;			// When to reset OP_Once counters
};
extern __hostb_device__ WSD_ RuntimeConfig _runtimeConfig;
#define _runtimeConfig GLOBAL_(RuntimeConfig, _runtimeConfig)

/* This macro is used inside of assert() statements to indicate that the assert is only valid on a well-formed database.  Instead of:
**
**     assert(X);
**
** One writes:
**
**     assert(X || CORRUPT_DB);
**
** CORRUPT_DB is true during normal operation.  CORRUPT_DB does not indicate that the database is definitely corrupt, only that it might be corrupt.
** For most test cases, CORRUPT_DB is set to false using a special sqlite3_test_control().  This enables assert() statements to prove
** things that are always true for well-formed databases.
*/
#define CORRUPT_DB (!_runtimeConfig.neverCorrupt)


#if defined(LIBCU_NEED_ERR_NAME)
__host_device__ const char *libcuErrName(int);
#endif
__host_device__ const char *libcuErrStr(int);

/* The LIBCU_*_BKPT macros are substitutes for the error codes with the same name but without the _BKPT suffix.  These macros invoke
** routines that report the line-number on which the error originated using sqlite3_log().  The routines also provide a convenient place
** to set a debugger breakpoint.
*/
__host_device__ int libcuCorruptError(int line); //: sqlite3CorruptError
__host_device__ int libcuMisuseError(int line); //: sqlite3MisuseError
__host_device__ int libcuCantopenError(int line); //: sqlite3CantopenError
#define RC_CORRUPT_BKPT libcuCorruptError(__LINE__)
#define RC_MISUSE_BKPT libcuMisuseError(__LINE__)
#define RC_CANTOPEN_BKPT libcuCantopenError(__LINE__)
#ifdef _DEBUG
__host_device__ int libcuCorruptPgnoError(int line, Pgno pgno); //: sqlite3CorruptPgnoError
__host_device__ int libcuNomemError(int line); //: sqlite3NomemError
__host_device__ int libcuIoerrnomemError(int line); //: sqlite3IoerrnomemError
#define RC_CORRUPT_PGNO(P) libCorruptPgnoError(__LINE__, (P))
#define RC_NOMEM_BKPT libcuNomemError(__LINE__)
#define RC_IOERR_NOMEM_BKPT libcuIoerrnomemError(__LINE__)
#else
#define RC_CORRUPT_PGNO(P) libcuCorruptPgnoError(__LINE__)
#define RC_NOMEM_BKPT RC_NOMEM
#define RC_IOERR_NOMEM_BKPT RC_IOERR_NOMEM
#endif

/* All current savepoints are stored in a linked list starting at sqlite3.pSavepoint. The first element in the list is the most recently
** opened savepoint. Savepoints are added to the list by the vdbe OP_Savepoint instruction.
*/
typedef struct Savepoint Savepoint;
struct Savepoint {
	char *name;                       // Savepoint name (nul-terminated)
	int64_t deferredCons;				// Number of deferred fk violations
	int64_t neferredImmCons;          // Number of deferred imm fk.
	Savepoint *next;                  // Parent savepoint (if any)
};

/* The following are used as the second parameter to sqlite3Savepoint(), and as the P1 argument to the OP_Savepoint instruction. */
#define SAVEPOINT_BEGIN      0
#define SAVEPOINT_RELEASE    1
#define SAVEPOINT_ROLLBACK   2


__END_DECLS;


// CAPI3REF: Online Backup Object
typedef struct backupbase_t backupbase_t;

// CAPI3REF: Online Backup API.
__host_device__ backupbase_t *backup_init(tagbase_t *dest, const char *destName, tagbase_t *source, const char *sourceName); //: sqlite3_backup_init
__host_device__ int backup_step(backupbase_t *p, int pages); //: sqlite3_backup_step
__host_device__ int backup_finish(backupbase_t *p); //: sqlite3_backup_finish
__host_device__ int backup_remaining(backupbase_t *p); //: sqlite3_backup_remaining
__host_device__ int backup_pagecount(backupbase_t *p); //: sqlite3_backup_pagecount


#pragma region From: notify.c
__BEGIN_DECLS;

// CAPI3REF: Unlock Notification
__host_device__ RC notify_unlock(tagbase_t *tag, void (*notify)(void**,int), void *arg);

#ifdef LIBCU_ENABLE_UNLOCK_NOTIFY
__host_device__ void notifyConnectionBlocked(tagbase_t *tag, tagbase_t *);
__host_device__ void notifyConnectionUnlocked(tagbase_t *tag);
__host_device__ void notifyConnectionClosed(tagbase_t *tag);
#else
#define notifyConnectionBlocked(x,y)
#define notifyConnectionUnlocked(x)
#define notifyConnectionClosed(x)
#endif

__END_DECLS;
#pragma endregion

#pragma region From: printf.c
__BEGIN_DECLS;

// CAPI3REF: Error Logging Interface
/* Format and write a message to the log if logging is enabled. */
__host_device__ void _logv(int errCode, const char *format, va_list va); //: sqlite3_log
#if defined(_DEBUG) || defined(LIBCU_HAVE_OS_TRACE)
__host_device__ void _debugv(const char *format, va_list va); //: sqlite3DebugPrintf
#endif

//
__host_device__ char *vmtagprintf(void *tag, const char *format, va_list va);
__host_device__ char *vmprintf(const char *format, va_list va);
__host_device__ char *vmsnprintf(char *__restrict s, size_t maxlen, const char *format, va_list va);

__END_DECLS;
// // STDARG
#ifndef __CUDA_ARCH__
__forceinline void _log(int errCode, const char *format, ...) { va_list va; va_start(va, format); _logv(errCode, format, va); va_end(va); }
#if defined(_DEBUG) || defined(LIBCU_HAVE_OS_TRACE)
__forceinline void _debug(const char *format, ...) { va_list va; va_start(va, format); _debugv(format, va); va_end(va); }
#endif
__forceinline char *mtagprintf(tagbase_t *tag, const char *format, ...) { char *r; va_list va; va_start(va, format); r = vmtagprintf(tag, format, va); va_end(va); return r; }
__forceinline char *mprintf(const char *format, ...) { char *r; va_list va; va_start(va, format); r = vmprintf(format, va); va_end(va); return r; }
__forceinline char *msnprintf(char *__restrict s, size_t maxlen, const char *format, ...) { char *r; va_list va; va_start(va, format); r = vmsnprintf(s, maxlen, format, va); va_end(va); return r; }
#else
STDARG1void(_log, _logv(errCode, format, va), int errCode, const char *format);
STDARG2void(_log, _logv(errCode, format, va), int errCode, const char *format);
STDARG3void(_log, _logv(errCode, format, va), int errCode, const char *format);
#if defined(_DEBUG) || defined(LIBCU_HAVE_OS_TRACE)
STDARG1void(_debug, _debugv(format, va), const char *format);
STDARG2void(_debug, _debugv(format, va), const char *format);
STDARG3void(_debug, _debugv(format, va), const char *format);
#endif
STDARG1(char *, mtagprintf, vmtagprintf(tag, format, va), tagbase_t *tag, const char *format);
STDARG2(char *, mtagprintf, vmtagprintf(tag, format, va), tagbase_t *tag, const char *format);
STDARG3(char *, mtagprintf, vmtagprintf(tag, format, va), tagbase_t *tag, const char *format);
STDARG1(char *, mprintf, vmprintf(format, va), const char *format);
STDARG2(char *, mprintf, vmprintf(format, va), const char *format);
STDARG3(char *, mprintf, vmprintf(format, va), const char *format);
STDARG1(char *, msnprintf, vmsnprintf(s, maxlen, format, va), char *__restrict s, size_t maxlen, const char *format);
STDARG2(char *, msnprintf, vmsnprintf(s, maxlen, format, va), char *__restrict s, size_t maxlen, const char *format);
STDARG3(char *, msnprintf, vmsnprintf(s, maxlen, format, va), char *__restrict s, size_t maxlen, const char *format);
#endif
#pragma endregion

#endif  /* _EXTGLOBAL__H */

