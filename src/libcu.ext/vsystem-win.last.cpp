#include <ext/global.h>
#include <string.h>
#include <assert.h>

#define LIBCU_OS_WIN 1

// OS WIN
#pragma region OS WIN
#if LIBCU_OS_WIN // This file is used for Windows only
#include <windows.h>
#include <new.h>

#pragma region Polyfill

/* Compiling and using WAL mode requires several APIs that are only available in Windows platforms based on the NT kernel. */
//#if !__OS_WINNT && !defined(LIBCU_OMIT_WAL)
//#error "WAL mode requires support from the Windows NT kernel, compile with LIBCU_OMIT_WAL."
//#endif
//
//#if !LIBCU_OS_WINNT && LIBCU_MAXMMAPSIZE > 0
//#error "Memory mapped files require support from the Windows NT kernel, compile with LIBCU_MAXMMAPSIZE=0."
//#endif

/* Are most of the Win32 ANSI APIs available (i.e. with certain exceptions based on the sub-platform)? */
#if !LIBCU_OS_WINCE && !LIBCU_OS_WINRT && !defined(LIBCU_WIN32_NO_ANSI)
#define LIBCU_WIN32_HAS_ANSI
#endif

/* Are most of the Win32 Unicode APIs available (i.e. with certain exceptions based on the sub-platform)? */
#if (LIBCU_OS_WINCE || LIBCU_OS_WINNT || LIBCU_OS_WINRT) && !defined(LIBCU_WIN32_NO_WIDE)
#define LIBCU_WIN32_HAS_WIDE
#endif

/* Make sure at least one set of Win32 APIs is available. */
#if !defined(LIBCU_WIN32_HAS_ANSI) && !defined(LIBCU_WIN32_HAS_WIDE)
#error "At least one of LIBCU_WIN32_HAS_ANSI and LIBCU_WIN32_HAS_WIDE must be defined."
#endif

/* Define the required Windows SDK version constants if they are not already available. */
#ifndef NTDDI_WIN8
#define NTDDI_WIN8                        0x06020000
#endif
#ifndef NTDDI_WINBLUE
#define NTDDI_WINBLUE                     0x06030000
#endif
#ifndef NTDDI_WINTHRESHOLD
#define NTDDI_WINTHRESHOLD                0x06040000
#endif

/* Check to see if the GetVersionEx[AW] functions are deprecated on the target system.  GetVersionEx was first deprecated in Win8.1. */
#ifndef LIBCU_WIN32_GETVERSIONEX
#if defined(NTDDI_VERSION) && NTDDI_VERSION >= NTDDI_WINBLUE
#define LIBCU_WIN32_GETVERSIONEX   0   // GetVersionEx() is deprecated
#else
#define LIBCU_WIN32_GETVERSIONEX   1   // GetVersionEx() is current
#endif
#endif

/*
** Check to see if the CreateFileMappingA function is supported on the target system.  It is unavailable when using "mincore.lib" on Win10.
** When compiling for Windows 10, always assume "mincore.lib" is in use.
*/
#ifndef LIBCU_WIN32_CREATEFILEMAPPINGA
#if defined(NTDDI_VERSION) && NTDDI_VERSION >= NTDDI_WINTHRESHOLD
#define LIBCU_WIN32_CREATEFILEMAPPINGA   0
#else
#define LIBCU_WIN32_CREATEFILEMAPPINGA   1
#endif
#endif

/* This constant should already be defined (in the "WinDef.h" SDK file). */
#ifndef MAX_PATH
#define MAX_PATH                      (260)
#endif

/* Maximum pathname length (in chars) for Win32.  This should normally be MAX_PATH. */
#ifndef LIBCU_WIN32_MAX_PATH_CHARS
#define LIBCU_WIN32_MAX_PATH_CHARS   (MAX_PATH)
#endif

/* This constant should already be defined (in the "WinNT.h" SDK file). */
#ifndef UNICODE_STRING_MAX_CHARS
#define UNICODE_STRING_MAX_CHARS      (32767)
#endif

/* Maximum pathname length (in chars) for WinNT.  This should normally be UNICODE_STRING_MAX_CHARS. */
#ifndef LIBCU_WINNT_MAX_PATH_CHARS
#define LIBCU_WINNT_MAX_PATH_CHARS   (UNICODE_STRING_MAX_CHARS)
#endif

/*
** Maximum pathname length (in bytes) for Win32.  The MAX_PATH macro is in characters, so we allocate 4 bytes per character assuming worst-case of
** 4-bytes-per-character for UTF8.
*/
#ifndef LIBCU_WIN32_MAX_PATH_BYTES
#define LIBCU_WIN32_MAX_PATH_BYTES   (LIBCU_WIN32_MAX_PATH_CHARS*4)
#endif

/* Maximum pathname length (in bytes) for WinNT.  This should normally be UNICODE_STRING_MAX_CHARS * sizeof(WCHAR). */
#ifndef LIBCU_WINNT_MAX_PATH_BYTES
#define LIBCU_WINNT_MAX_PATH_BYTES  (sizeof(WCHAR) * LIBCU_WINNT_MAX_PATH_CHARS)
#endif

/* Maximum error message length (in chars) for WinRT. */
#ifndef LIBCU_WIN32_MAX_ERRMSG_CHARS
#define LIBCU_WIN32_MAX_ERRMSG_CHARS (1024)
#endif

/* Returns non-zero if the character should be treated as a directory separator. */
#ifndef winIsDirSep
#define winIsDirSep(a)				((a) == '/' || (a) == '\\')
#endif

/* This macro is used when a local variable is set to a value that is [sometimes] not used by the code (e.g. via conditional compilation). */
#ifndef UNUSED_VARIABLE_VALUE
#define UNUSED_VARIABLE_VALUE(x)      (void)(x)
#endif

/* Returns the character that should be used as the directory separator. */
#ifndef winGetDirSep
#define winGetDirSep()                '\\'
#endif

/*
** Do we need to manually define the Win32 file mapping APIs for use with WAL mode or memory mapped files (e.g. these APIs are available in the Windows
** CE SDK; however, they are not present in the header file)?
*/
#if LIBCU_WIN32_FILEMAPPING_API && (!defined(LIBCU_OMIT_WAL) || LIBCU_MAXMMAPSIZE > 0)
// Two of the file mapping APIs are different under WinRT.  Figure out which set we need.
#if LIBCU_OS_WINRT
WINBASEAPI HANDLE WINAPI CreateFileMappingFromApp(HANDLE, LPSECURITY_ATTRIBUTES, ULONG, ULONG64, LPCWSTR);
WINBASEAPI LPVOID WINAPI MapViewOfFileFromApp(HANDLE, ULONG, ULONG64, SIZE_T);
#else
#if defined(LIBCU_WIN32_HAS_ANSI)
WINBASEAPI HANDLE WINAPI CreateFileMappingA(HANDLE, LPSECURITY_ATTRIBUTES, DWORD, DWORD, DWORD, LPCSTR);
#endif
#if defined(LIBCU_WIN32_HAS_WIDE)
WINBASEAPI HANDLE WINAPI CreateFileMappingW(HANDLE, LPSECURITY_ATTRIBUTES, DWORD, DWORD, DWORD, LPCWSTR);
#endif
WINBASEAPI LPVOID WINAPI MapViewOfFile(HANDLE, DWORD, DWORD, DWORD, SIZE_T);
#endif
// These file mapping APIs are common to both Win32 and WinRT.
WINBASEAPI BOOL WINAPI FlushViewOfFile(LPCVOID, SIZE_T);
WINBASEAPI BOOL WINAPI UnmapViewOfFile(LPCVOID);
#endif /* LIBCU_WIN32_FILEMAPPING_API */

/* Some Microsoft compilers lack this definition. */
#ifndef INVALID_FILE_ATTRIBUTES
#define INVALID_FILE_ATTRIBUTES ((DWORD)-1)
#endif

#ifndef FILE_FLAG_MASK
#define FILE_FLAG_MASK          (0xFF3C0000)
#endif

#ifndef FILE_ATTRIBUTE_MASK
#define FILE_ATTRIBUTE_MASK     (0x0003FFF7)
#endif

#ifndef LIBCU_OMIT_WAL
/* Forward references to structures used for WAL */
typedef struct winShm winShm;           // A connection to shared-memory
typedef struct winShmNode winShmNode;   // A region of shared-memory
#endif

/* WinCE lacks native support for file locking so we have to fake it with some code of our own. */
#if LIBCU_OS_WINCE
typedef struct winceLock {
	int readers;       // Number of reader locks obtained
	BOOL pending;      // Indicates a pending lock has been obtained
	BOOL reserved;     // Indicates a reserved lock has been obtained
	BOOL exclusive;    // Indicates an exclusive lock has been obtained
} winceLock;
#endif

/* The winFile structure is a subclass of vsysfile* specific to the win32 portability layer. */
typedef struct winFile winFile;
struct winFile {
	const vsysfile_methods *filesystem; /*** Must be first ***/
	vsystem *vsystem;		// The VFS used to open this file
	HANDLE h;               // Handle for accessing the file
	uint8_t locktype;       // Type of lock currently held on this file
	short sharedLockByte;	// Randomly chosen byte used as a shared lock
	uint8_t ctrlFlags;		// Flags.  See WINFILE_* below
	DWORD lastErrno;		// The Windows errno from the last I/O error
#ifndef LIB_OMIT_WAL
	winShm *shm;			// Instance of shared memory on this file
#endif
	const char *path;		// Full pathname of this file
	int sizeChunk;          // Chunk size configured by FCNTL_CHUNK_SIZE
#if LIBCU_OS_WINCE
	LPWSTR deleteOnClose;	// Name of file to delete when closing
	HANDLE hMutex;          // Mutex used to control access to shared lock
	HANDLE hShared;         // Shared memory segment used for locking
	winceLock local;        // Locks obtained by this instance of winFile
	winceLock *shared;      // Global shared lock memory for the file
#endif
#if LIBCU_MAXMMAPSIZE > 0
	int fetchOuts;          // Number of outstanding _.fetch references
	HANDLE hMap;            // Handle for accessing memory mapping
	void *mapRegion;        // Area memory mapped
	int64_t mmapSize;       // Usable size of mapped region
	int64_t mmapSizeActual; // Actual size of mapped region
	int64_t mmapSizeMax;    // Configured FCNTL_MMAP_SIZE value
#endif
};

/* The winVfsAppData structure is used for the pAppData member for all of the Win32 VFS variants. */
typedef struct winVfsAppData winVfsAppData;
struct winVfsAppData {
	const vsysfile_methods *filesystem;		// The file I/O methods to use.
	void *appData;			// The extra pAppData, if any.
	BOOL noLock;			// Non-zero if locking is disabled.
};

/* Allowed values for winFile.ctrlFlags */
#define WINFILE_RDONLY          0x02   // Connection is read only
#define WINFILE_PERSIST_WAL     0x04   // Persistent WAL mode
#define WINFILE_PSOW            0x10   // LIBCU_IOCAP_POWERSAFE_OVERWRITE

/* The size of the buffer used by sqlite3_win32_write_debug(). */
#ifndef LIBCU_WIN32_DBG_BUF_SIZE
#define LIBCU_WIN32_DBG_BUF_SIZE   ((int)(4096-sizeof(DWORD)))
#endif

/* The value used with sqlite3_win32_set_directory() to specify that the data directory should be changed. */
#ifndef LIBCU_WIN32_DATA_DIRECTORY_TYPE
#define LIBCU_WIN32_DATA_DIRECTORY_TYPE (1)
#endif

/* The value used with sqlite3_win32_set_directory() to specify that the temporary directory should be changed. */
#ifndef LIBCU_WIN32_TEMP_DIRECTORY_TYPE
#define LIBCU_WIN32_TEMP_DIRECTORY_TYPE (2)
#endif

#pragma endregion

#pragma region Malloc1

/* If compiled with LIBCU_WIN32_MALLOC on Windows, we will use the various Win32 API heap functions instead of our own. */
#ifdef LIBCU_WIN32_MALLOC

/*
* If this is non-zero, an isolated heap will be created by the native Win32 allocator subsystem; otherwise, the default process heap will be used.  This
* setting has no effect when compiling for WinRT.  By default, this is enabled and an isolated heap will be created to store all allocated data.
*
******************************************************************************
* WARNING: It is important to note that when this setting is non-zero and the winMemShutdown function is called (e.g. by the sqlite3_shutdown
*          function), all data that was allocated using the isolated heap will be freed immediately and any attempt to access any of that freed
*          data will almost certainly result in an immediate access violation.
******************************************************************************
*/
#ifndef LIBCU_WIN32_HEAP_CREATE
#define LIBCU_WIN32_HEAP_CREATE        (TRUE)
#endif

/* This is the maximum possible initial size of the Win32-specific heap, in bytes. */
#ifndef LIBCU_WIN32_HEAP_MAX_INIT_SIZE
#define LIBCU_WIN32_HEAP_MAX_INIT_SIZE (4294967295U)
#endif

/* This is the extra space for the initial size of the Win32-specific heap, in bytes.  This value may be zero. */
#ifndef LIBCU_WIN32_HEAP_INIT_EXTRA
#define LIBCU_WIN32_HEAP_INIT_EXTRA  (4194304)
#endif

/* Calculate the maximum legal cache size, in pages, based on the maximum possible initial heap size and the default page size, setting aside the needed extra space. */
#ifndef LIBCU_WIN32_MAX_CACHE_SIZE
#define LIBCU_WIN32_MAX_CACHE_SIZE   (((LIBCU_WIN32_HEAP_MAX_INIT_SIZE) - (LIBCU_WIN32_HEAP_INIT_EXTRA)) / (LIBCU_DEFAULT_PAGE_SIZE))
#endif

/* This is cache size used in the calculation of the initial size of the Win32-specific heap.  It cannot be negative. */
#ifndef LIBCU_WIN32_CACHE_SIZE
#if LIBCU_DEFAULT_CACHE_SIZE>=0
#define LIBCU_WIN32_CACHE_SIZE     (LIBCU_DEFAULT_CACHE_SIZE)
#else
#define LIBCU_WIN32_CACHE_SIZE     (-(LIBCU_DEFAULT_CACHE_SIZE))
#endif
#endif

/*
* Make sure that the calculated cache size, in pages, cannot cause the initial size of the Win32-specific heap to exceed the maximum amount
* of memory that can be specified in the call to HeapCreate.
*/
#if LIBCU_WIN32_CACHE_SIZE > LIBCU_WIN32_MAX_CACHE_SIZE
#undef LIBCU_WIN32_CACHE_SIZE
#define LIBCU_WIN32_CACHE_SIZE       (2000)
#endif

/* The initial size of the Win32-specific heap.  This value may be zero. */
#ifndef LIBCU_WIN32_HEAP_INIT_SIZE
#define LIBCU_WIN32_HEAP_INIT_SIZE   ((LIBCU_WIN32_CACHE_SIZE) * (LIBCU_DEFAULT_PAGE_SIZE) + (LIBCU_WIN32_HEAP_INIT_EXTRA))
#endif

/* The maximum size of the Win32-specific heap.  This value may be zero. */
#ifndef LIBCU_WIN32_HEAP_MAX_SIZE
#define LIBCU_WIN32_HEAP_MAX_SIZE    (0)
#endif

/* The extra flags to use in calls to the Win32 heap APIs.  This value may be zero for the default behavior. */
#ifndef LIBCU_WIN32_HEAP_FLAGS
#define LIBCU_WIN32_HEAP_FLAGS       (0)
#endif

/* The winMemData structure stores information required by the Win32-specific sqlite3_mem_methods implementation. */
typedef struct winMemData winMemData;
struct WinMemData {
#ifndef NDEBUG
	uint32_t magic1;	// Magic number to detect structure corruption.
#endif
	HANDLE heap;		// The handle to our heap.
	BOOL owned;			// Do we own the heap (i.e. destroy it on shutdown)?
#ifndef NDEBUG
	uint32_t magic2;	// Magic number to detect structure corruption.
#endif
};

#ifndef NDEBUG
#define WINMEM_MAGIC1     0x42b2830b
#define WINMEM_MAGIC2     0xbd4d7cf4
#endif

static struct WinMemData win_mem_data = {
#ifndef NDEBUG
	WINMEM_MAGIC1,
#endif
	NULL, FALSE
#ifndef NDEBUG
	,WINMEM_MAGIC2
#endif
};

#ifndef NDEBUG
#define winMemAssertMagic1() assert(win_mem_data.magic1 == WINMEM_MAGIC1)
#define winMemAssertMagic2() assert(win_mem_data.magic2 == WINMEM_MAGIC2)
#define winMemAssertMagic()  winMemAssertMagic1(); winMemAssertMagic2();
#else
#define winMemAssertMagic()
#endif

#define winMemGetDataPtr()  &win_mem_data
#define winMemGetHeap()		win_mem_data.heap
#define winMemGetOwned()    win_mem_data.owned

static void *winMemMalloc(int size);
static void winMemFree(void *prior);
static void *winMemRealloc(void *prior, int newSize);
static int winMemSize(void *p);
static int winMemRoundup(int size);
static RC winMemInitialize(void *appData);
static RC winMemShutdown(void *appData);

const allloc_methods *sqlite3MemGetWin32();
#endif /* LIBCU_WIN32_MALLOC */

#pragma endregion

#pragma region Syscall

#define LIBCU_WIN32_VOLATILE volatile

/*
** The following variable is (normally) set once and never changes thereafter.  It records whether the operating system is Win9x or WinNT.
**
** 0:   Operating system unknown.
** 1:   Operating system is Win9x.
** 2:   Operating system is WinNT.
**
** In order to facilitate testing on a WinNT system, the test fixture can manually set this value to 1 to emulate Win98 behavior.
*/
#ifdef LIBCU_TEST
LONG LIBCU_WIN32_VOLATILE _os_type = 0;
#else
static LONG LIBCU_WIN32_VOLATILE _os_type = 0;
#endif

#ifndef SYSCALL
#define SYSCALL vsystemcall_ptr
#endif

/* This function is not available on Windows CE or WinRT. */
#if LIBCU_OS_WINCE || LIBCU_OS_WINRT
#define osAreFileApisANSI()       1
#endif

/*
** Many system calls are accessed through pointer-to-functions so that they may be overridden at runtime to facilitate fault injection during
** testing and sandboxing.  The following array holds the names and pointers to all overrideable system calls.
*/
static struct win_syscall {
	const char *name;           // Name of the system call
	vsystemcall_ptr current;		// Current value of the system call
	vsystemcall_ptr default;		// Default value
} Syscalls[] = {
#if !LIBCU_OS_WINCE && !LIBCU_OS_WINRT
	{ "AreFileApisANSI",         (SYSCALL)AreFileApisANSI, nullptr },
#else
	{ "AreFileApisANSI",         (SYSCALL)nullptr, nullptr },
#endif
#ifndef osAreFileApisANSI
#define osAreFileApisANSI ((BOOL(WINAPI*)(VOID))Syscalls[0].current)
#endif
#if LIBCU_OS_WINCE && defined(LIBCU_WIN32_HAS_WIDE)
	{ "CharLowerW",              (SYSCALL)CharLowerW, nullptr },
#else
	{ "CharLowerW",              (SYSCALL)nullptr, nullptr },
#endif
#define osCharLowerW ((LPWSTR(WINAPI*)(LPWSTR))Syscalls[1].current)
#if LIBCU_OS_WINCE && defined(LIBCU_WIN32_HAS_WIDE)
	{ "CharUpperW",              (SYSCALL)CharUpperW, nullptr },
#else
	{ "CharUpperW",              (SYSCALL)nullptr, nullptr },
#endif
#define osCharUpperW ((LPWSTR(WINAPI*)(LPWSTR))Syscalls[2].current)
	{ "CloseHandle",             (SYSCALL)CloseHandle, nullptr },
#define osCloseHandle ((BOOL(WINAPI*)(HANDLE))Syscalls[3].current)
#if defined(LIBCU_WIN32_HAS_ANSI)
	{ "CreateFileA",             (SYSCALL)CreateFileA, nullptr },
#else
	{ "CreateFileA",             (SYSCALL)nullptr, nullptr },
#endif
#define osCreateFileA ((HANDLE(WINAPI*)(LPCSTR,DWORD,DWORD,LPSECURITY_ATTRIBUTES,DWORD,DWORD,HANDLE))Syscalls[4].current)
#if !LIBCU_OS_WINRT && defined(LIBCU_WIN32_HAS_WIDE)
	{ "CreateFileW",             (SYSCALL)CreateFileW, nullptr },
#else
	{ "CreateFileW",             (SYSCALL)nullptr, nullptr },
#endif
#define osCreateFileW ((HANDLE(WINAPI*)(LPCWSTR,DWORD,DWORD,LPSECURITY_ATTRIBUTES,DWORD,DWORD,HANDLE))Syscalls[5].current)
#if !LIBCU_OS_WINRT && defined(LIBCU_WIN32_HAS_ANSI) && (!defined(LIBCU_OMIT_WAL) || LIBCU_MAXMMAPSIZE > 0) && LIBCU_WIN32_CREATEFILEMAPPINGA
	{ "CreateFileMappingA",      (SYSCALL)CreateFileMappingA, nullptr },
#else
	{ "CreateFileMappingA",      (SYSCALL)nullptr, nullptr },
#endif
#define osCreateFileMappingA ((HANDLE(WINAPI*)(HANDLE,LPSECURITY_ATTRIBUTES,DWORD,DWORD,DWORD,LPCSTR))Syscalls[6].current)
#if LIBCU_OS_WINCE || (!LIBCU_OS_WINRT && defined(LIBCU_WIN32_HAS_WIDE) && (!defined(LIBCU_OMIT_WAL) || LIBCU_MAXMMAPSIZE > 0))
	{ "CreateFileMappingW",      (SYSCALL)CreateFileMappingW, nullptr },
#else
	{ "CreateFileMappingW",      (SYSCALL)nullptr, nullptr },
#endif
#define osCreateFileMappingW ((HANDLE(WINAPI*)(HANDLE,LPSECURITY_ATTRIBUTES,DWORD,DWORD,DWORD,LPCWSTR))Syscalls[7].current)
#if !LIBCU_OS_WINRT && defined(LIBCU_WIN32_HAS_WIDE)
	{ "CreateMutexW",            (SYSCALL)CreateMutexW, nullptr },
#else
	{ "CreateMutexW",            (SYSCALL)nullptr, nullptr },
#endif
#define osCreateMutexW ((HANDLE(WINAPI*)(LPSECURITY_ATTRIBUTES,BOOL,LPCWSTR))Syscalls[8].current)
#if defined(LIBCU_WIN32_HAS_ANSI)
	{ "DeleteFileA",             (SYSCALL)DeleteFileA, nullptr },
#else
	{ "DeleteFileA",             (SYSCALL)nullptr, nullptr },
#endif
#define osDeleteFileA ((BOOL(WINAPI*)(LPCSTR))Syscalls[9].current)
#if defined(LIBCU_WIN32_HAS_WIDE)
	{ "DeleteFileW",             (SYSCALL)DeleteFileW, nullptr },
#else
	{ "DeleteFileW",             (SYSCALL)nullptr, nullptr },
#endif
#define osDeleteFileW ((BOOL(WINAPI*)(LPCWSTR))Syscalls[10].current)
#if LIBCU_OS_WINCE
	{ "FileTimeToLocalFileTime", (SYSCALL)FileTimeToLocalFileTime, nullptr },
#else
	{ "FileTimeToLocalFileTime", (SYSCALL)nullptr, nullptr },
#endif
#define osFileTimeToLocalFileTime ((BOOL(WINAPI*)(CONST FILETIME*,LPFILETIME))Syscalls[11].current)
#if LIBCU_OS_WINCE
	{ "FileTimeToSystemTime",    (SYSCALL)FileTimeToSystemTime, nullptr },
#else
	{ "FileTimeToSystemTime",    (SYSCALL)nullptr, nullptr },
#endif
#define osFileTimeToSystemTime ((BOOL(WINAPI*)(CONST FILETIME*,LPSYSTEMTIME))Syscalls[12].current)
	{ "FlushFileBuffers",        (SYSCALL)FlushFileBuffers, nullptr },
#define osFlushFileBuffers ((BOOL(WINAPI*)(HANDLE))Syscalls[13].current)
#if defined(LIBCU_WIN32_HAS_ANSI)
	{ "FormatMessageA",          (SYSCALL)FormatMessageA, nullptr },
#else
	{ "FormatMessageA",          (SYSCALL)nullptr, nullptr },
#endif
#define osFormatMessageA ((DWORD(WINAPI*)(DWORD,LPCVOID,DWORD,DWORD,LPSTR,DWORD,va_list*))Syscalls[14].current)
#if defined(LIBCU_WIN32_HAS_WIDE)
	{ "FormatMessageW",          (SYSCALL)FormatMessageW, nullptr },
#else
	{ "FormatMessageW",          (SYSCALL)nullptr, nullptr },
#endif
#define osFormatMessageW ((DWORD(WINAPI*)(DWORD,LPCVOID,DWORD,DWORD,LPWSTR,DWORD,va_list*))Syscalls[15].current)
#if !defined(LIBCU_OMIT_LOAD_EXTENSION)
	{ "FreeLibrary",             (SYSCALL)FreeLibrary, nullptr },
#else
	{ "FreeLibrary",             (SYSCALL)nullptr, nullptr },
#endif
#define osFreeLibrary ((BOOL(WINAPI*)(HMODULE))Syscalls[16].current)
	{ "GetCurrentProcessId",     (SYSCALL)GetCurrentProcessId, nullptr },
#define osGetCurrentProcessId ((DWORD(WINAPI*)(VOID))Syscalls[17].current)
#if !LIBCU_OS_WINCE && defined(LIBCU_WIN32_HAS_ANSI)
	{ "GetDiskFreeSpaceA",       (SYSCALL)GetDiskFreeSpaceA, nullptr },
#else
	{ "GetDiskFreeSpaceA",       (SYSCALL)nullptr, nullptr },
#endif
#define osGetDiskFreeSpaceA ((BOOL(WINAPI*)(LPCSTR,LPDWORD,LPDWORD,LPDWORD,LPDWORD))Syscalls[18].current)
#if !LIBCU_OS_WINCE && !LIBCU_OS_WINRT && defined(LIBCU_WIN32_HAS_WIDE)
	{ "GetDiskFreeSpaceW",       (SYSCALL)GetDiskFreeSpaceW, nullptr },
#else
	{ "GetDiskFreeSpaceW",       (SYSCALL)nullptr, nullptr },
#endif
#define osGetDiskFreeSpaceW ((BOOL(WINAPI*)(LPCWSTR,LPDWORD,LPDWORD,LPDWORD,LPDWORD))Syscalls[19].current)
#if defined(LIBCU_WIN32_HAS_ANSI)
	{ "GetFileAttributesA",      (SYSCALL)GetFileAttributesA, nullptr },
#else
	{ "GetFileAttributesA",      (SYSCALL)nullptr, nullptr },
#endif
#define osGetFileAttributesA ((DWORD(WINAPI*)(LPCSTR))Syscalls[20].current)
#if !LIBCU_OS_WINRT && defined(LIBCU_WIN32_HAS_WIDE)
	{ "GetFileAttributesW",      (SYSCALL)GetFileAttributesW, nullptr },
#else
	{ "GetFileAttributesW",      (SYSCALL)nullptr, nullptr },
#endif
#define osGetFileAttributesW ((DWORD(WINAPI*)(LPCWSTR))Syscalls[21].current)
#if defined(LIBCU_WIN32_HAS_WIDE)
	{ "GetFileAttributesExW",    (SYSCALL)GetFileAttributesExW, nullptr },
#else
	{ "GetFileAttributesExW",    (SYSCALL)nullptr, nullptr },
#endif
#define osGetFileAttributesExW ((BOOL(WINAPI*)(LPCWSTR,GET_FILEEX_INFO_LEVELS,LPVOID))Syscalls[22].current)
#if !LIBCU_OS_WINRT
	{ "GetFileSize",             (SYSCALL)GetFileSize, nullptr },
#else
	{ "GetFileSize",             (SYSCALL)nullptr, nullptr },
#endif
#define osGetFileSize ((DWORD(WINAPI*)(HANDLE,LPDWORD))Syscalls[23].current)
#if !LIBCU_OS_WINCE && defined(LIBCU_WIN32_HAS_ANSI)
	{ "GetFullPathNameA",        (SYSCALL)GetFullPathNameA, nullptr },
#else
	{ "GetFullPathNameA",        (SYSCALL)nullptr, nullptr },
#endif
#define osGetFullPathNameA ((DWORD(WINAPI*)(LPCSTR,DWORD,LPSTR,LPSTR*))Syscalls[24].current)
#if !LIBCU_OS_WINCE && !LIBCU_OS_WINRT && defined(LIBCU_WIN32_HAS_WIDE)
	{ "GetFullPathNameW",        (SYSCALL)GetFullPathNameW, nullptr },
#else
	{ "GetFullPathNameW",        (SYSCALL)nullptr, nullptr },
#endif
#define osGetFullPathNameW ((DWORD(WINAPI*)(LPCWSTR,DWORD,LPWSTR,LPWSTR*))Syscalls[25].current)
	{ "GetLastError",            (SYSCALL)GetLastError, nullptr },
#define osGetLastError ((DWORD(WINAPI*)(VOID))Syscalls[26].current)
#if !defined(LIBCU_OMIT_LOAD_EXTENSION)
#if LIBCU_OS_WINCE
	// The GetProcAddressA() routine is only available on Windows CE.
	{ "GetProcAddressA",         (SYSCALL)GetProcAddressA, nullptr },
#else
	// All other Windows platforms expect GetProcAddress() to take an ANSI string regardless of the _UNICODE setting
	{ "GetProcAddressA",         (SYSCALL)GetProcAddress, nullptr },
#endif
#else
	{ "GetProcAddressA",         (SYSCALL)nullptr, nullptr },
#endif
#define osGetProcAddressA ((FARPROC(WINAPI*)(HMODULE,LPCSTR))Syscalls[27].current)
#if !LIBCU_OS_WINRT
	{ "GetSystemInfo",           (SYSCALL)GetSystemInfo, nullptr },
#else
	{ "GetSystemInfo",           (SYSCALL)nullptr, nullptr },
#endif
#define osGetSystemInfo ((VOID(WINAPI*)(LPSYSTEM_INFO))Syscalls[28].current)
	{ "GetSystemTime",           (SYSCALL)GetSystemTime, nullptr },
#define osGetSystemTime ((VOID(WINAPI*)(LPSYSTEMTIME))Syscalls[29].current)
#if !LIBCU_OS_WINCE
	{ "GetSystemTimeAsFileTime", (SYSCALL)GetSystemTimeAsFileTime, nullptr },
#else
	{ "GetSystemTimeAsFileTime", (SYSCALL)nullptr, nullptr },
#endif
#define osGetSystemTimeAsFileTime ((VOID(WINAPI*)(LPFILETIME))Syscalls[30].current)
#if defined(LIBCU_WIN32_HAS_ANSI)
	{ "GetTempPathA",            (SYSCALL)GetTempPathA, nullptr },
#else
	{ "GetTempPathA",            (SYSCALL)nullptr, nullptr },
#endif
#define osGetTempPathA ((DWORD(WINAPI*)(DWORD,LPSTR))Syscalls[31].current)
#if !LIBCU_OS_WINRT && defined(LIBCU_WIN32_HAS_WIDE)
	{ "GetTempPathW",            (SYSCALL)GetTempPathW, nullptr },
#else
	{ "GetTempPathW",            (SYSCALL)nullptr, nullptr },
#endif
#define osGetTempPathW ((DWORD(WINAPI*)(DWORD,LPWSTR))Syscalls[32].current)
#if !LIBCU_OS_WINRT
	{ "GetTickCount",            (SYSCALL)GetTickCount, nullptr },
#else
	{ "GetTickCount",            (SYSCALL)nullptr, nullptr },
#endif
#define osGetTickCount ((DWORD(WINAPI*)(VOID))Syscalls[33].current)
#if defined(LIBCU_WIN32_HAS_ANSI) && LIBCU_WIN32_GETVERSIONEX
	{ "GetVersionExA",           (SYSCALL)GetVersionExA, nullptr },
#else
	{ "GetVersionExA",           (SYSCALL)nullptr, nullptr },
#endif
#define osGetVersionExA ((BOOL(WINAPI*)(LPOSVERSIONINFOA))Syscalls[34].current)
#if !LIBCU_OS_WINRT && defined(LIBCU_WIN32_HAS_WIDE) && LIBCU_WIN32_GETVERSIONEX
	{ "GetVersionExW",           (SYSCALL)GetVersionExW, nullptr },
#else
	{ "GetVersionExW",           (SYSCALL)nullptr, nullptr },
#endif
#define osGetVersionExW ((BOOL(WINAPI*)(LPOSVERSIONINFOW))Syscalls[35].current)
	{ "HeapAlloc",               (SYSCALL)HeapAlloc, nullptr },
#define osHeapAlloc ((LPVOID(WINAPI*)(HANDLE,DWORD,SIZE_T))Syscalls[36].current)
#if !LIBCU_OS_WINRT
	{ "HeapCreate",              (SYSCALL)HeapCreate, nullptr },
#else
	{ "HeapCreate",              (SYSCALL)nullptr, nullptr },
#endif
#define osHeapCreate ((HANDLE(WINAPI*)(DWORD,SIZE_T,SIZE_T))Syscalls[37].current)
#if !LIBCU_OS_WINRT
	{ "HeapDestroy",             (SYSCALL)HeapDestroy, nullptr },
#else
	{ "HeapDestroy",             (SYSCALL)nullptr, nullptr },
#endif
#define osHeapDestroy ((BOOL(WINAPI*)(HANDLE))Syscalls[38].current)
	{ "HeapFree",                (SYSCALL)HeapFree, nullptr },
#define osHeapFree ((BOOL(WINAPI*)(HANDLE,DWORD,LPVOID))Syscalls[39].current)
	{ "HeapReAlloc",             (SYSCALL)HeapReAlloc, nullptr },
#define osHeapReAlloc ((LPVOID(WINAPI*)(HANDLE,DWORD,LPVOID,SIZE_T))Syscalls[40].current)
	{ "HeapSize",                (SYSCALL)HeapSize, nullptr },
#define osHeapSize ((SIZE_T(WINAPI*)(HANDLE,DWORD,LPCVOID))Syscalls[41].current)
#if !LIBCU_OS_WINRT
	{ "HeapValidate",            (SYSCALL)HeapValidate, nullptr },
#else
	{ "HeapValidate",            (SYSCALL)nullptr, nullptr },
#endif
#define osHeapValidate ((BOOL(WINAPI*)(HANDLE,DWORD,LPCVOID))Syscalls[42].current)
#if !LIBCU_OS_WINCE && !LIBCU_OS_WINRT
	{ "HeapCompact",             (SYSCALL)HeapCompact, nullptr },
#else
	{ "HeapCompact",             (SYSCALL)nullptr, nullptr },
#endif
#define osHeapCompact ((UINT(WINAPI*)(HANDLE,DWORD))Syscalls[43].current)
#if defined(LIBCU_WIN32_HAS_ANSI) && !defined(LIBCU_OMIT_LOAD_EXTENSION)
	{ "LoadLibraryA",            (SYSCALL)LoadLibraryA, nullptr },
#else
	{ "LoadLibraryA",            (SYSCALL)nullptr, nullptr },
#endif
#define osLoadLibraryA ((HMODULE(WINAPI*)(LPCSTR))Syscalls[44].current)
#if !LIBCU_OS_WINRT && defined(LIBCU_WIN32_HAS_WIDE) && !defined(LIBCU_OMIT_LOAD_EXTENSION)
	{ "LoadLibraryW",            (SYSCALL)LoadLibraryW, nullptr },
#else
	{ "LoadLibraryW",            (SYSCALL)nullptr, nullptr },
#endif
#define osLoadLibraryW ((HMODULE(WINAPI*)(LPCWSTR))Syscalls[45].current)
#if !LIBCU_OS_WINRT
	{ "LocalFree",               (SYSCALL)LocalFree, nullptr },
#else
	{ "LocalFree",               (SYSCALL)nullptr, nullptr },
#endif
#define osLocalFree ((HLOCAL(WINAPI*)(HLOCAL))Syscalls[46].current)
#if !LIBCU_OS_WINCE && !LIBCU_OS_WINRT
	{ "LockFile",                (SYSCALL)LockFile, nullptr },
#else
	{ "LockFile",                (SYSCALL)nullptr, nullptr },
#endif
#ifndef osLockFile
#define osLockFile ((BOOL(WINAPI*)(HANDLE,DWORD,DWORD,DWORD,DWORD))Syscalls[47].current)
#endif
#if !LIBCU_OS_WINCE
	{ "LockFileEx",              (SYSCALL)LockFileEx, nullptr },
#else
	{ "LockFileEx",              (SYSCALL)nullptr, nullptr },
#endif
#ifndef osLockFileEx
#define osLockFileEx ((BOOL(WINAPI*)(HANDLE,DWORD,DWORD,DWORD,DWORD,LPOVERLAPPED))Syscalls[48].current)
#endif
#if LIBCU_OS_WINCE || (!LIBCU_OS_WINRT && (!defined(LIBCU_OMIT_WAL) || LIBCU_MAXMMAPSIZE > 0))
	{ "MapViewOfFile",           (SYSCALL)MapViewOfFile, nullptr },
#else
	{ "MapViewOfFile",           (SYSCALL)nullptr, nullptr },
#endif
#define osMapViewOfFile ((LPVOID(WINAPI*)(HANDLE,DWORD,DWORD,DWORD,SIZE_T))Syscalls[49].current)
	{ "MultiByteToWideChar",     (SYSCALL)MultiByteToWideChar, nullptr },
#define osMultiByteToWideChar ((int(WINAPI*)(UINT,DWORD,LPCSTR,int,LPWSTR,int))Syscalls[50].current)
	{ "QueryPerformanceCounter", (SYSCALL)QueryPerformanceCounter, nullptr },
#define osQueryPerformanceCounter ((BOOL(WINAPI*)(LARGE_INTEGER*))Syscalls[51].current)
	{ "ReadFile",                (SYSCALL)ReadFile,  nullptr },
#define osReadFile ((BOOL(WINAPI*)(HANDLE,LPVOID,DWORD,LPDWORD,LPOVERLAPPED))Syscalls[52].current)
	{ "SetEndOfFile",            (SYSCALL)SetEndOfFile, nullptr },
#define osSetEndOfFile ((BOOL(WINAPI*)(HANDLE))Syscalls[53].current)
#if !LIBCU_OS_WINRT
	{ "SetFilePointer",          (SYSCALL)SetFilePointer, nullptr },
#else
	{ "SetFilePointer",          (SYSCALL)nullptr, nullptr },
#endif
#define osSetFilePointer ((DWORD(WINAPI*)(HANDLE,LONG,PLONG,DWORD))Syscalls[54].current)
#if !LIBCU_OS_WINRT
	{ "Sleep",                   (SYSCALL)Sleep, nullptr },
#else
	{ "Sleep",                   (SYSCALL)nullptr, nullptr },
#endif
#define osSleep ((VOID(WINAPI*)(DWORD))Syscalls[55].current)
	{ "SystemTimeToFileTime",    (SYSCALL)SystemTimeToFileTime, nullptr },
#define osSystemTimeToFileTime ((BOOL(WINAPI*)(CONST SYSTEMTIME*,LPFILETIME))Syscalls[56].current)
#if !LIBCU_OS_WINCE && !LIBCU_OS_WINRT
	{ "UnlockFile",              (SYSCALL)UnlockFile, nullptr },
#else
	{ "UnlockFile",              (SYSCALL)nullptr, nullptr },
#endif
#ifndef osUnlockFile
#define osUnlockFile ((BOOL(WINAPI*)(HANDLE,DWORD,DWORD,DWORD,DWORD))Syscalls[57].current)
#endif
#if !LIBCU_OS_WINCE
	{ "UnlockFileEx",            (SYSCALL)UnlockFileEx, nullptr },
#else
	{ "UnlockFileEx",            (SYSCALL)nullptr, nullptr },
#endif
#define osUnlockFileEx ((BOOL(WINAPI*)(HANDLE,DWORD,DWORD,DWORD,LPOVERLAPPED))Syscalls[58].current)
#if LIBCU_OS_WINCE || !defined(LIBCU_OMIT_WAL) || LIBCU_MAXMMAPSIZE>0
	{ "UnmapViewOfFile",         (SYSCALL)UnmapViewOfFile, nullptr },
#else
	{ "UnmapViewOfFile",         (SYSCALL)nullptr, nullptr },
#endif
#define osUnmapViewOfFile ((BOOL(WINAPI*)(LPCVOID))Syscalls[59].current)
	{ "WideCharToMultiByte",     (SYSCALL)WideCharToMultiByte, nullptr },
#define osWideCharToMultiByte ((int(WINAPI*)(UINT,DWORD,LPCWSTR,int,LPSTR,int,LPCSTR,LPBOOL))Syscalls[60].current)
	{ "WriteFile",               (SYSCALL)WriteFile, nullptr },
#define osWriteFile ((BOOL(WINAPI*)(HANDLE,LPCVOID,DWORD,LPDWORD,LPOVERLAPPED))Syscalls[61].current)
#if LIBCU_OS_WINRT
	{ "CreateEventExW",          (SYSCALL)CreateEventExW, nullptr },
#else
	{ "CreateEventExW",          (SYSCALL)nullptr, nullptr },
#endif
#define osCreateEventExW ((HANDLE(WINAPI*)(LPSECURITY_ATTRIBUTES,LPCWSTR,DWORD,DWORD))Syscalls[62].current)
#if !LIBCU_OS_WINRT
	{ "WaitForSingleObject",     (SYSCALL)WaitForSingleObject, nullptr },
#else
	{ "WaitForSingleObject",     (SYSCALL)nullptr, nullptr },
#endif
#define osWaitForSingleObject ((DWORD(WINAPI*)(HANDLE,DWORD))Syscalls[63].current)
#if !LIBCU_OS_WINCE
	{ "WaitForSingleObjectEx",   (SYSCALL)WaitForSingleObjectEx, nullptr },
#else
	{ "WaitForSingleObjectEx",   (SYSCALL)nullptr, nullptr },
#endif
#define osWaitForSingleObjectEx ((DWORD(WINAPI*)(HANDLE,DWORD,BOOL))Syscalls[64].current)
#if LIBCU_OS_WINRT
	{ "SetFilePointerEx",        (SYSCALL)SetFilePointerEx, nullptr },
#else
	{ "SetFilePointerEx",        (SYSCALL)nullptr, nullptr },
#endif
#define osSetFilePointerEx ((BOOL(WINAPI*)(HANDLE,LARGE_INTEGER,PLARGE_INTEGER,DWORD))Syscalls[65].current)
#if LIBCU_OS_WINRT
	{ "GetFileInformationByHandleEx", (SYSCALL)GetFileInformationByHandleEx, nullptr },
#else
	{ "GetFileInformationByHandleEx", (SYSCALL)nullptr, nullptr },
#endif
#define osGetFileInformationByHandleEx ((BOOL(WINAPI*)(HANDLE,FILE_INFO_BY_HANDLE_CLASS,LPVOID,DWORD))Syscalls[66].current)
#if LIBCU_OS_WINRT && (!defined(LIBCU_OMIT_WAL) || LIBCU_MAXMMAPSIZE>0)
	{ "MapViewOfFileFromApp",    (SYSCALL)MapViewOfFileFromApp, nullptr },
#else
	{ "MapViewOfFileFromApp",    (SYSCALL)nullptr, nullptr },
#endif
#define osMapViewOfFileFromApp ((LPVOID(WINAPI*)(HANDLE,ULONG,ULONG64,SIZE_T))Syscalls[67].current)
#if LIBCU_OS_WINRT
	{ "CreateFile2",             (SYSCALL)CreateFile2, nullptr },
#else
	{ "CreateFile2",             (SYSCALL)nullptr, nullptr },
#endif
#define osCreateFile2 ((HANDLE(WINAPI*)(LPCWSTR,DWORD,DWORD,DWORD,LPCREATEFILE2_EXTENDED_PARAMETERS))Syscalls[68].current)
#if LIBCU_OS_WINRT && !defined(LIBCU_OMIT_LOAD_EXTENSION)
	{ "LoadPackagedLibrary",     (SYSCALL)LoadPackagedLibrary, nullptr },
#else
	{ "LoadPackagedLibrary",     (SYSCALL)nullptr, nullptr },
#endif
#define osLoadPackagedLibrary ((HMODULE(WINAPI*)(LPCWSTR,DWORD))Syscalls[69].current)
#if LIBCU_OS_WINRT
	{ "GetTickCount64",          (SYSCALL)GetTickCount64, nullptr },
#else
	{ "GetTickCount64",          (SYSCALL)nullptr, nullptr },
#endif
#define osGetTickCount64 ((ULONGLONG(WINAPI*)(VOID))Syscalls[70].current)
#if LIBCU_OS_WINRT
	{ "GetNativeSystemInfo",     (SYSCALL)GetNativeSystemInfo, nullptr },
#else
	{ "GetNativeSystemInfo",     (SYSCALL)nullptr, nullptr },
#endif
#define osGetNativeSystemInfo ((VOID(WINAPI*)(LPSYSTEM_INFO))Syscalls[71].current)
#if defined(LIBCU_WIN32_HAS_ANSI)
	{ "OutputDebugStringA",      (SYSCALL)OutputDebugStringA, nullptr },
#else
	{ "OutputDebugStringA",      (SYSCALL)nullptr, nullptr },
#endif
#define osOutputDebugStringA ((VOID(WINAPI*)(LPCSTR))Syscalls[72].current)
#if defined(LIBCU_WIN32_HAS_WIDE)
	{ "OutputDebugStringW",      (SYSCALL)OutputDebugStringW, nullptr },
#else
	{ "OutputDebugStringW",      (SYSCALL)nullptr, nullptr },
#endif
#define osOutputDebugStringW ((VOID(WINAPI*)(LPCWSTR))Syscalls[73].current)
	{ "GetProcessHeap",          (SYSCALL)GetProcessHeap, nullptr },
#define osGetProcessHeap ((HANDLE(WINAPI*)(VOID))Syscalls[74].current)
#if LIBCU_OS_WINRT && (!defined(LIBCU_OMIT_WAL) || LIBCU_MAXMMAPSIZE>0)
	{ "CreateFileMappingFromApp", (SYSCALL)CreateFileMappingFromApp, nullptr },
#else
	{ "CreateFileMappingFromApp", (SYSCALL)nullptr, nullptr },
#endif
#define osCreateFileMappingFromApp ((HANDLE(WINAPI*)(HANDLE,LPSECURITY_ATTRIBUTES,ULONG,ULONG64,LPCWSTR))Syscalls[75].current)
	/*
	** NOTE: On some sub-platforms, the InterlockedCompareExchange "function" is really just a macro that uses a compiler intrinsic (e.g. x64).
	**       So do not try to make this is into a redefinable interface.
	*/
#if defined(InterlockedCompareExchange)
	{ "InterlockedCompareExchange", (SYSCALL)nullptr, nullptr },
#define osInterlockedCompareExchange InterlockedCompareExchange
#else
	{ "InterlockedCompareExchange", (SYSCALL)InterlockedCompareExchange, nullptr },
#define osInterlockedCompareExchange ((LONG(WINAPI*)(LONG LIBCU_WIN32_VOLATILE*,LONG,LONG))Syscalls[76].current)
#endif /* defined(InterlockedCompareExchange) */
#if !LIBCU_OS_WINCE && !LIBCU_OS_WINRT && LIBCU_WIN32_USE_UUID
	{ "UuidCreate",               (SYSCALL)UuidCreate, nullptr },
#else
	{ "UuidCreate",               (SYSCALL)nullptr, nullptr },
#endif
#define osUuidCreate ((RPC_STATUS(RPC_ENTRY*)(UUID*))Syscalls[77].current)
#if !LIBCU_OS_WINCE && !LIBCU_OS_WINRT && LIBCU_WIN32_USE_UUID
	{ "UuidCreateSequential",     (SYSCALL)UuidCreateSequential, nullptr },
#else
	{ "UuidCreateSequential",     (SYSCALL)nullptr, nullptr },
#endif
#define osUuidCreateSequential ((RPC_STATUS(RPC_ENTRY*)(UUID*))Syscalls[78].current)
#if !defined(LIBCU_NO_SYNC) && LIBCU_MAXMMAPSIZE > 0
	{ "FlushViewOfFile",          (SYSCALL)FlushViewOfFile, nullptr },
#else
	{ "FlushViewOfFile",          (SYSCALL)nullptr, nullptr },
#endif
#define osFlushViewOfFile ((BOOL(WINAPI*)(LPCVOID,SIZE_T))Syscalls[79].current)

}; /* End of the overrideable system calls */

/*
** This is the xSetSystemCall() method of vsystem for all of the "win32" VFSes.  Return RC_OK opon successfully updating the
** system call pointer, or LIBCU_NOTFOUND if there is no configurable system call named zName.
*/
static int winSetSystemCall(vsystem *notUsed, const char *name, SYSCALL newFunc)
{
	RC rc = RC_NOTFOUND;
	UNUSED_SYMBOL(notUsed);
	if (!name) {
		// If no zName is given, restore all system calls to their default settings and return NULL
		rc = RC_OK;
		for (unsigned int i = 0; i < _LENGTHOF(Syscalls); i++)
			if (Syscalls[i].default)
				Syscalls[i].current = Syscalls[i].default;
	}
	else for (unsigned int i = 0; i < _LENGTHOF(Syscalls); i++) {
		// If zName is specified, operate on only the one system call specified.
		if (!strcmp(name, Syscalls[i].name)) {
			if (!Syscalls[i].default)
				Syscalls[i].default = Syscalls[i].current;
			rc = RC_OK;
			if (!newFunc) newFunc = Syscalls[i].default;
			Syscalls[i].current = newFunc;
			break;
		}
	}
	return rc;
}

/*
** Return the value of a system call.  Return NULL if zName is not a recognized system call name.  NULL is also returned if the system call
** is currently undefined.
*/
static SYSCALL winGetSystemCall(vsystem *notUsed, const char *name)
{
	UNUSED_SYMBOL(notUsed);
	for (unsigned int i = 0; i < _LENGTHOF(Syscalls); i++)
		if (!strcmp(name, Syscalls[i].name)) return Syscalls[i].current;
	return nullptr;
}

/*
** Return the name of the first system call after zName.  If zName==NULL then return the name of the first system call.  Return NULL if zName
** is the last system call or if zName is not the name of a valid system call.
*/
static const char *winNextSystemCall(vsystem *p, const char *name)
{
	UNUSED_SYMBOL(p);
	int i = -1;
	if (name)
		for (i = 0; i < _LENGTHOF(Syscalls)-1; i++)
			if (!strcmp(name, Syscalls[i].name)) break;
	for (i++; i < _LENGTHOF(Syscalls); i++)
		if (Syscalls[i].current) return Syscalls[i].name;
	return nullptr;
}

#pragma endregion

#pragma region Malloc2
#ifdef LIBCU_WIN32_MALLOC
/*
** If a Win32 native heap has been configured, this function will attempt to compact it.  Upon success, RC_OK will be returned.  Upon failure, one
** of LIBCU_NOMEM, LIBCU_ERROR, or LIBCU_NOTFOUND will be returned.  The "pnLargest" argument, if non-zero, will be used to return the size of the
** largest committed free block in the heap, in bytes.
*/
int sqlite3_win32_compact_heap(LPUINT pLargest)
{
	RC rc = RC_OK;
	winMemAssertMagic();
	HANDLE heap = winMemGetHeap();
	assert(heap);
	assert(heap != INVALID_HANDLE_VALUE);
#if !LIBCU_OS_WINRT && defined(LIBCU_WIN32_MALLOC_VALIDATE)
	assert(osHeapValidate(heap, LIBCU_WIN32_HEAP_FLAGS, NULL));
#endif
	UINT largest = 0;
#if !LIBCU_OS_WINCE && !LIBCU_OS_WINRT
	if ((largest = osHeapCompact(heap, LIBCU_WIN32_HEAP_FLAGS)) == 0) {
		DWORD lastErrno = osGetLastError();
		if (lastErrno == NO_ERROR) {
			_log(LIBCU_NOMEM, "failed to HeapCompact (no space), heap=%p", (void *)heap);
			rc = RC_NOMEM_BKPT;
		}
		else {
			_log(LIBCU_ERROR, "failed to HeapCompact (%lu), heap=%p", osGetLastError(), (void *)heap);
			rc = RC_ERROR;
		}
	}
#else
	_log(RC_NOTFOUND, "failed to HeapCompact, heap=%p", (void *)heap);
	rc = RC_NOTFOUND;
#endif
	if (pLargest) *pLargest = largest;
	return rc;
}

/*
** If a Win32 native heap has been configured, this function will attempt to destroy and recreate it.  If the Win32 native heap is not isolated and/or
** the sqlite3_memory_used() function does not return zero, LIBCU_BUSY will be returned and no changes will be made to the Win32 native heap.
*/
int sqlite3_win32_reset_heap()
{
	MUTEX_LOGIC(mutex *master = mutexAlloc(LIBCU_MUTEX_STATIC_MASTER);) // The main static mutex
		MUTEX_LOGIC(mutex *mem = mutexAlloc(LIBCU_MUTEX_STATIC_MEM);) // The memsys static mutex
		mutex_enter(master);
	mutex_enter(mem);
	winMemAssertMagic();
	RC rc;
	if (winMemGetHeap() && winMemGetOwned() && !sqlite3_memory_used()) {
		// At this point, there should be no outstanding memory allocations on the heap.  Also, since both the master and memsys locks are currently
		// being held by us, no other function (i.e. from another thread) should be able to even access the heap.  Attempt to destroy and recreate our
		// isolated Win32 native heap now.
		assert(winMemGetHeap());
		assert(winMemGetOwned());
		assert(!sqlite3_memory_used());
		winMemShutdown(winMemGetDataPtr());
		assert(!winMemGetHeap());
		assert(!winMemGetOwned());
		assert(!sqlite3_memory_used());
		rc = winMemInit(winMemGetDataPtr());
		assert(rc != RC_OK || winMemGetHeap());
		assert(rc != RC_OK || winMemGetOwned());
		assert(rc != RC_OK || sqlite3_memory_used()==0 );
	}
	// The Win32 native heap cannot be modified because it may be in use.
	else rc = RC_BUSY;
	mutex_leave(mem);
	mutex_leave(master);
	return rc;
}
#endif /* LIBCU_WIN32_MALLOC */

#pragma endregion

#pragma region X

/* This function outputs the specified (ANSI) string to the Win32 debugger (if available). */
void sqlite3_win32_write_debug(const char *buf, int bufLength)
{
	char dbgBuf[LIBCU_WIN32_DBG_BUF_SIZE];
	int min = _MIN(bufLength, (LIBCU_WIN32_DBG_BUF_SIZE - 1)); // may be negative.
	if (min < -1) min = -1; // all negative values become -1.
	assert(min == -1 || min == 0 || min < LIBCU_WIN32_DBG_BUF_SIZE);
#ifdef ENABLE_API_ARMOR
	if (!buf) {
		(void)RC_MISUSE_BKPT;
		return;
	}
#endif
#if defined(LIBCU_WIN32_HAS_ANSI)
	if (min > 0) {
		memset(dbgBuf, 0, LIBCU_WIN32_DBG_BUF_SIZE);
		memcpy(dbgBuf, buf, min);
		osOutputDebugStringA(dbgBuf);
	}
	else osOutputDebugStringA(buf);
#elif defined(LIBCU_WIN32_HAS_WIDE)
	memset(dbgBuf, 0, LIBCU_WIN32_DBG_BUF_SIZE);
	if (osMultiByteToWideChar(osAreFileApisANSI() ? CP_ACP : CP_OEMCP, 0, buf, min, (LPWSTR)dbgBuf, LIBCU_WIN32_DBG_BUF_SIZE/sizeof(WCHAR)) <= 0)
		return;
	osOutputDebugStringW((LPCWSTR)dbgBuf);
#else
	if (min > 0) {
		memset(dbgBuf, 0, LIBCU_WIN32_DBG_BUF_SIZE);
		memcpy(dbgBuf, buf, nMin);
		fprintf(stderr, "%s", dbgBuf);
	}
	else fprintf(stderr, "%s", buf);
#endif
}

/* The following routine suspends the current thread for at least ms milliseconds.  This is equivalent to the Win32 Sleep() interface. */
#if LIBCU_OS_WINRT
static HANDLE _sleepObj = NULL;
#endif

void sqlite3_win32_sleep(DWORD milliseconds)
{
#if LIBCU_OS_WINRT
	if (!sleepObj)
		sleepObj = osCreateEventExW(NULL, NULL, CREATE_EVENT_MANUAL_RESET, SYNCHRONIZE);
	assert(sleepObj);
	osWaitForSingleObjectEx(sleepObj, milliseconds, FALSE);
#else
	osSleep(milliseconds);
#endif
}

#if LIBCU_MAX_WORKER_THREADS > 0 && !LIBCU_OS_WINCE && !LIBCU_OS_WINRT && LIBCU_THREADSAFE > 0
DWORD sqlite3Win32Wait(HANDLE hObject)
{
	DWORD rc;
	while ((rc = osWaitForSingleObjectEx(hObject, INFINITE, TRUE)) == WAIT_IO_COMPLETION) {}
	return rc;
}
#endif

/*
** Return true (non-zero) if we are running under WinNT, Win2K, WinXP, or WinCE.  Return false (zero) for Win95, Win98, or WinME.
**
** Here is an interesting observation:  Win95, Win98, and WinME lack the LockFileEx() API.  But we can still statically link against that
** API as long as we don't call it when running Win95/98/ME.  A call to this routine is used to determine if the host is Win95/98/ME or
** WinNT/2K/XP so that we will know whether or not we can safely call the LockFileEx() API.
*/

#if !LIBCU_WIN32_GETVERSIONEX
#define osIsNT()  (1)
#elif LIBCU_OS_WINCE || LIBCU_OS_WINRT || !defined(LIBCU_WIN32_HAS_ANSI)
#define osIsNT()  (1)
#elif !defined(LIBCU_WIN32_HAS_WIDE)
#define osIsNT()  (0)
#else
#define osIsNT()  ((_os_type == 2) || sqlite3_win32_is_nt())
#endif

/* This function determines if the machine is running a version of Windows based on the NT kernel. */
int sqlite3_win32_is_nt()
{
#if LIBCU_OS_WINRT
	// NOTE: The WinRT sub-platform is always assumed to be based on the NT kernel.
	return 1;
#elif LIBCU_WIN32_GETVERSIONEX
	if (osInterlockedCompareExchange(&_os_type, 0, 0) == 0) {
#if defined(LIBCU_WIN32_HAS_ANSI)
		OSVERSIONINFOA sInfo;
		sInfo.dwOSVersionInfoSize = sizeof(sInfo);
		osGetVersionExA(&sInfo);
		osInterlockedCompareExchange(&_os_type, (sInfo.dwPlatformId == VER_PLATFORM_WIN32_NT ? 2 : 1), 0);
#elif defined(LIBCU_WIN32_HAS_WIDE)
		OSVERSIONINFOW sInfo;
		sInfo.dwOSVersionInfoSize = sizeof(sInfo);
		osGetVersionExW(&sInfo);
		osInterlockedCompareExchange(&_os_type, (sInfo.dwPlatformId == VER_PLATFORM_WIN32_NT ? 2 : 1), 0);
#endif
	}
	return osInterlockedCompareExchange(&_os_type, 2, 2)==2;
#elif LIBCU_TEST
	return osInterlockedCompareExchange(&_os_type, 2, 2)==2;
#else
	// NOTE: All sub-platforms where the GetVersionEx[AW] functions are deprecated are always assumed to be based on the NT kernel.
	return 1;
#endif
}

#pragma endregion

#pragma region Malloc3

#ifdef LIBCU_WIN32_MALLOC
/* Allocate size of memory. */
static void *winMemMalloc(int size)
{
	winMemAssertMagic();
	HANDLE heap = winMemGetHeap();
	assert(heap);
	assert(heap != INVALID_HANDLE_VALUE);
#if !LIBCU_OS_WINRT && defined(LIBCU_WIN32_MALLOC_VALIDATE)
	assert(osHeapValidate(heap, LIBCU_WIN32_HEAP_FLAGS, NULL));
#endif
	assert(size >= 0);
	void *p = osHeapAlloc(heap, LIBCU_WIN32_HEAP_FLAGS, (SIZE_T)size);
	if (!p)
		_log(RC_NOMEM, "failed to HeapAlloc %u bytes (%lu), heap=%p", size, osGetLastError(), (void *)heap);
	return p;
}

/* Free memory. */
static void winMemFree(void *prior)
{
	winMemAssertMagic();
	HANDLE heap = winMemGetHeap();
	assert(heap);
	assert(heap != INVALID_HANDLE_VALUE);
#if !LIBCU_OS_WINRT && defined(LIBCU_WIN32_MALLOC_VALIDATE)
	assert(osHeapValidate(heap, LIBCU_WIN32_HEAP_FLAGS, prior));
#endif
	if (!prior) return; // Passing NULL to HeapFree is undefined.
	if (!osHeapFree(heap, LIBCU_WIN32_HEAP_FLAGS, prior))
		_log(RC_NOMEM, "failed to HeapFree block %p (%lu), heap=%p", prior, osGetLastError(), (void *)heap);
}

/* Change the size of an existing memory allocation */
static void *winMemRealloc(void *prior, int newSize)
{
	winMemAssertMagic();
	HANDLE heap = winMemGetHeap();
	assert(heap);
	assert(heap != INVALID_HANDLE_VALUE);
#if !LIBCU_OS_WINRT && defined(LIBCU_WIN32_MALLOC_VALIDATE)
	assert(osHeapValidate(heap, LIBCU_WIN32_HEAP_FLAGS, prior));
#endif
	assert(newSize >= 0);
	void *p;
	if (!prior) p = osHeapAlloc(heap, LIBCU_WIN32_HEAP_FLAGS, (SIZE_T)newSize);
	else p = osHeapReAlloc(heap, LIBCU_WIN32_HEAP_FLAGS, prior, (SIZE_T)newSize);
	if (!p)
		_log(RC_NOMEM, "failed to %s %u bytes (%lu), heap=%p", prior ? "HeapReAlloc" : "HeapAlloc", newSize, osGetLastError(), (void *)heap);
	return p;
}

/* Return the size of an outstanding allocation, in bytes. */
static int winMemSize(void *p)
{
	winMemAssertMagic();
	HANDLE heap = winMemGetHeap();
	assert(heap);
	assert(heap != INVALID_HANDLE_VALUE);
#if !LIBCU_OS_WINRT && defined(LIBCU_WIN32_MALLOC_VALIDATE)
	assert(osHeapValidate(heap, LIBCU_WIN32_HEAP_FLAGS, p));
#endif
	if (!p) return 0;
	SIZE_T n = osHeapSize(heap, LIBCU_WIN32_HEAP_FLAGS, p);
	if (n == (SIZE_T)-1) {
		_log(RC_NOMEM, "failed to HeapSize block %p (%lu), heap=%p", p, osGetLastError(), (void *)heap);
		return 0;
	}
	return (int)n;
}

/* Round up a request size to the next valid allocation size. */
static int winMemRoundup(int size)
{
	return size;
}

/* Initialize this module. */
static RC winMemInitialize(void *appData)
{
	WinMemData *winMemData = (WinMemData *)appData;
	if (!winMemData) return RC_ERROR;
	assert(winMemData->magic1 == WINMEM_MAGIC1);
	assert(winMemData->magic2 == WINMEM_MAGIC2);
#if !LIBCU_OS_WINRT && LIBCU_WIN32_HEAP_CREATE
	if (!winMemData->heap) {
		DWORD dwInitialSize = LIBCU_WIN32_HEAP_INIT_SIZE;
		DWORD dwMaximumSize = (DWORD)_runtimeConfig.heaps;
		if (!dwMaximumSize)
			dwMaximumSize = LIBCU_WIN32_HEAP_MAX_SIZE;
		else if (dwInitialSize > dwMaximumSize)
			dwInitialSize = dwMaximumSize;
		winMemData->heap = osHeapCreate(LIBCU_WIN32_HEAP_FLAGS, dwInitialSize, dwMaximumSize);
		if (!winMemData->heap) {
			_log(RC_NOMEM, "failed to HeapCreate (%lu), flags=%u, initSize=%lu, maxSize=%lu", osGetLastError(), LIBCU_WIN32_HEAP_FLAGS, dwInitialSize, dwMaximumSize);
			return RC_NOMEM_BKPT;
		}
		winMemData->owned = TRUE;
		assert(winMemData->owned);
	}
#else
	winMemData->heap = osGetProcessHeap();
	if (!winMemData->heap) {
		_log(RC_NOMEM, "failed to GetProcessHeap (%lu)", osGetLastError());
		return RC_NOMEM_BKPT;
	}
	winMemData->owned = FALSE;
	assert(!winMemData->owned);
#endif
	assert(winMemData->heap);
	assert(winMemData->heap != INVALID_HANDLE_VALUE);
#if !LIBCU_OS_WINRT && defined(LIBCU_WIN32_MALLOC_VALIDATE)
	assert(osHeapValidate(winMemData->heap, LIBCU_WIN32_HEAP_FLAGS, NULL));
#endif
	return RC_OK;
}

/* Deinitialize this module. */
static RC winMemShutdown(void *appData)
{
	WinMemData *winMemData = (WinMemData *)appData;
	if (!winMemData) return;
	assert(winMemData->magic1 == WINMEM_MAGIC1);
	assert(winMemData->magic2 == WINMEM_MAGIC2);
	if (winMemData->heap) {
		assert(winMemData->heap != INVALID_HANDLE_VALUE);
#if !LIBCU_OS_WINRT && defined(LIBCU_WIN32_MALLOC_VALIDATE)
		assert(osHeapValidate(winMemData->heap, LIBCU_WIN32_HEAP_FLAGS, NULL));
#endif
		if (winMemData->owned) {
			if (!osHeapDestroy(winMemData->heap))
				_log(RC_NOMEM, "failed to HeapDestroy (%lu), heap=%p", osGetLastError(), (void *)winMemData->heap);
			winMemData->owned = FALSE;
		}
		winMemData->heap = NULL;
	}
}

/*
** Populate the low-level memory allocation function pointers in _.alloc with pointers to the routines in this file. The
** arguments specify the block of memory to manage.
**
** This routine is only called by runtimeConfig(), and therefore is not required to be threadsafe (it is not).
*/
const alloc_methods *__allocsystemGetWin32()
{
	static const alloc_methods _winMemMethods = {
		winMemMalloc,
		winMemFree,
		winMemRealloc,
		winMemSize,
		winMemRoundup,
		winMemInitialize,
		winMemShutdown,
		&win_mem_data
	};
	return &_winMemMethods;
}

void __allocsystemSetDefault()
{
	runtimeConfig(CONFIG_MALLOC, __allocsystemGetWin32());
}
#endif /* LIBCU_WIN32_MALLOC */

#pragma endregion

#pragma region String Converters

/*
** Convert a UTF-8 string to Microsoft Unicode.
**
** Space to hold the returned string is obtained from sqlite3_malloc().
*/
static LPWSTR winUtf8ToUnicode(const char *text)
{
	int chars = osMultiByteToWideChar(CP_UTF8, 0, text, -1, NULL, 0);
	if (!chars)
		return nullptr;
	LPWSTR wideText = (LPWSTR)allocZero(chars * sizeof(WCHAR));
	if (!wideText)
		return nullptr;
	chars = osMultiByteToWideChar(CP_UTF8, 0, text, -1, wideText, chars);
	if (!chars) {
		mfree(wideText);
		wideText = nullptr;
	}
	return wideText;
}

/*
** Convert a Microsoft Unicode string to UTF-8.
**
** Space to hold the returned string is obtained from sqlite3_malloc().
*/
static char *winUnicodeToUtf8(LPCWSTR wideText)
{
	int bytes = osWideCharToMultiByte(CP_UTF8, 0, wideText, -1, 0, 0, 0, 0);
	if (!bytes)
		return nullptr;
	char *text = (char *)allocZero(bytes);
	if (!text)
		return nullptr;
	bytes = osWideCharToMultiByte(CP_UTF8, 0, wideText, -1, text, bytes, 0, 0);
	if (!bytes){
		mfree(text);
		text = nullptr;
	}
	return text;
}

/*
** Convert an ANSI string to Microsoft Unicode, using the ANSI or OEM code page.
**
** Space to hold the returned string is obtained from sqlite3_malloc().
*/
static LPWSTR winMbcsToUnicode(const char *text, int useAnsi)
{
	int codepage = useAnsi ? CP_ACP : CP_OEMCP;
	int bytes = osMultiByteToWideChar(codepage, 0, text, -1, NULL, 0) * sizeof(WCHAR);
	if (!bytes)
		return nullptr;
	LPWSTR mbcsText = (LPWSTR)allocZero(bytes * sizeof(WCHAR));
	if (!mbcsText)
		return nullptr;
	bytes = osMultiByteToWideChar(codepage, 0, text, -1, mbcsText, bytes);
	if (!bytes) {
		mfree(mbcsText);
		mbcsText = nullptr;
	}
	return mbcsText;
}

/*
** Convert a Microsoft Unicode string to a multi-byte character string, using the ANSI or OEM code page.
**
** Space to hold the returned string is obtained from sqlite3_malloc().
*/
static char *winUnicodeToMbcs(LPCWSTR wideText, int useAnsi)
{
	int codepage = useAnsi ? CP_ACP : CP_OEMCP;
	int bytes = osWideCharToMultiByte(codepage, 0, wideText, -1, 0, 0, 0, 0);
	if (!bytes)
		return nullptr;
	char *text = (char *)allocZero(bytes);
	if (!text)
		return nullptr;
	bytes = osWideCharToMultiByte(codepage, 0, wideText, -1, text, bytes, 0, 0);
	if (!bytes) {
		mfree(text);
		text = nullptr;
	}
	return text;
}

/*
** Convert a multi-byte character string to UTF-8.
**
** Space to hold the returned string is obtained from sqlite3_malloc().
*/
static char *winMbcsToUtf8(const char *text, int useAnsi)
{
	LPWSTR tmpWide = winMbcsToUnicode(text, useAnsi);
	if (!tmpWide)
		return nullptr;
	char *textUtf8 = winUnicodeToUtf8(tmpWide);
	mfree(tmpWide);
	return textUtf8;
}

/*
** Convert a UTF-8 string to a multi-byte character string.
**
** Space to hold the returned string is obtained from sqlite3_malloc().
*/
static char *winUtf8ToMbcs(const char *text, int useAnsi)
{
	LPWSTR tmpWide = winUtf8ToUnicode(text);
	if (!tmpWide)
		return nullptr;
	char *textMbcs = winUnicodeToMbcs(tmpWide, useAnsi);
	mfree(tmpWide);
	return textMbcs;
}

/* This is a public wrapper for the winUtf8ToUnicode() function. */
LPWSTR sqlite3_win32_utf8_to_unicode(const char *text)
{
#ifdef ENABLE_API_ARMOR
	if (!text) {
		(void)RC_MISUSE_BKPT;
		return 0;
	}
#endif
#ifndef LIBCU_OMIT_AUTOINIT
	if (runtimeInitialize()) return 0;
#endif
	return winUtf8ToUnicode(text);
}

/*
** This is a public wrapper for the winUnicodeToUtf8() function.
*/
char *sqlite3_win32_unicode_to_utf8(LPCWSTR wideText)
{
#ifdef ENABLE_API_ARMOR
	if (!wideText) {
		(void)RC_MISUSE_BKPT;
		return nullptr;
	}
#endif
#ifndef LIBCU_OMIT_AUTOINIT
	if (runtimeInitialize()) return nullptr;
#endif
	return winUnicodeToUtf8(wideText);
}

/* This is a public wrapper for the winMbcsToUtf8() function. */
char *sqlite3_win32_mbcs_to_utf8(const char *text)
{
#ifdef ENABLE_API_ARMOR
	if (!text) {
		(void)RC_MISUSE_BKPT;
		return nullptr;
	}
#endif
#ifndef LIBCU_OMIT_AUTOINIT
	if (runtimeInitialize()) return nullptr;
#endif
	return winMbcsToUtf8(text, osAreFileApisANSI());
}

/* This is a public wrapper for the winMbcsToUtf8() function. */
char *sqlite3_win32_mbcs_to_utf8_v2(const char *text, int useAnsi)
{
#ifdef ENABLE_API_ARMOR
	if (!text) {
		(void)RC_MISUSE_BKPT;
		return nullptr;
	}
#endif
#ifndef LIBCU_OMIT_AUTOINIT
	if (runtimeInitialize()) return nullptr;
#endif
	return winMbcsToUtf8(text, useAnsi);
}

/* This is a public wrapper for the winUtf8ToMbcs() function. */
char *sqlite3_win32_utf8_to_mbcs(const char *text)
{
#ifdef ENABLE_API_ARMOR
	if (!text) {
		(void)RC_MISUSE_BKPT;
		return nullptr;
	}
#endif
#ifndef LIBCU_OMIT_AUTOINIT
	if (runtimeInitialize()) return nullptr;
#endif
	return winUtf8ToMbcs(text, osAreFileApisANSI());
}

/* This is a public wrapper for the winUtf8ToMbcs() function. */
char *sqlite3_win32_utf8_to_mbcs_v2(const char *text, int useAnsi)
{
#ifdef ENABLE_API_ARMOR
	if (!text) {
		(void)RC_MISUSE_BKPT;
		return nullptr;
	}
#endif
#ifndef LIBCU_OMIT_AUTOINIT
	if (runtimeInitialize()) return nullptr;
#endif
	return winUtf8ToMbcs(text, useAnsi);
}

#pragma endregion

#pragma region Win32
/*
** This function sets the data directory or the temporary directory based on the provided arguments.  The type argument must be 1 in order to set the
** data directory or 2 in order to set the temporary directory.  The zValue argument is the name of the directory to use.  The return value will be
** RC_OK if successful.
*/
int sqlite3_win32_set_directory(DWORD type, LPCWSTR value)
{
#ifndef LIBCU_OMIT_AUTOINIT
	RC rc = runtimeInitialize();
	if (rc) return rc;
#endif
	char **directory = nullptr;
	if (type == LIBCU_WIN32_DATA_DIRECTORY_TYPE) directory = &libcu_dataDirectory;
	else if (type == LIBCU_WIN32_TEMP_DIRECTORY_TYPE) directory = &libcu_tempDirectory;
	assert(!directory || type == LIBCU_WIN32_DATA_DIRECTORY_TYPE || type == LIBCU_WIN32_TEMP_DIRECTORY_TYPE);
	assert(!directory || memdebug_hastype(*directory, MEMTYPE_HEAP));
	if (directory) {
		char *valueUtf8 = 0;
		if (value && value[0]) {
			valueUtf8 = winUnicodeToUtf8(value);
			if (!valueUtf8)
				return RC_NOMEM_BKPT;
		}
		mfree(*directory);
		*directory = valueUtf8;
		return RC_OK;
	}
	return RC_ERROR;
}

#pragma endregion

#pragma region OS Errors

/*
** The return value of winGetLastErrorMsg is zero if the error message fits in the buffer, or non-zero
** otherwise (if the message was truncated).
*/
static int winGetLastErrorMsg(DWORD lastErrno, int bufSize, char *buf)
{
	// FormatMessage returns 0 on failure.  Otherwise it returns the number of TCHARs written to the output
	// buffer, excluding the terminating null char.
	DWORD dwLen = 0;
	char *out = nullptr;
	if (osIsNT()) {
#if LIBCU_OS_WINRT
		WCHAR tempWide[LIBCU_WIN32_MAX_ERRMSG_CHARS+1];
		dwLen = osFormatMessageW(FORMAT_MESSAGE_FROM_SYSTEM|FORMAT_MESSAGE_IGNORE_INSERTS, NULL, lastErrno, 0, tempWide, LIBCU_WIN32_MAX_ERRMSG_CHARS, 0);
#else
		LPWSTR tempWide = NULL;
		dwLen = osFormatMessageW(FORMAT_MESSAGE_ALLOCATE_BUFFER|FORMAT_MESSAGE_FROM_SYSTEM|FORMAT_MESSAGE_IGNORE_INSERTS, NULL, lastErrno, 0, (LPWSTR)&tempWide, 0, 0);
#endif
		if (dwLen > 0) {
			// allocate a buffer and convert to UTF8
			allocBenignBegin();
			out = winUnicodeToUtf8(tempWide);
			allocBenignEnd();
#if !LIBCU_OS_WINRT
			// free the system buffer allocated by FormatMessage
			osLocalFree(tempWide);
#endif
		}
	}
#ifdef LIBCU_WIN32_HAS_ANSI
	else {
		char *temp = NULL;
		dwLen = osFormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER|FORMAT_MESSAGE_FROM_SYSTEM|FORMAT_MESSAGE_IGNORE_INSERTS, NULL, lastErrno, 0, (LPSTR)&temp, 0, 0);
		if (dwLen > 0) {
			// allocate a buffer and convert to UTF8
			allocBenignBegin();
			out = winMbcsToUtf8(temp, osAreFileApisANSI());
			allocBenignEnd();
			// free the system buffer allocated by FormatMessage
			osLocalFree(temp);
		}
	}
#endif
	if (!dwLen)
		snprintf(buf, bufSize, "OsError 0x%lx (%lu)", lastErrno, lastErrno);
	else {
		// copy a maximum of nBuf chars to output buffer
		snprintf(buf, bufSize, "%s", out);
		// free the UTF8 buffer
		mfree(out);
	}
	return 0;
}

/*
**
** This function - winLogErrorAtLine() - is only ever called via the macro winLogError().
**
** This routine is invoked after an error occurs in an OS function. It logs a message using _log() containing the current value of
** error code and, if possible, the human-readable equivalent from FormatMessage.
**
** The first argument passed to the macro should be the error code that will be returned to Libcu (e.g. LIBCU_IOERR_DELETE, LIBCU_CANTOPEN).
** The two subsequent arguments should be the name of the OS function that failed and the associated file-system path, if any.
*/
#define winLogError(a,b,c,d) winLogErrorAtLine(a,b,c,d,__LINE__)
static int winLogErrorAtLine(RC errcode, DWORD lastErrno, const char *func, const char *path, int line)
{
	char msg[500]; // Human readable error text
	msg[0] = 0;
	winGetLastErrorMsg(lastErrno, sizeof(msg), msg);
	assert(errcode != RC_OK);
	if (!path) path = "";
	int i;
	for (i = 0; msg[i] && msg[i] != '\r' && msg[i] != '\n'; i++) { }
	msg[i] = 0;
	_log(errcode, "os_win.c:%d: (%lu) %s(%s) - %s", line, lastErrno, func, path, msg);
	return errcode;
}

/*
** The number of times that a ReadFile(), WriteFile(), and DeleteFile() will be retried following a locking error - probably caused by
** antivirus software.  Also the initial delay before the first retry. The delay increases linearly with each retry.
*/
#ifndef LIBCU_WIN32_IOERR_RETRY
#define LIBCU_WIN32_IOERR_RETRY 10
#endif
#ifndef LIBCU_WIN32_IOERR_RETRY_DELAY
#define LIBCU_WIN32_IOERR_RETRY_DELAY 25
#endif
static int winIoerrRetry = LIBCU_WIN32_IOERR_RETRY;
static int winIoerrRetryDelay = LIBCU_WIN32_IOERR_RETRY_DELAY;

/*
** The "winIoerrCanRetry1" macro is used to determine if a particular I/O error code obtained via GetLastError() is eligible to be retried.  It
** must accept the error code DWORD as its only argument and should return non-zero if the error code is transient in nature and the operation
** responsible for generating the original error might succeed upon being retried.  The argument to this macro should be a variable.
**
** Additionally, a macro named "winIoerrCanRetry2" may be defined.  If it is defined, it will be consulted only when the macro "winIoerrCanRetry1"
** returns zero.  The "winIoerrCanRetry2" macro is completely optional and may be used to include additional error codes in the set that should
** result in the failing I/O operation being retried by the caller.  If defined, the "winIoerrCanRetry2" macro must exhibit external semantics
** identical to those of the "winIoerrCanRetry1" macro.
*/
#if !defined(winIoerrCanRetry1)
#define winIoerrCanRetry1(a) ((a) == ERROR_ACCESS_DENIED || \
	(a) == ERROR_SHARING_VIOLATION || \
	(a) == ERROR_LOCK_VIOLATION || \
	(a) == ERROR_DEV_NOT_EXIST || \
	(a) == ERROR_NETNAME_DELETED || \
	(a) == ERROR_SEM_TIMEOUT || \
	(a) == ERROR_NETWORK_UNREACHABLE)
#endif

/*
** If a ReadFile() or WriteFile() error occurs, invoke this routine to see if it should be retried.  Return TRUE to retry.  Return FALSE
** to give up with an error.
*/
static int winRetryIoerr(int *retry, DWORD *error)
{
	DWORD e = osGetLastError();
	if (*retry >= winIoerrRetry) {
		if (error)
			*error = e;
		return 0;
	}
	if (winIoerrCanRetry1(e)) {
		sqlite3_win32_sleep(winIoerrRetryDelay*(1+*retry));
		++*retry;
		return 1;
	}
#if defined(winIoerrCanRetry2)
	else if (winIoerrCanRetry2(e)) {
		sqlite3_win32_sleep(winIoerrRetryDelay * (1 + *retry));
		++*rRetry;
		return 1;
	}
#endif
	if (error)
		*error = e;
	return 0;
}

/* Log a I/O error retry episode. */
static void winLogIoerr(int retry, int lineno)
{
	if (retry)
		_log(RC_NOTICE, "delayed %dms for lock/sharing conflict at line %d", winIoerrRetryDelay * retry * (retry+1) / 2, lineno);
}

#pragma endregion

#pragma region VSystemFile

#pragma region WinCE Only

/* This #if does not rely on the LIBCU_OS_WINCE define because the corresponding section in "date.c" cannot use it. */
#if !defined(LIBCU_OMIT_LOCALTIME) && defined(_WIN32_WCE) && (!defined(LIBCU_MSVC_LOCALTIME_API) || !LIBCU_MSVC_LOCALTIME_API)
/* The MSVC CRT on Windows CE may not have a localtime() function. So define a substitute. */
#include <time.h>
struct tm *__cdecl localtime(const time_t *t)
{
	static struct tm y;
	FILETIME uTm, lTm;
	SYSTEMTIME pTm;
	int64_t t64;
	t64 = *t;
	t64 = (t64 + 11644473600)*10000000;
	uTm.dwLowDateTime = (DWORD)(t64 & 0xFFFFFFFF);
	uTm.dwHighDateTime= (DWORD)(t64 >> 32);
	osFileTimeToLocalFileTime(&uTm,&lTm);
	osFileTimeToSystemTime(&lTm,&pTm);
	y.tm_year = pTm.wYear - 1900;
	y.tm_mon = pTm.wMonth - 1;
	y.tm_wday = pTm.wDayOfWeek;
	y.tm_mday = pTm.wDay;
	y.tm_hour = pTm.wHour;
	y.tm_min = pTm.wMinute;
	y.tm_sec = pTm.wSecond;
	return &y;
}
#endif

#if LIBCU_OS_WINCE
/*************************************************************************
** This section contains code for WinCE only.
*/
#define HANDLE_TO_WINFILE(a) (winFile*)&((char*)a)[-(int)offsetof(winFile,h)]

/* Acquire a lock on the handle h */
static void winceMutexAcquire(HANDLE h)
{
	DWORD dwErr;
	do {
		dwErr = osWaitForSingleObject(h, INFINITE);
	} while (dwErr != WAIT_OBJECT_0 && dwErr != WAIT_ABANDONED);
}
/* Release a lock acquired by winceMutexAcquire() */
#define winceMutexRelease(h) ReleaseMutex(h)

/* Create the mutex and shared memory used for locking in the file descriptor pFile */
static int winceCreateLock(const char *filename, winFile *file)
{
	LPWSTR name = winUtf8ToUnicode(zFilename);
	if (!name)
		/* out of memory */
			return RC_IOERR_NOMEM_BKPT;
	// Initialize the local lockdata
	memset(&file->local, 0, sizeof(file->local));

	// Replace the backslashes from the filename and lowercase it to derive a mutex name.
	LPWSTR tok = osCharLowerW(name);
	for (; *tok; tok++)
		if (*tok == '\\') *tok = '_';

	// Create/open the named mutex
	file->mutex = osCreateMutexW(NULL, FALSE, name);
	if (!file->mutex) {
		file->lastErrno = osGetLastError();
		mfree(name);
		return winLogError(RC_IOERR, file->lastErrno, "winceCreateLock1", filename);
	}

	// Acquire the mutex before continuing
	winceMutexAcquire(file->mutex);

	// Since the names of named mutexes, semaphores, file mappings etc are case-sensitive, take advantage of that by uppercasing the mutex name
	// and using that as the shared filemapping name.
	osCharUpperW(name);
	file->hShared = osCreateFileMappingW(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, sizeof(winceLock), name);

	// Set a flag that indicates we're the first to create the memory so it must be zero-initialized
	DWORD lastErrno = osGetLastError();
	BOOL init = TRUE;
	if (lastErrno == ERROR_ALREADY_EXISTS)
		init = FALSE;

	mfree(name);

	/* If we succeeded in making the shared memory handle, map it. */
	BOOL logged = FALSE;
	if (file->hShared) {
		file->shared = (winceLock *)osMapViewOfFile(file->shared, FILE_MAP_READ|FILE_MAP_WRITE, 0, 0, sizeof(winceLock));
		// If mapping failed, close the shared memory handle and erase it
		if (!file->shared) {
			file->lastErrno = osGetLastError();
			winLogError(RC_IOERR, file->lastErrno, "winceCreateLock2", filename);
			logged = TRUE;
			osCloseHandle(file->hShared);
			file->hShared = NULL;
		}
	}

	/* If shared memory could not be created, then close the mutex and fail */
	if (!file->hShared) {
		if (!logged) {
			file->lastErrno = lastErrno;
			winLogError(RC_IOERR, file->lastErrno, "winceCreateLock3", filename);
			logged = TRUE;
		}
		winceMutexRelease(file->mutex);
		osCloseHandle(file->mutex);
		file->mutex = NULL;
		return RC_IOERR;
	}

	// Initialize the shared memory if we're supposed to
	if (init)
		memset(file->shared, 0, sizeof(winceLock));
	winceMutexRelease(file->mutex);
	return RC_OK;
}

/* Destroy the part of winFile that deals with wince locks */
static void winceDestroyLock(winFile *file)
{
	if (file->mutex) {
		// Acquire the mutex 
		winceMutexAcquire(file->mutex);

		// The following blocks should probably assert in debug mode, but they are to cleanup in case any locks remained open
		if (file->local.readers) file->shared->readers --;
		if (file->local.reserved) file->shared->reserved = FALSE;
		if (file->local.pending) file->shared->pending = FALSE;
		if (file->local.exclusive) file->shared->exclusive = FALSE;

		// De-reference and close our copy of the shared memory handle
		osUnmapViewOfFile(file->shared);
		osCloseHandle(file->hShared);

		// Done with the mutex
		winceMutexRelease(file->mutex);
		osCloseHandle(file->mutex);
		file->mutex = NULL;
	}
}

/* An implementation of the LockFile() API of Windows for CE */
static BOOL winceLockFile(LPHANDLE phFile, DWORD dwFileOffsetLow, DWORD dwFileOffsetHigh, DWORD nNumberOfBytesToLockLow, DWORD nNumberOfBytesToLockHigh)
{
	winFile *file = HANDLE_TO_WINFILE(phFile);
	BOOL r = FALSE;

	UNUSED_SYMBOL(dwFileOffsetHigh);
	UNUSED_SYMBOL(nNumberOfBytesToLockHigh);

	if (!file->mutex) return TRUE;
	winceMutexAcquire(file->mutex);

	/* Wanting an exclusive lock? */
	if (dwFileOffsetLow == (DWORD)SHARED_FIRST && nNumberOfBytesToLockLow == (DWORD)SHARED_SIZE) {
		if (!file->shared->readers && !file->shared->exclusive) {
			file->shared->exclusive = TRUE;
			file->local.exclusive = TRUE;
			r = TRUE;
		}
	}

	// Want a read-only lock?
	else if (dwFileOffsetLow == (DWORD)SHARED_FIRST && nNumberOfBytesToLockLow == 1) {
		if (!file->shared->exclusive) {
			file->local.readers++;
			if (file->local.readers == 1)
				file->shared->readers++;
			r = TRUE;
		}
	}

	// Want a pending lock?
	else if (dwFileOffsetLow == (DWORD)PENDING_BYTE && nNumberOfBytesToLockLow == 1) {
		// If no pending lock has been acquired, then acquire it
		if (!file->shared->pending) {
			file->shared->pending = TRUE;
			file->local.pending = TRUE;
			r = TRUE;
		}
	}

	// Want a reserved lock?
	else if (dwFileOffsetLow == (DWORD)RESERVED_BYTE && nNumberOfBytesToLockLow == 1) {
		if (!file->shared->reserved) {
			file->shared->reserved = TRUE;
			file->local.reserved = TRUE;
			r = TRUE;
		}
	}
	winceMutexRelease(file->mutex);
	return r;
}

/* An implementation of the UnlockFile API of Windows for CE */
static BOOL winceUnlockFile(LPHANDLE phFile, DWORD dwFileOffsetLow, DWORD dwFileOffsetHigh, DWORD nNumberOfBytesToUnlockLow, DWORD nNumberOfBytesToUnlockHigh)
{
	winFile *file = HANDLE_TO_WINFILE(phFile);
	BOOL r = FALSE;

	UNUSED_SYMBOL(dwFileOffsetHigh);
	UNUSED_SYMBOL(nNumberOfBytesToUnlockHigh);

	if (!file->mutex) return TRUE;
	winceMutexAcquire(file->mutex);

	// Releasing a reader lock or an exclusive lock
	if (dwFileOffsetLow == (DWORD)SHARED_FIRST) {
		// Did we have an exclusive lock?
		if (file->local.exclusive) {
			assert(nNumberOfBytesToUnlockLow == (DWORD)SHARED_SIZE);
			file->local.exclusive = FALSE;
			file->shared->exclusive = FALSE;
			r = TRUE;
		}

		// Did we just have a reader lock?
		else if (file->local.readers) {
			assert(nNumberOfBytesToUnlockLow == (DWORD)SHARED_SIZE || nNumberOfBytesToUnlockLow == 1);
			file->local.readers--;
			if (!file->local.readers)
				file->shared->readers--;
			r = TRUE;
		}
	}

	// Releasing a pending lock
	else if (dwFileOffsetLow == (DWORD)PENDING_BYTE && nNumberOfBytesToUnlockLow == 1){
		if (file->local.pending) {
			file->local.pending = FALSE;
			file->shared->pending = FALSE;
			r = TRUE;
		}
	}

	// Releasing a reserved lock
	else if (dwFileOffsetLow == (DWORD)RESERVED_BYTE && nNumberOfBytesToUnlockLow == 1){
		if (file->local.reserved) {
			file->local.reserved = FALSE;
			file->shared->reserved = FALSE;
			r = TRUE;
		}
	}
	winceMutexRelease(file->mutex);
	return bReturn;
}
/*
** End of the special code for wince
*****************************************************************************/
#endif /* LIBCU_OS_WINCE */

#pragma endregion

#pragma region Locking

/* Lock a file region. */
static BOOL winLockFile(LPHANDLE phFile, DWORD flags, DWORD offsetLow, DWORD offsetHigh, DWORD numBytesLow, DWORD numBytesHigh)
{
#if LIBCU_OS_WINCE
	// NOTE: Windows CE is handled differently here due its lack of the Win32 API LockFile.
	return winceLockFile(hFile, offsetLow, offsetHigh, numBytesLow, numBytesHigh);
#else
	if (osIsNT()) {
		OVERLAPPED ovlp;
		memset(&ovlp, 0, sizeof(OVERLAPPED));
		ovlp.Offset = offsetLow;
		ovlp.OffsetHigh = offsetHigh;
		return osLockFileEx(*phFile, flags, 0, numBytesLow, numBytesHigh, &ovlp);
	}
	else return osLockFile(*phFile, offsetLow, offsetHigh, numBytesLow, numBytesHigh);
#endif
}

/* Unlock a file region. */
static BOOL winUnlockFile(LPHANDLE phFile, DWORD offsetLow, DWORD offsetHigh, DWORD numBytesLow, DWORD numBytesHigh )
{
#if LIBCU_OS_WINCE
	// NOTE: Windows CE is handled differently here due its lack of the Win32 API UnlockFile.
	return winceUnlockFile(phFile, offsetLow, offsetHigh, numBytesLow, numBytesHigh);
#else
	if (osIsNT()) {
		OVERLAPPED ovlp;
		memset(&ovlp, 0, sizeof(OVERLAPPED));
		ovlp.Offset = offsetLow;
		ovlp.OffsetHigh = offsetHigh;
		return osUnlockFileEx(*phFile, 0, numBytesLow, numBytesHigh, &ovlp);
	}
	else return osUnlockFile(*phFile, offsetLow, offsetHigh, numBytesLow, numBytesHigh);
#endif
}

#pragma endregion

#pragma region File

/*****************************************************************************
** The next group of routines implement the I/O methods specified
** by the sqlite3_io_methods object.
******************************************************************************/

/* Some Microsoft compilers lack this definition. */
#ifndef INVALID_SET_FILE_POINTER
#define INVALID_SET_FILE_POINTER ((DWORD)-1)
#endif

/*
** Move the current position of the file handle passed as the first argument to offset offset within the file. If successful, return 0.
** Otherwise, set pFile->lastErrno and return non-zero.
*/
static int winSeekFile(winFile *file, int64_t offset)
{
#if !LIBCU_OS_WINRT
	OSTRACE(("SEEK file=%p, offset=%lld\n", file->h, offset));

	LONG upperBits = (LONG)((offset>>32) & 0x7fffffff); // Most sig. 32 bits of new offset
	LONG lowerBits = (LONG)(offset & 0xffffffff); // Least sig. 32 bits of new offset

	// API oddity: If successful, SetFilePointer() returns a dword containing the lower 32-bits of the new file-offset. Or, if it fails,
	// it returns INVALID_SET_FILE_POINTER. However according to MSDN, INVALID_SET_FILE_POINTER may also be a valid new offset. So to determine
	// whether an error has actually occurred, it is also necessary to call GetLastError().
	DWORD dwRet = osSetFilePointer(file->h, lowerBits, &upperBits, FILE_BEGIN); // Value returned by SetFilePointer()

	DWORD lastErrno; // Value returned by GetLastError()
	if ((dwRet == INVALID_SET_FILE_POINTER && ((lastErrno = osGetLastError()) != NO_ERROR))) {
		file->lastErrno = lastErrno;
		winLogError(RC_IOERR_SEEK, file->lastErrno, "winSeekFile", file->path);
		OSTRACE(("SEEK file=%p, rc=LIBCU_IOERR_SEEK\n", file->h));
		return 1;
	}
	OSTRACE(("SEEK file=%p, rc=RC_OK\n", file->h));
	return 0;
#else
	// Same as above, except that this implementation works for WinRT.
	LARGE_INTEGER x; x.QuadPart = offset; // The new offset
	BOOL r = osSetFilePointerEx(file->h, x, 0, FILE_BEGIN); // Value returned by SetFilePointerEx()
	if (!r) {
		file->lastErrno = osGetLastError();
		winLogError(LIBCU_IOERR_SEEK, file->lastErrno, "winSeekFile", file->path);
		OSTRACE(("SEEK file=%p, rc=LIBCU_IOERR_SEEK\n", file->h));
		return 1;
	}
	OSTRACE(("SEEK file=%p, rc=RC_OK\n", file->h));
	return 0;
#endif
}

#if LIBCU_MAXMMAPSIZE > 0
/* Forward references to VFS helper methods used for memory mapped files */
static int winMapfile(winFile *, int64_t);
static int winUnmapfile(winFile *);
#endif

/*
** Close a file.
**
** It is reported that an attempt to close a handle might sometimes fail.  This is a very unreasonable result, but Windows is notorious
** for being unreasonable so I do not doubt that it might happen.  If the close fails, we pause for 100 milliseconds and try again.  As
** many as MX_CLOSE_ATTEMPT attempts to close the handle are made before giving up and returning an error.
*/
#define MX_CLOSE_ATTEMPT 3
static RC winClose(vsysfile *id)
{
	winFile *file = (winFile *)id;
	assert(id);
#ifndef LIBCU_OMIT_WAL
	assert(!file->shm);
#endif
	assert(file->h && file->h != INVALID_HANDLE_VALUE);
	OSTRACE(("CLOSE pid=%lu, pFile=%p, file=%p\n", osGetCurrentProcessId(), file, file->h));
#if LIBCU_MAXMMAPSIZE > 0
	winUnmapfile(file);
#endif
	RC rc;
	int cnt = 0;
	do { rc = osCloseHandle(file->h); } // SimulateIOError( rc=0; cnt=MX_CLOSE_ATTEMPT; );
	while (!rc && ++cnt < MX_CLOSE_ATTEMPT && (sqlite3_win32_sleep(100), 1));
#if LIBCU_OS_WINCE
#define WINCE_DELETION_ATTEMPTS 3
	{
		WinVfsAppData *appData = (WinVfsAppData *)file->system->appData;
		if (!appData || !appData->noLock)
			winceDestroyLock(file);
	}
	if (file->deleteOnClose) {
		int cnt = 0;
		while (!osDeleteFileW(file->deleteOnClose) && osGetFileAttributesW(file->deleteOnClose) != 0xffffffff && cnt++ < WINCE_DELETION_ATTEMPTS)
			sqlite3_win32_sleep(100);  // Wait a little before trying again
		mfree(pFile->zDeleteOnClose);
	}
#endif
	if (rc)
		file->h = NULL;
	OpenCounter(-1);
	OSTRACE(("CLOSE pid=%lu, pFile=%p, file=%p, rc=%s\n", osGetCurrentProcessId(), file, file->h, rc ? "ok" : "failed"));
	return rc ? RC_OK : winLogError(LIBCU_IOERR_CLOSE, osGetLastError(), "winClose", file->path);
}

/* Read data from a file into a buffer.  Return RC_OK if all bytes were read successfully and LIBCU_IOERR if anything goes wrong. */
static RC winRead(vsysfile *id, void *buf, int amt, int64_t offset)
{
	winFile *file = (winFile *)id;  // file handle
	int retry = 0; // Number of retrys
	assert(id);
	assert(amt > 0);
	assert(offset >= 0);
	SimulateIOError(return RC_IOERR_READ);
	OSTRACE(("READ pid=%lu, pFile=%p, file=%p, buffer=%p, amount=%d, offset=%lld, lock=%d\n", osGetCurrentProcessId(), file, file->h, buf, amt, offset, file->locktype));
	DWORD read; // Number of bytes actually read from file
#if LIBCU_MAXMMAPSIZE > 0
	// Deal with as much of this read request as possible by transfering data from the memory mapping using memcpy().
	if (offset < file->mmapSize) {
		if (offset+amt <= file->mmapSize) {
			memcpy(buf, &((uint8_t *)(file->mapRegion))[offset], amt);
			OSTRACE(("READ-MMAP pid=%lu, pFile=%p, file=%p, rc=RC_OK\n", osGetCurrentProcessId(), pFile, file->h));
			return RC_OK;
		}
		else {
			int copy = (int)(file->mmapSize - offset);
			memcpy(buf, &((uint8_t *)(file->mapRegion))[offset], copy);
			buf = &((uint8_t *)buf)[copy];
			amt -= copy;
			offset += copy;
		}
	}
#endif
#if !LIBCU_OS_WINCE && !defined(LIBCU_WIN32_NO_OVERLAPPED)
	OVERLAPPED overlapped; // The offset for ReadFile.
#endif
#if LIBCU_OS_WINCE || defined(LIBCU_WIN32_NO_OVERLAPPED)
	if (winSeekFile(file, offset)) {
		OSTRACE(("READ pid=%lu, pFile=%p, file=%p, rc=LIBCU_FULL\n", osGetCurrentProcessId(), file, file->h));
		return RC_FULL;
	}
	while (!osReadFile(file->h, buf, amt, &read, 0)) {
#else
	memset(&overlapped, 0, sizeof(OVERLAPPED));
	overlapped.Offset = (LONG)(offset & 0xffffffff);
	overlapped.OffsetHigh = (LONG)((offset>>32) & 0x7fffffff);
	while (!osReadFile(file->h, buf, amt, &read, &overlapped) && osGetLastError() != ERROR_HANDLE_EOF) {
#endif
		DWORD lastErrno;
		if (winRetryIoerr(&retry, &lastErrno)) continue;
		file->lastErrno = lastErrno;
		OSTRACE(("READ pid=%lu, pFile=%p, file=%p, rc=LIBCU_IOERR_READ\n", osGetCurrentProcessId(), file, file->h));
		return winLogError(RC_IOERR_READ, file->lastErrno, "winRead", file->path);
	}
	winLogIoerr(retry, __LINE__);
	if (read < (DWORD)amt) {
		// Unread parts of the buffer must be zero-filled
		memset(&((char *)buf)[read], 0, amt-read);
		OSTRACE(("READ pid=%lu, pFile=%p, file=%p, rc=LIBCU_IOERR_SHORT_READ\n", osGetCurrentProcessId(), file, file->h));
		return RC_IOERR_SHORT_READ;
	}
	OSTRACE(("READ pid=%lu, pFile=%p, file=%p, rc=RC_OK\n", osGetCurrentProcessId(), file, file->h));
	return RC_OK;
}

/* Write data from a buffer into a file.  Return RC_OK on success or some other error code on failure. */
static RC winWrite(vsysfile *id, const void *buf, int amt, int64_t offset)
{
	RC rc = 0; // True if error has occurred, else false
	winFile *file = (winFile *)id; // File handle
	int retry = 0; // Number of retries
	assert(amt > 0);
	assert(file);
	SimulateIOError(return RC_IOERR_WRITE);
	SimulateDiskfullError(return RC_FULL);
	OSTRACE(("WRITE pid=%lu, pFile=%p, file=%p, buffer=%p, amount=%d, offset=%lld, lock=%d\n", osGetCurrentProcessId(), file, file->h, buf, amt, offset, file->locktype));
#if defined(LIBCU_MMAP_READWRITE) && LIBCU_MAXMMAPSIZE>0
	// Deal with as much of this write request as possible by transfering data from the memory mapping using memcpy().  */
	if (offset < file->mmapSize) {
		if (offset+amt <= file->mmapSize) {
			memcpy(&((uint8_t *)(file->mapRegion))[offset], buf, amt);
			OSTRACE(("WRITE-MMAP pid=%lu, pFile=%p, file=%p, rc=RC_OK\n", osGetCurrentProcessId(), file, file->h));
			return RC_OK;
		}
		else {
			int copy = (int)(file->mmapSize - offset);
			memcpy(&((uint8_t *)(file->mapRegion))[offset], buf, copy);
			buf = &((uint8_t *)buf)[copy];
			amt -= copy;
			offset += copy;
		}
	}
#endif
#if LIBCU_OS_WINCE || defined(LIBCU_WIN32_NO_OVERLAPPED)
	rc = winSeekFile(pFile, offset);
	if (!rc) {
#else
	{
#endif
#if !LIBCU_OS_WINCE && !defined(LIBCU_WIN32_NO_OVERLAPPED)
		OVERLAPPED overlapped; // The offset for WriteFile.
#endif
		uint8_t *rem = (uint8_t *)buf; // Data yet to be written
		int remAmt = amt;               // Number of bytes yet to be written
		DWORD write;                 // Bytes written by each WriteFile() call
		DWORD lastErrno = NO_ERROR;   // Value returned by GetLastError()
#if !LIBCU_OS_WINCE && !defined(LIBCU_WIN32_NO_OVERLAPPED)
		memset(&overlapped, 0, sizeof(OVERLAPPED));
		overlapped.Offset = (LONG)(offset & 0xffffffff);
		overlapped.OffsetHigh = (LONG)((offset>>32) & 0x7fffffff);
#endif
		while (remAmt > 0) {
#if LIBCU_OS_WINCE || defined(LIBCU_WIN32_NO_OVERLAPPED)
			if (!osWriteFile(file->h, rem, remAmt, &write, 0)) {
#else
			if (!osWriteFile(file->h, rem, remAmt, &write, &overlapped)) {
#endif
				if (winRetryIoerr(&retry, &lastErrno)) continue;
				break;
			}
			assert(!write || write <= (DWORD)remAmt);
			if (!write || write > (DWORD)remAmt) {
				lastErrno = osGetLastError();
				break;
			}
#if !LIBCU_OS_WINCE && !defined(LIBCU_WIN32_NO_OVERLAPPED)
			offset += write;
			overlapped.Offset = (LONG)(offset & 0xffffffff);
			overlapped.OffsetHigh = (LONG)((offset>>32) & 0x7fffffff);
#endif
			rem += write;
			remAmt -= write;
		}
		if (remAmount > 0) {
			file->lastErrno = lastErrno;
			rc = 1;
		}
	}
	if (rc) {
		if (file->lastErrno == ERROR_HANDLE_DISK_FULL) || file->lastErrno == ERROR_DISK_FULL) {
			OSTRACE(("WRITE pid=%lu, pFile=%p, file=%p, rc=LIBCU_FULL\n", osGetCurrentProcessId(), file, file->h));
			return winLogError(RC_FULL, file->lastErrno, "winWrite1", file->path);
		}
		OSTRACE(("WRITE pid=%lu, pFile=%p, file=%p, rc=LIBCU_IOERR_WRITE\n", osGetCurrentProcessId(), file, file->h));
		return winLogError(RC_IOERR_WRITE, file->lastErrno, "winWrite2", file->path);
	}
	else winLogIoerr(retry, __LINE__);
	OSTRACE(("WRITE pid=%lu, pFile=%p, file=%p, rc=RC_OK\n", osGetCurrentProcessId(), file, file->h));
	return RC_OK;
}

/* Truncate an open file to a specified size */
static RC winTruncate(vsysfile *id, int64_t size)
{
	winFile *file = (winFile *)id; // File handle object
	assert(file);
	SimulateIOError(return RC_IOERR_TRUNCATE);
	OSTRACE(("TRUNCATE pid=%lu, pFile=%p, file=%p, size=%lld, lock=%d\n", osGetCurrentProcessId(), file, file->h, size, file->locktype));
	// If the user has configured a chunk-size for this file, truncate the file so that it consists of an integer number of chunks (i.e. the
	// actual file size after the operation may be larger than the requested size).
	if (file->sizeChunk > 0)
		size = ((size + file->sizeChunk - 1) / file->sizeChunk) * file->sizeChunk;
	// SetEndOfFile() returns non-zero when successful, or zero when it fails.
	RC rc = RC_OK; // Return code for this function
	DWORD lastErrno;
	if (winSeekFile(file, size)) 
		rc = winLogError(RC_IOERR_TRUNCATE, file->lastErrno, "winTruncate1", file->path);
	else if (!osSetEndOfFile(file->h) && (lastErrno = osGetLastError()) != ERROR_USER_MAPPED_FILE) {
		file->lastErrno = lastErrno;
		rc = winLogError(RC_IOERR_TRUNCATE, file->lastErrno, "winTruncate2", file->path);
	}
#if LIBCU_MAXMMAPSIZE>0
	// If the file was truncated to a size smaller than the currently mapped region, reduce the effective mapping size as well. Libcu will
	// use read() and write() to access data beyond this point from now on.
	if (file->mapRegion && size < file->mmapSize)
		file->mmapSize = size;
#endif
	OSTRACE(("TRUNCATE pid=%lu, pFile=%p, file=%p, rc=%s\n", osGetCurrentProcessId(), file, file->h, runtimeErrName(rc)));
	return rc;
}

#ifdef LIBCU_TEST
/* Count the number of fullsyncs and normal syncs.  This is used to test that syncs and fullsyncs are occuring at the right times. */
int sqlite3_sync_count = 0;
int sqlite3_fullsync_count = 0;
#endif

/* Make sure all writes to a particular file are committed to disk. */
static RC winSync(vsysfile *id, int flags)
{
#ifndef LIBCU_NO_SYNC
	/* Used only when LIBCU_NO_SYNC is not defined. */
	BOOL rc;
#endif
#if !defined(NDEBUG) || !defined(LIBCU_NO_SYNC) || defined(LIBCU_HAVE_OS_TRACE)
	// Used when LIBCU_NO_SYNC is not defined and by the assert() and/or OSTRACE() macros.
	winFile *file = (winFile *)id;
#else
	UNUSED_SYMBOL(id);
#endif
	assert(file);
	// Check that one of LIBCU_SYNC_NORMAL or FULL was passed
	assert((flags&0x0F) == LIBCU_SYNC_NORMAL || (flags&0x0F) == LIBCU_SYNC_FULL);
	// Unix cannot, but some systems may return LIBCU_FULL from here. This line is to test that doing so does not cause any problems.
	SimulateDiskfullError( return LIBCU_FULL );
	OSTRACE(("SYNC pid=%lu, pFile=%p, file=%p, flags=%x, lock=%d\n", osGetCurrentProcessId(), file, file->h, flags, file->locktype));
#ifndef LIBCU_TEST
	UNUSED_SYMBOL(flags);
#else
	if ((flags&0x0F) == LIBCU_SYNC_FULL)
		sqlite3_fullsync_count++;
	sqlite3_sync_count++;
#endif
	// If we compiled with the LIBCU_NO_SYNC flag, then syncing is a no-op
#ifdef LIBCU_NO_SYNC
	OSTRACE(("SYNC-NOP pid=%lu, pFile=%p, file=%p, rc=RC_OK\n", osGetCurrentProcessId(), file, file->h));
	return RC_OK;
#else
#if LIBCU_MAXMMAPSIZE>0
	if (file->mapRegion) {
		if (osFlushViewOfFile(file->mapRegion, 0))
			OSTRACE(("SYNC-MMAP pid=%lu, pFile=%p, pMapRegion=%p, " "rc=RC_OK\n", osGetCurrentProcessId(), file, file->mapRegion));
		else {
			file->lastErrno = osGetLastError();
			OSTRACE(("SYNC-MMAP pid=%lu, pFile=%p, pMapRegion=%p, rc=LIBCU_IOERR_MMAP\n", osGetCurrentProcessId(), file, file->mapRegion));
			return winLogError(RC_IOERR_MMAP, file->lastErrno, "winSync1", file->path);
		}
	}
#endif
	rc = osFlushFileBuffers(file->h);
	SimulateIOError(rc = FALSE);
	if (rc) {
		OSTRACE(("SYNC pid=%lu, pFile=%p, file=%p, rc=RC_OK\n", osGetCurrentProcessId(), file, file->h));
		return RC_OK;
	}
	else {
		file->lastErrno = osGetLastError();
		OSTRACE(("SYNC pid=%lu, pFile=%p, file=%p, rc=LIBCU_IOERR_FSYNC\n", osGetCurrentProcessId(), file, file->h));
		return winLogError(RC_IOERR_FSYNC, file->lastErrno, "winSync2", file->path);
	}
#endif
}

/* Determine the current size of a file in bytes */
static RC winFileSize(vsysfile *id, int64_t *size)
{
	winFile *file = (winFile *)id;
	RC rc = RC_OK;
	assert(id);
	assert(size);
	SimulateIOError(return RC_IOERR_FSTAT);
	OSTRACE(("SIZE file=%p, pSize=%p\n", file->h, size));
#if LIBCU_OS_WINRT
	{
		FILE_STANDARD_INFO info;
		if (osGetFileInformationByHandleEx(file->h, FileStandardInfo, &info, sizeof(info)))
			*size = info.EndOfFile.QuadPart;
		else {
			file->lastErrno = osGetLastError();
			rc = winLogError(RC_IOERR_FSTAT, file->lastErrno, "winFileSize", file->path);
		}
	}
#else
	{
		DWORD upperBits;
		DWORD lowerBits = osGetFileSize(file->h, &upperBits);
		DWORD lastErrno;
		*size = (((int64_t)upperBits)<<32) + lowerBits;
		if (lowerBits == INVALID_FILE_SIZE && (lastErrno = osGetLastError()) != NO_ERROR) {
			file->lastErrno = lastErrno;
			rc = winLogError(RC_IOERR_FSTAT, file->lastErrno, "winFileSize", file->path);
		}
	}
#endif
	OSTRACE(("SIZE file=%p, pSize=%p, *pSize=%lld, rc=%s\n", file->h, size, *size, runtimeErrName(rc)));
	return rc;
}

/* LOCKFILE_FAIL_IMMEDIATELY is undefined on some Windows systems. */
#ifndef LOCKFILE_FAIL_IMMEDIATELY
#define LOCKFILE_FAIL_IMMEDIATELY 1
#endif

#ifndef LOCKFILE_EXCLUSIVE_LOCK
#define LOCKFILE_EXCLUSIVE_LOCK 2
#endif

/*
** Historically, Libcu has used both the LockFile and LockFileEx functions. When the LockFile function was used, it was always expected to fail
** immediately if the lock could not be obtained.  Also, it always expected to obtain an exclusive lock.  These flags are used with the LockFileEx function
** and reflect those expectations; therefore, they should not be changed.
*/
#ifndef LIBCU_LOCKFILE_FLAGS
#define LIBCU_LOCKFILE_FLAGS   (LOCKFILE_FAIL_IMMEDIATELY|LOCKFILE_EXCLUSIVE_LOCK)
#endif

/*
** Currently, Libcu never calls the LockFileEx function without wanting the call to fail immediately if the lock cannot be obtained.
*/
#ifndef LIBCU_LOCKFILEEX_FLAGS
#define LIBCU_LOCKFILEEX_FLAGS (LOCKFILE_FAIL_IMMEDIATELY)
#endif

/*
** Acquire a reader lock.
** Different API routines are called depending on whether or not this is Win9x or WinNT.
*/
static int winGetReadLock(winFile *file)
{
	OSTRACE(("READ-LOCK file=%p, lock=%d\n", file->h, file->locktype));
	int res;
	if (osIsNT()) {
#if LIBCU_OS_WINCE
		// NOTE: Windows CE is handled differently here due its lack of the Win32 API LockFileEx.
		res = winceLockFile(&file->h, SHARED_FIRST, 0, 1, 0);
#else
		res = winLockFile(&file->h, LIBCU_LOCKFILEEX_FLAGS, SHARED_FIRST, 0, SHARED_SIZE, 0);
#endif
	}
#ifdef LIBCU_WIN32_HAS_ANSI
	else {
		int lk;
		sqlite3_randomness(sizeof(lk), &lk);
		file->sharedLockByte = (short)((lk&0x7fffffff) % (SHARED_SIZE-1));
		res = winLockFile(&file->h, LIBCU_LOCKFILE_FLAGS, SHARED_FIRST+file->sharedLockByte, 0, 1, 0);
	}
#endif
	if (!res)
		file->lastErrno = osGetLastError(); // No need to log a failure to lock
	OSTRACE(("READ-LOCK file=%p, result=%d\n", file->h, res));
	return res;
}

/* Undo a readlock */
static int winUnlockReadLock(winFile *file)
{
	OSTRACE(("READ-UNLOCK file=%p, lock=%d\n", file->h, file->locktype));
	int res;
	if (osIsNT())
		res = winUnlockFile(&file->h, SHARED_FIRST, 0, SHARED_SIZE, 0);
#ifdef LIBCU_WIN32_HAS_ANSI
	else
		res = winUnlockFile(&file->h, SHARED_FIRST+pFile->sharedLockByte, 0, 1, 0);
#endif
	DWORD lastErrno;
	if (!res && (lastErrno = osGetLastError()) != ERROR_NOT_LOCKED) {
		file->lastErrno = lastErrno;
		winLogError(LIBCU_IOERR_UNLOCK, file->lastErrno, "winUnlockReadLock", file->path);
	}
	OSTRACE(("READ-UNLOCK file=%p, result=%d\n", file->h, res));
	return res;
}

/*
** Lock the file with the lock specified by parameter locktype - one of the following:
**
**     (1) SHARED_LOCK
**     (2) RESERVED_LOCK
**     (3) PENDING_LOCK
**     (4) EXCLUSIVE_LOCK
**
** Sometimes when requesting one lock state, additional lock states are inserted in between.  The locking might fail on one of the later
** transitions leaving the lock state different from what it started but still short of its goal.  The following chart shows the allowed
** transitions and the inserted intermediate states:
**
**    UNLOCKED -> SHARED
**    SHARED -> RESERVED
**    SHARED -> (PENDING) -> EXCLUSIVE
**    RESERVED -> (PENDING) -> EXCLUSIVE
**    PENDING -> EXCLUSIVE
**
** This routine will only increase a lock.  The winUnlock() routine erases all locks at once and returns us immediately to locking level 0.
** It is not possible to lower the locking level one step at a time.  You must go straight to locking level 0.
*/
static RC winLock(vsysfile *id, int locktype)
{
	RC rc = RC_OK; // Return code from subroutines
	int res = 1; // Result of a Windows lock call
	winFile *file = (winFile *)id;
	assert(id);
	OSTRACE(("LOCK file=%p, oldLock=%d(%d), newLock=%d\n", file->h, file->locktype, file->sharedLockByte, locktype));

	// If there is already a lock of this type or more restrictive on the OsFile, do nothing. Don't use the end_lock: exit path, as
	// sqlite3OsEnterMutex() hasn't been called yet.
	if (file->locktype >= locktype) {
		OSTRACE(("LOCK-HELD file=%p, rc=RC_OK\n", file->h));
		return RC_OK;
	}

	// Do not allow any kind of write-lock on a read-only database
	if ((file->ctrlFlags & WINFILE_RDONLY) != 0 && locktype >= RESERVED_LOCK)
		return RC_IOERR_LOCK;

	// Make sure the locking sequence is correct
	assert(file->locktype != NO_LOCK || locktype == SHARED_LOCK);
	assert(locktype != PENDING_LOCK );
	assert(locktype != RESERVED_LOCK || pFile->locktype == SHARED_LOCK);

	// Lock the PENDING_LOCK byte if we need to acquire a PENDING lock or a SHARED lock.  If we are acquiring a SHARED lock, the acquisition of
	// the PENDING_LOCK byte is temporary.
	int gotPendingLock = 0; // True if we acquired a PENDING lock this time
	int newLocktype = file->locktype; // Set file->locktype to this value before exiting
	DWORD lastErrno = NO_ERROR;
	if (file->locktype == NO_LOCK || (locktype == EXCLUSIVE_LOCK && file->locktype <= RESERVED_LOCK)) {
		int cnt = 3;
		while (cnt-- > 0 && !(res = winLockFile(&file->h, LIBCU_LOCKFILE_FLAGS, PENDING_BYTE, 0, 1, 0))) {
			// Try 3 times to get the pending lock.  This is needed to work around problems caused by indexing and/or anti-virus software on Windows systems.
			// If you are using this code as a model for alternative VFSes, do not copy this retry logic.  It is a hack intended for Windows only.
			lastErrno = osGetLastError();
			OSTRACE(("LOCK-PENDING-FAIL file=%p, count=%d, result=%d\n", file->h, cnt, res));
			if (lastErrno == ERROR_INVALID_HANDLE) {
				file->lastErrno = lastErrno;
				rc = RC_IOERR_LOCK;
				OSTRACE(("LOCK-FAIL file=%p, count=%d, rc=%s\n", file->h, cnt, runtimeErrName(rc)));
				return rc;
			}
			if (cnt) sqlite3_win32_sleep(1);
		}
		gotPendingLock = res;
		if (!res)
			lastErrno = osGetLastError();
	}

	// Acquire a shared lock
	if (locktype == SHARED_LOCK && res) {
		assert(file->locktype == NO_LOCK);
		res = winGetReadLock(file);
		if (res) newLocktype = SHARED_LOCK;
		else lastErrno = osGetLastError();
	}

	// Acquire a RESERVED lock
	if (locktype == RESERVED_LOCK && res) {
		assert(file->locktype == SHARED_LOCK);
		res = winLockFile(&file->h, LIBCU_LOCKFILE_FLAGS, RESERVED_BYTE, 0, 1, 0);
		if (res) newLocktype = RESERVED_LOCK;
		else lastErrno = osGetLastError();
	}

	// Acquire a PENDING lock
	if (locktype == EXCLUSIVE_LOCK && res) {
		newLocktype = PENDING_LOCK;
		gotPendingLock = 0;
	}

	// Acquire an EXCLUSIVE lock
	if (locktype == EXCLUSIVE_LOCK && res) {
		assert(file->locktype >= SHARED_LOCK);
		res = winUnlockReadLock(file);
		res = winLockFile(&file->h, LIBCU_LOCKFILE_FLAGS, SHARED_FIRST, 0, SHARED_SIZE, 0);
		if (res) newLocktype = EXCLUSIVE_LOCK;
		else { lastErrno = osGetLastError(); winGetReadLock(file); }
	}

	// If we are holding a PENDING lock that ought to be released, then release it now.
	if (gotPendingLock && locktype == SHARED_LOCK)
		winUnlockFile(&file->h, PENDING_BYTE, 0, 1, 0);

	// Update the state of the lock has held in the file descriptor then return the appropriate result code.
	if (res)
		rc = RC_OK;
	else {
		file->lastErrno = lastErrno;
		rc = RC_BUSY;
		OSTRACE(("LOCK-FAIL file=%p, wanted=%d, got=%d\n", file->h, locktype, newLocktype));
	}
	file->locktype = (uint8_t)newLocktype;
	OSTRACE(("LOCK file=%p, lock=%d, rc=%s\n", file->h, file->locktype, runtimeErrName(rc)));
	return rc;
}

/*
** This routine checks if there is a RESERVED lock held on the specified file by this or any other process. If such a lock is held, return
** non-zero, otherwise zero.
*/
static RC winCheckReservedLock(vsysfile *id, int *resOut)
{
	winFile *file = (winFile *)id;
	SimulateIOError(return RC_IOERR_CHECKRESERVEDLOCK;);
	OSTRACE(("TEST-WR-LOCK file=%p, pResOut=%p\n", file->h, resOut));
	assert(id);
	int res;
	if (file->locktype >= RESERVED_LOCK) {
		res = 1;
		OSTRACE(("TEST-WR-LOCK file=%p, result=%d (local)\n", file->h, res));
	}
	else {
		res = winLockFile(&file->h, LIBCU_LOCKFILEEX_FLAGS, RESERVED_BYTE, 0, 1, 0);
		if (res)
			winUnlockFile(&file->h, RESERVED_BYTE, 0, 1, 0);
		res = !res;
		OSTRACE(("TEST-WR-LOCK file=%p, result=%d (remote)\n", file->h, res));
	}
	*resOut = res;
	OSTRACE(("TEST-WR-LOCK file=%p, pResOut=%p, *pResOut=%d, rc=RC_OK\n", file->h, resOut, *resOut));
	return RC_OK;
}

/*
** Lower the locking level on file descriptor id to locktype.  locktype must be either NO_LOCK or SHARED_LOCK.
**
** If the locking level of the file descriptor is already at or below the requested locking level, this routine is a no-op.
**
** It is not possible for this routine to fail if the second argument is NO_LOCK.  If the second argument is SHARED_LOCK then this routine
** might return LIBCU_IOERR;
*/
static RC winUnlock(vsysfile *id, int locktype)
{
	winFile *file = (winFile *)id;
	RC rc = RC_OK;
	assert(file);
	assert(locktype <= SHARED_LOCK);
	OSTRACE(("UNLOCK file=%p, oldLock=%d(%d), newLock=%d\n", file->h, file->locktype, file->sharedLockByte, locktype));
	int type = file->locktype;
	if (type >= EXCLUSIVE_LOCK) {
		winUnlockFile(&file->h, SHARED_FIRST, 0, SHARED_SIZE, 0);
		if (locktype == SHARED_LOCK && !winGetReadLock(file)) {
			// This should never happen.  We should always be able to reacquire the read lock
			rc = winLogError(LIBCU_IOERR_UNLOCK, osGetLastError(), "winUnlock", file->path);
		}
	}
	if (type >= RESERVED_LOCK)
		winUnlockFile(&file->h, RESERVED_BYTE, 0, 1, 0);
	if (locktype == NO_LOCK && type >= SHARED_LOCK)
		winUnlockReadLock(pFile);
	if (type >= PENDING_LOCK)
		winUnlockFile(&file->h, PENDING_BYTE, 0, 1, 0);
	file->locktype = (uint8_t)locktype;
	OSTRACE(("UNLOCK file=%p, lock=%d, rc=%s\n", file->h, file->locktype, runtimeErrName(rc)));
	return rc;
}

/******************************************************************************
****************************** No-op Locking **********************************
**
** Of the various locking implementations available, this is by far the simplest:  locking is ignored.  No attempt is made to lock the database
** file for reading or writing.
**
** This locking mode is appropriate for use on read-only databases (ex: databases that are burned into CD-ROM, for example.)  It can
** also be used if the application employs some external mechanism to prevent simultaneous access of the same database by two or more
** database connections.  But there is a serious risk of database corruption if this locking mode is used in situations where multiple
** database connections are accessing the same database file at the same time and one or more of those connections are writing.
*/

static RC winNolockLock(vsysfile *id, int locktype)
{
	UNUSED_SYMBOL(id);
	UNUSED_SYMBOL(locktype);
	return RC_OK;
}

static RC winNolockCheckReservedLock(vsysfile *id, int *resOut)
{
	UNUSED_SYMBOL(id);
	UNUSED_SYMBOL(pResOut);
	return RC_OK;
}

static RC winNolockUnlock(vsysfile *id, int locktype)
{
	UNUSED_SYMBOL(id);
	UNUSED_SYMBOL(locktype);
	return RC_OK;
}

/******************* End of the no-op lock implementation *********************
******************************************************************************/

/*
** If *pArg is initially negative then this is a query.  Set *pArg to 1 or 0 depending on whether or not bit mask of pFile->ctrlFlags is set.
**
** If *pArg is 0 or 1, then clear or set the mask bit of pFile->ctrlFlags.
*/
static void winModeBit(winFile *file, unsigned char mask, int *arg)
{
	if (*arg < 0) *arg = (file->ctrlFlags & mask) != 0;
	else if (!(*arg)) file->ctrlFlags &= ~mask;
	else file->ctrlFlags |= mask;
}

/* Forward references to VFS helper methods used for temporary files */
static int winGetTempname(vsystem *, char **);
static int winIsDir(const void *);
static BOOL winIsDriveLetterAndColon(const char *);

/* Control and query of the open file handle. */
static int winFileControl(vsysfile *id, FCNTL op, void *arg)
{
	winFile *file = (winFile *)id;
	OSTRACE(("FCNTL file=%p, op=%d, pArg=%p\n", file->h, op, arg));
	switch (op) {
	case FCNTL_LOCKSTATE: {
		*(int *)arg = file->locktype;
		OSTRACE(("FCNTL file=%p, rc=RC_OK\n", file->h));
		return RC_OK; }
	case FCNTL_LAST_ERRNO: {
		*(int *)arg = (int)file->lastErrno;
		OSTRACE(("FCNTL file=%p, rc=RC_OK\n", file->h));
		return RC_OK; }
	case FCNTL_CHUNK_SIZE: {
		file->sizeChunk = *(int *)arg;
		OSTRACE(("FCNTL file=%p, rc=RC_OK\n", file->h));
		return RC_OK; }
	case FCNTL_SIZE_HINT: {
		if (file->sizeChunk > 0) {
			int64_t oldSize;
			RC rc = winFileSize(id, &oldSize);
			if (rc == RC_OK) {
				int64_t newSize = *(int64_t *)arg;
				if (newSize > oldSize) {
					SimulateIOErrorBenign(1);
					rc = winTruncate(id, newSize);
					SimulateIOErrorBenign(0);
				}
			}
			OSTRACE(("FCNTL file=%p, rc=%s\n", file->h, runtimeErrName(rc)));
			return rc;
		}
		OSTRACE(("FCNTL file=%p, rc=RC_OK\n", file->h));
		return RC_OK; }
	case FCNTL_PERSIST_WAL: {
		winModeBit(file, WINFILE_PERSIST_WAL, (int *)arg);
		OSTRACE(("FCNTL file=%p, rc=RC_OK\n", file->h));
		return RC_OK; }
	case FCNTL_POWERSAFE_OVERWRITE: {
		winModeBit(file, WINFILE_PSOW, (int *)arg);
		OSTRACE(("FCNTL file=%p, rc=RC_OK\n", file->h));
		return RC_OK; }
	case FCNTL_VFSNAME: {
		*(char **)arg = mprintf("%s", file->system->name);
		OSTRACE(("FCNTL file=%p, rc=RC_OK\n", file->h));
		return RC_OK; }
	case FCNTL_WIN32_AV_RETRY: {
		int *a = (int *)arg;
		if (a[0] > 0) winIoerrRetry = a[0];
		else a[0] = winIoerrRetry;
		if (a[1] > 0) winIoerrRetryDelay = a[1];
		else a[1] = winIoerrRetryDelay;
		OSTRACE(("FCNTL file=%p, rc=RC_OK\n", file->h));
		return RC_OK; }
	case FCNTL_WIN32_GET_HANDLE: {
		LPHANDLE phFile = (LPHANDLE)arg;
		*phFile = file->h;
		OSTRACE(("FCNTL file=%p, rc=RC_OK\n", file->h));
		return RC_OK; }
#ifdef LIBCU_TEST
	case FCNTL_WIN32_SET_HANDLE: {
		LPHANDLE phFile = (LPHANDLE)arg;
		HANDLE hOldFile = file->h;
		file->h = *phFile;
		*phFile = hOldFile;
		OSTRACE(("FCNTL oldFile=%p, newFile=%p, rc=RC_OK\n", hOldFile, file->h));
		return RC_OK; }
#endif
	case FCNTL_TEMPFILENAME: {
		char *tfile = nullptr;
		RC rc = winGetTempname(file->system, &tfile);
		if (rc == RC_OK)
			*(char **)aArg = tfile;
		OSTRACE(("FCNTL file=%p, rc=%s\n", file->h, runtimeErrName(rc)));
		return rc; }
#if LIBCU_MAXMMAPSIZE>0
	case FCNTL_MMAP_SIZE: {
		int64_t newLimit = *(int64_t *)arg;
		RC rc = RC_OK;
		if (newLimit > _runtimeConfig.maxMmap)
			newLimit = _runtimeConfig.maxMmap;
		*(int64_t *)arg = file->mmapSizeMax;
		if (newLimit >= 0 && newLimit != file->mmapSizeMax && !file->fetchOuts) {
			file->mmapSizeMax = newLimit;
			if (file->mmapSize > 0) {
				winUnmapfile(file);
				rc = winMapfile(file, -1);
			}
		}
		OSTRACE(("FCNTL file=%p, rc=%s\n", file->h, runtimeErrName(rc)));
		return rc; }
#endif
	}
	OSTRACE(("FCNTL file=%p, rc=LIBCU_NOTFOUND\n", file->h));
	return RC_NOTFOUND;
}

/*
** Return the sector size in bytes of the underlying block device for the specified file. This is almost always 512 bytes, but may be
** larger for some devices.
**
** Libcu code assumes this function cannot fail. It also assumes that if two files are created in the same file-system directory (i.e.
** a database and its journal file) that the sector size will be the same for both.
*/
static int winSectorSize(vsysfile *id)
{
	(void)id;
	return LIBCU_DEFAULT_SECTOR_SIZE;
}

/* Return a vector of device characteristics. */
static int winDeviceCharacteristics(vsysfile *id)
{
	winFile *p = (winFile *)id;
	return LIBCU_IOCAP_UNDELETABLE_WHEN_OPEN | ((p->ctrlFlags & WINFILE_PSOW) ? LIBCU_IOCAP_POWERSAFE_OVERWRITE : 0);
}

/*
** Windows will only let you create file view mappings on allocation size granularity boundaries.
** During sqlite3_os_init() we do a GetSystemInfo() to get the granularity size.
*/
static SYSTEM_INFO winSysInfo;

#pragma endregion

#pragma region WAL
#ifndef LIBCU_OMIT_WAL

/*
** Helper functions to obtain and relinquish the global mutex. The global mutex is used to protect the winLockInfo objects used by
** this file, all of which may be shared by multiple threads.
**
** Function winShmMutexHeld() is used to assert() that the global mutex is held when required. This function is only used as part of assert()
** statements. e.g.
**
**   winShmEnterMutex()
**     assert( winShmMutexHeld() );
**   winShmLeaveMutex()
*/
static void winShmEnterMutex()
{
	mutex_enter(mutexAlloc(MUTEX_STATIC_VFS1));
}
static void winShmLeaveMutex()
{
	mutex_leave(mutexAlloc(MUTEX_STATIC_VFS1));
}
#ifndef NDEBUG
static int winShmMutexHeld()
{
	return mutex_held(mutexAlloc(MUTEX_STATIC_VFS1));
}
#endif

/*
** Object used to represent a single file opened and mmapped to provide shared memory.  When multiple threads all reference the same
** log-summary, each thread has its own winFile object, but they all point to a single instance of this object.  In other words, each
** log-summary is opened only once per process.
**
** winShmMutexHeld() must be true when creating or destroying this object or while reading or writing the following fields:
**      nRef
**      pNext
**
** The following fields are read-only after the object is created:
**      fid
**      zFilename
**
** Either winShmNode.mutex must be held or winShmNode.nRef==0 and winShmMutexHeld() is true when reading or writing any other field
** in this structure.
*/
struct winShmNode {
	mutex *mutex;			// Mutex to access this object
	char *filename;        // Name of the file
	winFile file;           // File handle from winOpen
	int sizeRegion;         // Size of shared-memory regions
	int regionLength;       // Size of array apRegion
	struct ShmRegion {
		HANDLE hMap;        // File handle from CreateFileMapping
		void *map;
	} *regions;
	DWORD lastErrno;        // The Windows errno from the last I/O error
	int refs;               // Number of winShm objects pointing to this
	winShm *first;          // All winShm objects pointing to this
	winShmNode *next;       // Next in list of all winShmNode objects
#if defined(_DEBUG) || defined(LIBCU_HAVE_OS_TRACE)
	uint8_t nextShmId;      // Next available winShm.id value
#endif
};

/*
** A global array of all winShmNode objects.
**
** The winShmMutexHeld() must be true while reading or writing this list.
*/
static winShmNode *winShmNodeList = nullptr;

/*
** Structure used internally by this VFS to record the state of an open shared memory connection.
**
** The following fields are initialized when this object is created and are read-only thereafter:
**    winShm.shmNode
**    winShm.id
**
** All other fields are read/write.  The winShm.shmNode->mutex must be held while accessing any read/write fields.
*/
struct winShm {
	winShmNode *shmNode;		// The underlying winShmNode object
	winShm *next;				// Next winShm with the same winShmNode
	uint8_t hasMutex;			// True if holding the winShmNode mutex
	uint16_t sharedMask;		// Mask of shared locks held
	uint16_t exclMask;			// Mask of exclusive locks held
#if defined(_DEBUG) || defined(LIBCU_HAVE_OS_TRACE)
	uint8_t id;					// Id of this connection with its winShmNode
#endif
};

/* Constants used for locking */
#define WIN_SHM_BASE   ((22+LIBCU_SHM_NLOCK)*4)        // first lock byte
#define WIN_SHM_DMS    (WIN_SHM_BASE+LIBCU_SHM_NLOCK)  // deadman switch

/* Apply advisory locks for all n bytes beginning at offset. */
#define WINSHM_UNLCK  1
#define WINSHM_RDLCK  2
#define WINSHM_WRLCK  3
static int winShmSystemLock(winShmNode *file, int lockType, int offset, int bytes)
{
	RC rc = 0; // Result code form Lock/UnlockFileEx()
	// Access to the winShmNode object is serialized by the caller
	assert(mutex_held(file->mutex) || !file->refs);
	OSTRACE(("SHM-LOCK file=%p, lock=%d, offset=%d, size=%d\n", file->file.h, lockType, offset, bytes));
	// Release/Acquire the system-level lock
	if (lockType == WINSHM_UNLCK)
		rc = winUnlockFile(&file->file.h, offset, 0, bytes, 0);
	else {
		// Initialize the locking parameters
		DWORD dwFlags = LOCKFILE_FAIL_IMMEDIATELY;
		if (lockType == WINSHM_WRLCK) dwFlags |= LOCKFILE_EXCLUSIVE_LOCK;
		rc = winLockFile(&file->file.h, dwFlags, offset, 0, bytes, 0);
	}
	if (rc) rc = RC_OK;
	else { file->lastErrno = osGetLastError(); rc = RC_BUSY; }
	OSTRACE(("SHM-LOCK file=%p, func=%s, errno=%lu, rc=%s\n", file->file.h, lockType == WINSHM_UNLCK ? "winUnlockFile" : "winLockFile", file->lastErrno, runtimeErrName(rc)));
	return rc;
}

/* Forward references to VFS methods */
static int winOpen(vsystem *, const char *, vsysfile *, int, int *);
static int winDelete(vsystem *, const char *, int);

/*
** Purge the winShmNodeList list of all entries with winShmNode.nRef==0.
**
** This is not a VFS shared-memory method; it is a utility function called by VFS shared-memory methods.
*/
static void winShmPurge(vsystem *system, int deleteFlag)
{
	assert(winShmMutexHeld());
	OSTRACE(("SHM-PURGE pid=%lu, deleteFlag=%d\n", osGetCurrentProcessId(), deleteFlag));
	winShmNode **pp = &winShmNodeList;
	winShmNode *p;
	while ((p = *pp))
		if (!p->refs) {
			if (p->mutex) mutex_free(p->mutex);
			for (int i = 0; i < p->regionLength; i++) {
				BOOL rc = osUnmapViewOfFile(p->regions[i].map);
				OSTRACE(("SHM-PURGE-UNMAP pid=%lu, region=%d, rc=%s\n", osGetCurrentProcessId(), i, rc ? "ok" : "failed"));
				UNUSED_VARIABLE_VALUE(rc);
				rc = osCloseHandle(p->aRegion[i].hMap);
				OSTRACE(("SHM-PURGE-CLOSE pid=%lu, region=%d, rc=%s\n", osGetCurrentProcessId(), i, rc ? "ok" : "failed"));
				UNUSED_VARIABLE_VALUE(rc);
			}
			if (p->file.h && p->file.h != INVALID_HANDLE_VALUE) {
				SimulateIOErrorBenign(1);
				winClose((vsysfile *)&p->file);
				SimulateIOErrorBenign(0);
			}
			if (deleteFlag) {
				SimulateIOErrorBenign(1);
				allocBenignBegin();
				winDelete(system, p->filename, 0);
				allocBenignEnd();
				SimulateIOErrorBenign(0);
			}
			*pp = p->next;
			mfree(p->regions);
			mfree(p);
		}
		else pp = &p->next;
}

/*
** Open the shared-memory area associated with database file file.
**
** When opening a new shared-memory file, if no other instances of that file are currently open, in this process or in other processes, then
** the file must be truncated to zero length or have its header cleared.
*/
static int winOpenSharedMemory(winFile *file)
{
	assert(!file->shm); // Not previously opened

	// Allocate space for the new sqlite3_shm object.  Also speculatively allocate space for a new winShmNode and filename.
	struct winShm *p = allocZero(sizeof(*p)); // The connection to be opened
	if (!p) return RC_IOERR_NOMEM_BKPT;
	int nameLength = strlen(file->path); // Size of zName in bytes
	struct winShmNode *shmNode; // The underlying mmapped file
	struct winShmNode *newNode = allocZero(sizeof(*shmNode)+nameLength+17); //Newly allocated winShmNode
	if (!newNode) {
		mfree(p);
		return RC_IOERR_NOMEM_BKPT;
	}
	newNode->filename = (char*)&newNode[1];
	snprintf(newNode->filename, nameSize+15, "%s-shm", file->path);
	sqlite3FileSuffix3(file->fath, newNode->filename);

	// Look to see if there is an existing winShmNode that can be used. If no matching winShmNode currently exists, create a new one.
	winShmEnterMutex();
	for (shmNode = _winShmNodeList; shmNode; shmNode = shmNode->next)
		// TBD need to come up with better match here.  Perhaps use FILE_ID_BOTH_DIR_INFO Structure.
			if (!stricmp(shmNode->filename, newNode->filename)) break;
	RC rc;
	if (shmNode) 
		mfree(newNode);
	else {
		shmNode = newNode;
		newNode = nullptr;
		((winFile *)(&shmNode->file))->h = INVALID_HANDLE_VALUE;
		shmNode->next = winShmNodeList;
		_winShmNodeList = shmNode;

		if (_runtimeConfig.coreMutex) {
			shmNode->mutex = mutex_alloc(MUTEX_FAST);
			if (!shmNode->mutex) {
				rc = RC_IOERR_NOMEM_BKPT;
				goto shm_open_err;
			}
		}

		rc = winOpen(file->vsystem, shmNode->filename, (vsysfile *)&shmNode->file, VSYSTEM_OPEN_WAL|VSYSTEM_OPEN_READWRITE|VSYSTEM_OPEN_CREATE, nullptr);
		if (rc != RC_OK)
			goto shm_open_err;

		// Check to see if another process is holding the dead-man switch. If not, truncate the file to zero length.
		if (winShmSystemLock(shmNode, WINSHM_WRLCK, WIN_SHM_DMS, 1) == RC_OK) {
			rc = winTruncate((vsysfile *)&shmNode->file, 0);
			if (rc != RC_OK)
				rc = winLogError(RC_IOERR_SHMOPEN, osGetLastError(), "winOpenShm", file->path);
		}
		if (rc == RC_OK) {
			winShmSystemLock(shmNode, WINSHM_UNLCK, WIN_SHM_DMS, 1);
			rc = winShmSystemLock(shmNode, WINSHM_RDLCK, WIN_SHM_DMS, 1);
		}
		if (rc) goto shm_open_err;
	}

	// Make the new connection a child of the winShmNode
	p->shmNode = shmNode;
#if defined(_DEBUG) || defined(LIBCU_HAVE_OS_TRACE)
	p->id = shmNode->nextShmId++;
#endif
	shmNode->refs++;
	file->shm = p;
	winShmLeaveMutex();

	/* The reference count on shmNode has already been incremented under the cover of the winShmEnterMutex() mutex and the pointer from the
	** new (struct winShm) object to the shmNode has been set. All that is left to do is to link the new object into the linked list starting
	** at shmNode->pFirst. This must be done while holding the shmNode->mutex mutex.
	*/
	mutex_enter(shmNode->mutex);
	p->next = shmNode->first;
	shmNode->first = p;
	mutex_leave(shmNode->mutex);
	return RC_OK;

	// Jump here on any error
shm_open_err:
	winShmSystemLock(shmNode, WINSHM_UNLCK, WIN_SHM_DMS, 1);
	winShmPurge(file->system, 0); // This call frees shmNode if required
	mfree(p);
	mfree(newNode);
	winShmLeaveMutex();
	return rc;
}

/* Close a connection to shared-memory.  Delete the underlying storage if deleteFlag is true. */
static RC winShmUnmap(vsysfile *fd, bool deleteFlag)
{
	winFile *file = (winFile *)fd; // Database holding shared-memory
	winShm *p = file->shm; // The connection to be closed
	if (!p) return RC_OK;
	winShmNode *shmNode = p->shmNode; // The underlying shared-memory file

	// Remove connection p from the set of connections associated with shmNode
	mutex_enter(shmNode->mutex);
	winShm **pp; // For looping over sibling connections
	for (pp = &shmNode->first; (*pp) != p; pp = &(*pp)->next) { }
	*pp = p->next;

	// Free the connection p
	mfree(p);
	file->shm = nullptr;
	mutex_leave(shmNode->mutex);

	// If shmNode->nRef has reached 0, then close the underlying shared-memory file, too
	winShmEnterMutex();
	assert(shmNode->refs > 0);
	shmNode->refs--;
	if (!shmNode->refs)
		winShmPurge(file->system, deleteFlag);
	winShmLeaveMutex();
	return RC_OK;
}

/* Change the lock state for a shared-memory segment. */
static RC winShmLock(vsysfile *fd, int offset, int count, int flags)
{
	winFile *file = (winFile *)fd; // Connection holding shared memory
	winShm *p = file->shm; // The shared memory being locked
	winShmNode *shmNode = p->shmNode;
	int RC = RC_OK;

	assert(offset >= 0 && offset+count <= LIBCU_SHM_NLOCK);
	assert(count >= 1);
	assert(flags == (LIBCU_SHM_LOCK|LIBCU_SHM_SHARED) || flags == (LIBCU_SHM_LOCK|LIBCU_SHM_EXCLUSIVE)
		|| flags == (LIBCU_SHM_UNLOCK|LIBCU_SHM_SHARED) || flags == (LIBCU_SHM_UNLOCK|LIBCU_SHM_EXCLUSIVE));
	assert(count == 1 || (flags & LIBCU_SHM_EXCLUSIVE) != 0);

	uint16_t mask = (uint16_t)((1U<<(offset+count)) - (1U<<offset)); // Mask of locks to take or release
	assert(count > 1 || mask == (1<<offset));
	mutex_enter(shmNode->mutex);
	winShm *x; // For looping over all siblings
	if (flags & LIBCU_SHM_UNLOCK) {
		uint16_t allMask = 0; // Mask of locks held by siblings

		// See if any siblings hold this same lock
		for (x = shmNode->first; x; x = x->next) {
			if (x == p) continue;
			assert((x->exclMask & (p->exclMask|p->sharedMask)) == 0);
			allMask |= x->sharedMask;
		}

		// Unlock the system-level locks
		if ((mask & allMask) == 0) rc = winShmSystemLock(shmNode, WINSHM_UNLCK, offset+WIN_SHM_BASE, count);
		else rc = RC_OK;

		// Undo the local locks
		if (rc == RC_OK) {
			p->exclMask &= ~mask;
			p->sharedMask &= ~mask;
		}
	}
	else if (flags & LIBCU_SHM_SHARED) {
		uint16_t allShared = 0;  // Union of locks held by connections other than "p"

		// Find out which shared locks are already held by sibling connections. If any sibling already holds an exclusive lock, go ahead and return LIBCU_BUSY.
		for (x = shmNode->first; x; x = x->next) {
			if ((x->exclMask & mask) != 0) {
				rc = RC_BUSY;
				break;
			}
			allShared |= x->sharedMask;
		}

		// Get shared locks at the system level, if necessary
		if (rc == RC_OK) {
			if ((allShared & mask) == 0) rc = winShmSystemLock(shmNode, WINSHM_RDLCK, offset+WIN_SHM_BASE, count);
			else rc = RC_OK;
		}

		// Get the local shared locks
		if (rc == RC_OK)
			p->sharedMask |= mask;
	}
	else {
		// Make sure no sibling connections hold locks that will block this lock.  If any do, return LIBCU_BUSY right away.
		for (x = shmNode->first; x; x = x->next) {
			if ((x->exclMask & mask) != 0 || (x->sharedMask & mask) != 0) {
				rc = RC_BUSY;
				break;
			}
		}

		// Get the exclusive locks at the system level.  Then if successful also mark the local connection as being locked.
		if (rc == RC_OK) {
			rc = winShmSystemLock(shmNode, WINSHM_WRLCK, offset+WIN_SHM_BASE, n);
			if (rc == RC_OK) {
				assert((p->sharedMask & mask) == 0);
				p->exclMask |= mask;
			}
		}
	}
	mutex_leave(shmNode->mutex);
	OSTRACE(("SHM-LOCK pid=%lu, id=%d, sharedMask=%03x, exclMask=%03x, rc=%s\n", osGetCurrentProcessId(), p->id, p->sharedMask, p->exclMask, runtimeErrName(rc)));
	return rc;
}

/*
** Implement a memory barrier or memory fence on shared memory.
**
** All loads and stores begun before the barrier must complete before any load or store begun after the barrier.
*/
static void winShmBarrier(vsysfile *fd)
{
	UNUSED_SYMBOL(fd);
	systemMemoryBarrier(); // compiler-defined memory barrier
	winShmEnterMutex(); // Also mutex, for redundancy
	winShmLeaveMutex();
}

/*
** This function is called to obtain a pointer to region of the shared-memory associated with the database file fd. Shared-memory regions
** are numbered starting from zero. Each shared-memory region is sizeRegion bytes in size.
**
** If an error occurs, an error code is returned and *pp is set to NULL.
**
** Otherwise, if the isWrite parameter is 0 and the requested shared-memory region has not been allocated (by any client, including one running in a
** separate process), then *pp is set to NULL and RC_OK returned. If isWrite is non-zero and the requested shared-memory region has not yet
** been allocated, it is allocated by this function.
**
** If the shared-memory region has already been allocated or is allocated by this call as described above, then it is mapped into this processes
** address space (if it is not already), *pp is set to point to the mapped memory and RC_OK returned.
*/
static int winShmMap(vsysfile *fd, int region, int sizeRegion, bool isWrite, void volatile **pp)
{
	winFile *file = (winFile *)fd;
	winShm *shm = file->shm;
	RC rc = RC_OK;
	if (!shm) {
		rc = winOpenSharedMemory(file);
		if (rc != RC_OK) return rc;
		shm = file->shm;
	}
	winShmNode *shmNode = shm->shmNode;

	mutex_enter(shmNode->mutex);
	assert(sizeRegion == shmNode->sizeRegion || !shmNode->regionLength);

	if (shmNode->regionLength <= region) {
		newRegion; // New aRegion[] array
		int bytes = (region+1)*sizeRegion; // Minimum required file size
		shmNode->sizeRegion = sizeRegion;

		// The requested region is not mapped into this processes address space. Check to see if it has been allocated (i.e. if the wal-index file is large enough to contain the requested region).
		int64_t size; // Current size of wal-index file
		rc = winFileSize((vsysfile *)&shmNode->file, &size);
		if (rc != RC_OK) {
			rc = winLogError(RC_IOERR_SHMSIZE, osGetLastError(), "winShmMap1", file->path);
			goto shmpage_out;
		}

		if (size < bytes) {
			// The requested memory region does not exist. If isWrite is set to zero, exit early. *pp will be set to NULL and RC_OK returned.
			//
			// Alternatively, if isWrite is non-zero, use ftruncate() to allocate the requested memory region.
			if (!isWrite) goto shmpage_out;
			rc = winTruncate((vsysfile *)&shmNode->file, bytes);
			if (rc != RC_OK) {
				rc = winLogError(LIBCU_IOERR_SHMSIZE, osGetLastError(), "winShmMap2", file->path);
				goto shmpage_out;
			}
		}

		// Map the requested memory region into this processes address space.
		struct ShmRegion *newRegions = (struct ShmRegion *)realloc64(shmNode->regions, (region+1)*sizeof(newRegions[0]));
		if (!newRegions) {
			rc = RC_IOERR_NOMEM_BKPT;
			goto shmpage_out;
		}
		shmNode->regions = newRegions;

		while (shmNode->regionLength <= region) {
			HANDLE hMap = NULL; // file-mapping handle
#if LIBCU_OS_WINRT
			hMap = osCreateFileMappingFromApp(shmNode->file.h, NULL, PAGE_READWRITE, bytes, NULL);
#elif defined(LIBCU_WIN32_HAS_WIDE)
			hMap = osCreateFileMappingW(shmNode->file.h, NULL, PAGE_READWRITE, 0, bytes, NULL);
#elif defined(LIBCU_WIN32_HAS_ANSI) && LIBCU_WIN32_CREATEFILEMAPPINGA
			hMap = osCreateFileMappingA(shmNode->file.h, NULL, PAGE_READWRITE, 0, bytes, NULL);
#endif
			OSTRACE(("SHM-MAP-CREATE pid=%lu, region=%d, size=%d, rc=%s\n", osGetCurrentProcessId(), shmNode->regionLength, bytes, hMap ? "ok" : "failed"));
			void *map = nullptr; // Mapped memory region
			if (hMap) {
				int offset = shmNode->regionLength*regionSizeregionSize;
				int offsetShift = offset % winSysInfo.dwAllocationGranularity;
#if LIBCU_OS_WINRT
				map = osMapViewOfFileFromApp(hMap, FILE_MAP_WRITE|FILE_MAP_READ, offset - offsetShift, sizeRegion + offsetShift);
#else
				map = osMapViewOfFile(hMap, FILE_MAP_WRITE|FILE_MAP_READ, 0, offset - offsetShift, sizeRegion + offsetShift);
#endif
				OSTRACE(("SHM-MAP-MAP pid=%lu, region=%d, offset=%d, size=%d, rc=%s\n", osGetCurrentProcessId(), shmNode->regionLength, offset, sizeRegion, map ? "ok" : "failed"));
			}
			if (!map) {
				shmNode->lastErrno = osGetLastError();
				rc = winLogError(RC_IOERR_SHMMAP, shmNode->lastErrno, "winShmMap3", file->path);
				if (hMap) osCloseHandle(hMap);
				goto shmpage_out;
			}

			shmNode->regions[shmNode->regionLength].pMap = pMap;
			shmNode->regions[shmNode->regionLength].hMap = hMap;
			shmNode->regionLength++;
		}
	}

shmpage_out:
	if (shmNode->regionLength > region) {
		int offset = region*sizeRegion;
		int offsetShift = offset % winSysInfo.dwAllocationGranularity;
		char *p = (char *)shmNode->regions[region].map;
		*pp = (void *)&p[offsetShift];
	}
	else *pp = nullptr;
	mutex_leave(shmNode->mutex);
	return rc;
}

#else
#define winShmMap     RC_OK
#define winShmLock    RC_OK
#define winShmBarrier RC_OK
#define winShmUnmap   RC_OK
#endif /* #ifndef LIBCU_OMIT_WAL */
#pragma endregion

#pragma region Mapping

/* Cleans up the mapped region of the specified file, if any. */
#if LIBCU_MAXMMAPSIZE > 0
static int winUnmapfile(winFile *file)
{
	assert(file);
	OSTRACE(("UNMAP-FILE pid=%lu, pFile=%p, hMap=%p, pMapRegion=%p, mmapSize=%lld, mmapSizeActual=%lld, mmapSizeMax=%lld\n", osGetCurrentProcessId(), file, file->hMap, file->mapRegion, file->mmapSize, file->mmapSizeActual, file->mmapSizeMax));
	if (file->mapRegion) {
		if (!osUnmapViewOfFile(file->mapRegion)) {
			file->lastErrno = osGetLastError();
			OSTRACE(("UNMAP-FILE pid=%lu, pFile=%p, pMapRegion=%p, " "rc=LIBCU_IOERR_MMAP\n", osGetCurrentProcessId(), file, file->mapRegion));
			return winLogError(RC_IOERR_MMAP, file->lastErrno, "winUnmapfile1", file->path);
		}
		file->mapRegion = nullptr;
		file->mmapSize = 0;
		file->mmapSizeActual = 0;
	}
	if (file->hMap != NULL) {
		if (!osCloseHandle(file->hMap)) {
			file->lastErrno = osGetLastError();
			OSTRACE(("UNMAP-FILE pid=%lu, pFile=%p, hMap=%p, rc=LIBCU_IOERR_MMAP\n", osGetCurrentProcessId(), file, file->hMap));
			return winLogError(RC_IOERR_MMAP, file->lastErrno, "winUnmapfile2", file->path);
		}
		file->hMap = NULL;
	}
	OSTRACE(("UNMAP-FILE pid=%lu, pFile=%p, rc=RC_OK\n", osGetCurrentProcessId(), file));
	return RC_OK;
}

/*
** Memory map or remap the file opened by file-descriptor file (if the file is already mapped, the existing mapping is replaced by the new). Or, if
** there already exists a mapping for this file, and there are still outstanding xFetch() references to it, this function is a no-op.
**
** If parameter bytes is non-negative, then it is the requested size of the mapping to create. Otherwise, if bytes is less than zero, then the
** requested size is the size of the file on disk. The actual size of the created mapping is either the requested size or the value configured
** using LIBCU_FCNTL_MMAP_SIZE, whichever is smaller.
**
** RC_OK is returned if no error occurs (even if the mapping is not recreated as a result of outstanding references) or an Libcu error
** code otherwise.
*/
static int winMapfile(winFile *file, int64_t bytes)
{
	int64_t mapSize = bytes;
	assert(mapSize >= 0 || !file->fetchOuts);
	OSTRACE(("MAP-FILE pid=%lu, pFile=%p, size=%lld\n", osGetCurrentProcessId(), file, bytes));
	if (file->fetchOuts > 0) return RC_OK;

	RC rc;
	if (mapSize < 0) {
		rc = winFileSize((vsysfile *)file, &mapSize);
		if (rc) {
			OSTRACE(("MAP-FILE pid=%lu, pFile=%p, rc=LIBCU_IOERR_FSTAT\n", osGetCurrentProcessId(), file));
			return RC_IOERR_FSTAT;
		}
	}
	if (mapSize > file->mmapSizeMax)
		mapSize = file->mmapSizeMax;
	mapSize &= ~(int64_t)(winSysInfo.dwPageSize - 1);

	if (!mapSize && file->mmapSize > 0)
		winUnmapfile(file);
	if (mapSize != file->mmapSize) {
		DWORD protect = PAGE_READONLY;
		DWORD flags = FILE_MAP_READ;
		winUnmapfile(file);
#ifdef LIBCU_MMAP_READWRITE
		if ((file->ctrlFlags & WINFILE_RDONLY) == 0) {
			protect = PAGE_READWRITE;
			flags |= FILE_MAP_WRITE;
		}
#endif
#if LIBCU_OS_WINRT
		file->hMap = osCreateFileMappingFromApp(file->h, NULL, protect, mapSize, NULL);
#elif defined(LIBCU_WIN32_HAS_WIDE)
		file->hMap = osCreateFileMappingW(file->h, NULL, protect, (DWORD)((mapSize>>32) & 0xffffffff), (DWORD)(mapSize & 0xffffffff), NULL);
#elif defined(LIBCU_WIN32_HAS_ANSI) && LIBCU_WIN32_CREATEFILEMAPPINGA
		file->hMap = osCreateFileMappingA(file->h, NULL, protect, (DWORD)((mapSize>>32) & 0xffffffff), (DWORD)(mapSize & 0xffffffff), NULL);
#endif
		if (file->hMap == NULL) {
			file->lastErrno = osGetLastError();
			rc = winLogError(LIBCU_IOERR_MMAP, file->lastErrno, "winMapfile1", file->path);
			// Log the error, but continue normal operation using xRead/xWrite
			OSTRACE(("MAP-FILE-CREATE pid=%lu, pFile=%p, rc=%s\n", osGetCurrentProcessId(), file, runtimeErrName(rc)));
			return RC_OK;
		}
		assert((mapSize % winSysInfo.dwPageSize) == 0);
		assert(sizeof(SIZE_T) == sizeof(int64_t) || mapSize <= 0xffffffff);
		void *newMap;
#if LIBCU_OS_WINRT
		newMap = osMapViewOfFileFromApp(file->hMap, flags, 0, (SIZE_T)mapSize);
#else
		newMap = osMapViewOfFile(file->hMap, flags, 0, 0, (SIZE_T)mapSize);
#endif
		if (newMap == NULL) {
			osCloseHandle(file->hMap);
			file->hMap = NULL;
			file->lastErrno = osGetLastError();
			rc = winLogError(RC_IOERR_MMAP, file->lastErrno, "winMapfile2", file->path);
			// Log the error, but continue normal operation using xRead/xWrite
			OSTRACE(("MAP-FILE-MAP pid=%lu, pFile=%p, rc=%s\n", osGetCurrentProcessId(), file, runtimeErrName(rc)));
			return RC_OK;
		}
		file->mapRegion = newMap;
		file->mmapSize = mapSize;
		file->mmapSizeActual = mapSize;
	}

	OSTRACE(("MAP-FILE pid=%lu, pFile=%p, rc=RC_OK\n", osGetCurrentProcessId(), file));
	return RC_OK;
}
#endif /* LIBCU_MAXMMAPSIZE > 0 */

/*
** If possible, return a pointer to a mapping of file fd starting at offset. The mapping must be valid for at least amount bytes.
**
** If such a pointer can be obtained, store it in *pp and return RC_OK. Or, if one cannot but no error occurs, set *pp to 0 and return RC_OK.
** Finally, if an error does occur, return an Libcu error code. The final value of *pp is undefined in this case.
**
** If this function does return a pointer, the caller must eventually release the reference by calling winUnfetch().
*/
static int winFetch(vsysfile *fd, int64_t offset, int amount, void **pp)
{
	*pp = nullptr;
	OSTRACE(("FETCH pid=%lu, pFile=%p, offset=%lld, amount=%d, pp=%p\n", osGetCurrentProcessId(), fd, offset, amount, pp));
#if LIBCU_MAXMMAPSIZE > 0
	winFile *file = (winFile *)fd; // The underlying database file
	if (file->mmapSizeMax > 0) {
		if (!file->mapRegion) {
			int rc = winMapfile(file, -1);
			if (rc != RC_OK) {
				OSTRACE(("FETCH pid=%lu, pFile=%p, rc=%s\n", osGetCurrentProcessId(), file, runtimeErrName(rc)));
				return rc;
			}
		}
		if (file->mmapSize >= offset+amount) {
			*pp = &((uint8_t *)file->mapRegion)[offset];
			file->fetchOuts++;
		}
	}
#endif
	OSTRACE(("FETCH pid=%lu, pFile=%p, pp=%p, *pp=%p, rc=RC_OK\n", osGetCurrentProcessId(), fd, pp, *pp));
	return RC_OK;
}

/*
** If the third argument is non-NULL, then this function releases a reference obtained by an earlier call to winFetch(). The second
** argument passed to this function must be the same as the corresponding argument that was passed to the winFetch() invocation.
**
** Or, if the third argument is NULL, then this function is being called to inform the VFS layer that, according to POSIX, any existing mapping
** may now be invalid and should be unmapped.
*/
static int winUnfetch(vsysfile *fd, int64_t offset, void *p)
{
#if LIBCU_MAXMMAPSIZE > 0
	winFile *file = (winFile *)fd; // The underlying database file
	// If p==0 (unmap the entire file) then there must be no outstanding xFetch references. Or, if p!=0 (meaning it is an xFetch reference),
	// then there must be at least one outstanding.
	assert((!p) == (!file->fetchOuts));
	// If p!=0, it must match the offset value.
	assert(!p || p == &((uint8_t *)file->pMapRegion)[offset] );
	OSTRACE(("UNFETCH pid=%lu, pFile=%p, offset=%lld, p=%p\n", osGetCurrentProcessId(), file, offset, p));
	if (p)  file->fetchOuts--;
	// FIXME:  If Windows truly always prevents truncating or deleting a file while a mapping is held, then the following winUnmapfile() call
	// is unnecessary can be omitted - potentially improving performance.  */
	else winUnmapfile(file); 
	assert(file->fetchOuts >= 0);
#endif
	OSTRACE(("UNFETCH pid=%lu, pFile=%p, rc=RC_OK\n", osGetCurrentProcessId(), fd));
	return RC_OK;
}

#pragma endregion

/*
** Here ends the implementation of all vsysfile methods.
**
********************** End vsysfile Methods *******************************
******************************************************************************/

/* This vector defines all the methods that can operate on an vsysfile for win32. */
static const vsysfile_methods _winIoMethod = {
	3,                              // iVersion
	winClose,                       // xClose
	winRead,                        // xRead
	winWrite,                       // xWrite
	winTruncate,                    // xTruncate
	winSync,                        // xSync
	winFileSize,                    // xFileSize
	winLock,                        // xLock
	winUnlock,                      // xUnlock
	winCheckReservedLock,           // xCheckReservedLock
	winFileControl,                 // xFileControl
	winSectorSize,                  // xSectorSize
	winDeviceCharacteristics,       // xDeviceCharacteristics
	winShmMap,                      // xShmMap
	winShmLock,                     // xShmLock
	winShmBarrier,                  // xShmBarrier
	winShmUnmap,                    // xShmUnmap
	winFetch,                       // xFetch
	winUnfetch                      // xUnfetch
};

/* This vector defines all the methods that can operate on an vsysfile for win32 without performing any locking. */
static const vsysfile_methods _winIoNolockMethod = {
	3,                              // iVersion
	winClose,                       // xClose
	winRead,                        // xRead
	winWrite,                       // xWrite
	winTruncate,                    // xTruncate
	winSync,                        // xSync
	winFileSize,                    // xFileSize
	winNolockLock,                  // xLock
	winNolockUnlock,                // xUnlock
	winNolockCheckReservedLock,     // xCheckReservedLock
	winFileControl,                 // xFileControl
	winSectorSize,                  // xSectorSize
	winDeviceCharacteristics,       // xDeviceCharacteristics
	winShmMap,                      // xShmMap
	winShmLock,                     // xShmLock
	winShmBarrier,                  // xShmBarrier
	winShmUnmap,                    // xShmUnmap
	winFetch,                       // xFetch
	winUnfetch                      // xUnfetch
};

static winVfsAppData _winAppData = {
	&_winIoMethod,      // pMethod
	nullptr,            // pAppData
	false               // bNoLock
};

static winVfsAppData _winNolockAppData = {
	&_winIoNolockMethod, // pMethod
	nullptr,					// pAppData
	true                // bNoLock
};

#pragma endregion

#pragma region WinVSystem

/****************************************************************************
**************************** vsystem methods ****************************
**
** This division contains the implementation of methods on the vsystem object.
*/

#if defined(__CYGWIN__)
/*
** Convert a filename from whatever the underlying operating system supports for filenames into UTF-8.  Space to hold the result is
** obtained from malloc and must be freed by the calling function.
*/
static char *winConvertToUtf8Filename(const void *filename)
{
	char *converted = nullpt;
	if (osIsNT()) converted = winUnicodeToUtf8(filename);
#ifdef LIBCU_WIN32_HAS_ANSI
	else converted = winMbcsToUtf8(filename, osAreFileApisANSI());
#endif
	// caller will handle out of memory
	return converted;
}
#endif

/*
** Convert a UTF-8 filename into whatever form the underlying operating system wants filenames in.  Space to hold the result
** is obtained from malloc and must be freed by the calling function.
*/
static void *winConvertFromUtf8Filename(const char *filename)
{
	void *converted = nullptr;
	if (osIsNT()) converted = winUtf8ToUnicode(filename);
#ifdef LIBCU_WIN32_HAS_ANSI
	else converted = winUtf8ToMbcs(filename, osAreFileApisANSI());
#endif
	// caller will handle out of memory
	return converted;
}

/* This function returns non-zero if the specified UTF-8 string buffer ends with a directory separator character or one was successfully added to it. */
static bool winMakeEndInDirSep(int bufLength, char *buf)
{
	if (buf) {
		int length = strlen(buf);
		if (length > 0) {
			if (winIsDirSep(buf[length-1]))
				return true;
			else if (length+1 < bufLength) {
				buf[length] = winGetDirSep();
				buf[length+1] = '\0';
				return true;
			}
		}
	}
	return false;
}

/* Create a temporary file name and store the resulting pointer into pBuf. The pointer returned in pzBuf must be freed via mfree(). */
static int winGetTempname(vsystem *sys, char **pBuf)
{
	static char chars[] =
		"abcdefghijklmnopqrstuvwxyz"
		"ABCDEFGHIJKLMNOPQRSTUVWXYZ"
		"0123456789";

	// It's odd to simulate an io-error here, but really this is just using the io-error infrastructure to test that Libcu handles this function failing.
	SimulateIOError(return RC_IOERR);

	// Allocate a temporary buffer to store the fully qualified file name for the temporary file.  If this fails, we cannot continue.
	int max = sys->maxPathname; int bufLength = max + 2;
	char *buf = (char *)allocZero(bufLength);
	if (!buf) {
		OSTRACE(("TEMP-FILENAME rc=LIBCU_IOERR_NOMEM\n"));
		return RC_IOERR_NOMEM_BKPT;
	}

	// Figure out the effective temporary directory.  First, check if one has been explicitly set by the application; otherwise, use the one
	// configured by the operating system.
	int preLength = strlen(LIBCU_TEMP_FILE_PREFIX);
	int dirLength = max - (preLength+15);
	assert(dirLength > 0);

	if (_libcu_temp_directory) {
		int dirLength2 = strlen(_libcu_temp_directory);
		if (dirLength2 > 0) {
			if (!winIsDirSep(_libcu_temp_directory[dirLength2-1]))
				dirLength2++;
			if (dirLength2 > dirLength) {
				mfree(buf);
				OSTRACE(("TEMP-FILENAME rc=LIBCU_ERROR\n"));
				return winLogError(RC_ERROR, 0, "winGetTempname1", 0);
			}
			snprintf(buf, max, "%s", _libcu_temp_directory);
		}
	}
#if defined(__CYGWIN__)
	else {
		static const char *dirs[] = {
			nullptr, // getenv("LIBCU_TMPDIR")
			nullptr, // getenv("TMPDIR")
			nullptr, // getenv("TMP")
			nullptr, // getenv("TEMP")
			nullptr, // getenv("USERPROFILE")
			"/var/tmp",
			"/usr/tmp",
			"/tmp",
			".",
			nullptr // List terminator
		};
		const char *dir = nullptr;
		if (!dirs[0]) dirs[0] = getenv("LIBCU_TMPDIR");
		if (!dirs[1]) dirs[1] = getenv("TMPDIR");
		if (!dirs[2]) dirs[2] = getenv("TMP");
		if (!dirs[3]) dirs[3] = getenv("TEMP");
		if (!dirs[4]) dirs[4] = getenv("USERPROFILE");
		for (unsigned int i = 0; i < _LENGTHOF(dirs); dir = dirs[i++]) {
			void *converted;
			if (!dir) continue;
			// If the path starts with a drive letter followed by the colon character, assume it is already a native Win32 path; otherwise,
			// it must be converted to a native Win32 path via the Cygwin API prior to using it.
			if (winIsDriveLetterAndColon(dir)) {
				converted = winConvertFromUtf8Filename(zDir);
				if (!converted) {
					mfree(buf);
					OSTRACE(("TEMP-FILENAME rc=LIBCU_IOERR_NOMEM\n"));
					return RC_IOERR_NOMEM_BKPT;
				}
				if (winIsDir(converted)) {
					snprintf(buf, max, "%s", dir);
					mfree(converted);
					break;
				}
				mfree(converted);
			}
			else {
				converted = allocZero(max+1);
				if (!converted) {
					mfree(buf);
					OSTRACE(("TEMP-FILENAME rc=LIBCU_IOERR_NOMEM\n"));
					return RC_IOERR_NOMEM_BKPT;
				}
				if (cygwin_conv_path(osIsNT() ? CCP_POSIX_TO_WIN_W : CCP_POSIX_TO_WIN_A, dir, converted, max+1) < 0) {
					mfree(converted);
					mfree(buf);
					OSTRACE(("TEMP-FILENAME rc=LIBCU_IOERR_CONVPATH\n"));
					return winLogError(RC_IOERR_CONVPATH, (DWORD)errno, "winGetTempname2", dir);
				}
				if (winIsDir(converted)) {
					// At this point, we know the candidate directory exists and should be used.  However, we may need to convert the string containing
					// its name into UTF-8 (i.e. if it is UTF-16 right now).
					char *utf8 = winConvertToUtf8Filename(converted);
					if (!utf8) {
						mfree(converted);
						mfree(buf);
						OSTRACE(("TEMP-FILENAME rc=LIBCU_IOERR_NOMEM\n"));
						return RC_IOERR_NOMEM_BKPT;
					}
					snprintf(buf, max, "%s", utf8);
					mfree(utf8);
					mfree(converted);
					break;
				}
				mfree(zConverted);
			}
		}
	}
#elif !LIBCU_OS_WINRT && !defined(__CYGWIN__)
	else if (osIsNT()) {
		LPWSTR widePath = (LPWSTR)allocZero(max * sizeof(WCHAR));
		if (!widePath) {
			mfree(buf);
			OSTRACE(("TEMP-FILENAME rc=LIBCU_IOERR_NOMEM\n"));
			return RC_IOERR_NOMEM_BKPT;
		}
		if (!osGetTempPathW(max, widePath)) {
			mfree(widePath);
			mfree(buf);
			OSTRACE(("TEMP-FILENAME rc=LIBCU_IOERR_GETTEMPPATH\n"));
			return winLogError(RC_IOERR_GETTEMPPATH, osGetLastError(), "winGetTempname2", 0);
		}
		char *multi = winUnicodeToUtf8(widePath);
		if (multi) {
			snprintf(buf, max, "%s", multi);
			mfree(multi);
			mfree(widePath);
		}
		else {
			mfree(widePath);
			mfree(buf);
			OSTRACE(("TEMP-FILENAME rc=LIBCU_IOERR_NOMEM\n"));
			return RC_IOERR_NOMEM_BKPT;
		}
	}
#ifdef LIBCU_WIN32_HAS_ANSI
	else {
		char *mbcsPath = allocZero(max);
		if (!mbcsPath) {
			mfree(buf);
			OSTRACE(("TEMP-FILENAME rc=LIBCU_IOERR_NOMEM\n"));
			return RC_IOERR_NOMEM_BKPT;
		}
		if (!osGetTempPathA(max, mbcsPath)) {
			mfree(buf);
			OSTRACE(("TEMP-FILENAME rc=LIBCU_IOERR_GETTEMPPATH\n"));
			return winLogError(RC_IOERR_GETTEMPPATH, osGetLastError(), "winGetTempname3", 0);
		}
		char *utf8 = winMbcsToUtf8(mbcsPath, osAreFileApisANSI());
		if (utf8) {
			snprintf(buf, max, "%s", utf8);
			mfree(utf8);
		}
		else {
			mfree(buf);
			OSTRACE(("TEMP-FILENAME rc=LIBCU_IOERR_NOMEM\n"));
			return RC_IOERR_NOMEM_BKPT;
		}
	}
#endif /* LIBCU_WIN32_HAS_ANSI */
#endif /* !LIBCU_OS_WINRT */

	// Check to make sure the temporary directory ends with an appropriate separator.  If it does not and there is not enough space left to add one, fail.
	if (!winMakeEndInDirSep(dirLength+1, buf)) {
		mfree(buf);
		OSTRACE(("TEMP-FILENAME rc=LIBCU_ERROR\n"));
		return winLogError(RC_ERROR, 0, "winGetTempname4", 0);
	}

	// Check that the output buffer is large enough for the temporary file name in the following format:
	//   "<temporary_directory>/etilqs_XXXXXXXXXXXXXXX\0\0"
	//
	// If not, return LIBCU_ERROR.  The number 17 is used here in order to account for the space used by the 15 character random suffix and the
	// two trailing NUL characters.  The final directory separator character has already added if it was not already present.
	int length = strlen(buf);
	if ((length + preLength + 17) > bufLength) {
		mfree(buf);
		OSTRACE(("TEMP-FILENAME rc=LIBCU_ERROR\n"));
		return winLogError(RC_ERROR, 0, "winGetTempname5", 0);
	}
	snprintf(buf+length, bufLength-16-length, LIBCU_TEMP_FILE_PREFIX);

	size_t i, j = strlen(buf);
	sqlite3_randomness(15, &buf[j]);
	for (i = 0; i < 15; i++, j++)
		buf[j] = (char)chars[((unsigned char)buf[j])%(sizeof(chars)-1)];
	buf[j] = 0;
	buf[j+1] = 0;
	*pBuf = buf;
	OSTRACE(("TEMP-FILENAME name=%s, rc=RC_OK\n", buf));
	return RC_OK;
}

/* Return TRUE if the named file is really a directory.  Return false if it is something other than a directory, or if there is any kind of memory allocation failure. */
static bool winIsDir(const void *converted)
{
	DWORD attr;
	if (osIsNT()) {
		int cnt = 0;
		WIN32_FILE_ATTRIBUTE_DATA sAttrData;
		memset(&sAttrData, 0, sizeof(sAttrData));
		int rc = 0;
		DWORD lastErrno;
		while (!(rc = osGetFileAttributesExW((LPCWSTR)converted, GetFileExInfoStandard, &sAttrData)) && winRetryIoerr(&cnt, &lastErrno)) { }
		if (!rc)
			return false; // Invalid name?
		attr = sAttrData.dwFileAttributes;
	}
#if LIBCU_OS_WINCE == 0
	else attr = osGetFileAttributesA((char *)converted);
#endif
	return (attr != INVALID_FILE_ATTRIBUTES) && (attr & FILE_ATTRIBUTE_DIRECTORY);
}

/* Open a file. */
static RC winOpen(vsystem *sys, const char *name, vsysfile *id, int flags, int *outFlags)
{
	RC rc = RC_OK;
#if !defined(NDEBUG) || LIBCU_OS_WINCE
	int type = flags & 0xFFFFFF00; // Type of file to open
#endif
	bool isExclusive  = (flags & SYSTEM_OPEN_EXCLUSIVE);
	bool isDelete     = (flags & SYSTEM_OPEN_DELETEONCLOSE);
	bool isCreate     = (flags & SYSTEM_OPEN_CREATE);
	bool isReadonly   = (flags & SYSTEM_OPEN_READONLY);
	bool isReadWrite  = (flags & SYSTEM_OPEN_READWRITE);
#ifndef NDEBUG
	bool isOpenJournal = (isCreate && (type == SYSTEM_OPEN_MASTER_JOURNAL || type == SYSTEM_OPEN_MAIN_JOURNAL || type == SYSTEM_OPEN_WAL));
#endif
	const char *utf8Name = name; // Filename in UTF-8 encoding
	OSTRACE(("OPEN name=%s, pFile=%p, flags=%x, pOutFlags=%p\n", utf8Name, id, flags, outFlags));

	// Check the following statements are true:
	//
	//   (a) Exactly one of the READWRITE and READONLY flags must be set, and
	//   (b) if CREATE is set, then READWRITE must also be set, and
	//   (c) if EXCLUSIVE is set, then CREATE must also be set.
	//   (d) if DELETEONCLOSE is set, then CREATE must also be set.
	assert((!isReadonly || !isReadWrite) && (isReadWrite || isReadonly));
	assert(!isCreate || isReadWrite);
	assert(!isExclusive || isCreate);
	assert(!isDelete || isCreate);
	// The main DB, main journal, WAL file and master journal are never automatically deleted. Nor are they ever temporary files.
	assert((!isDelete && name) || type != SYSTEM_OPEN_MAIN_DB);
	assert((!isDelete && name) || type != SYSTEM_OPEN_MAIN_JOURNAL);
	assert((!isDelete && name) || type != SYSTEM_OPEN_MASTER_JOURNAL);
	assert((!isDelete && name) || type != SYSTEM_OPEN_WAL);
	// Assert that the upper layer has set one of the "file-type" flags.
	assert(type == SYSTEM_OPEN_MAIN_DB      || type == LIBCU_OPEN_TEMP_DB
		|| type == SYSTEM_OPEN_MAIN_JOURNAL || type == LIBCU_OPEN_TEMP_JOURNAL
		|| type == SYSTEM_OPEN_SUBJOURNAL   || type == LIBCU_OPEN_MASTER_JOURNAL
		|| type == SYSTEM_OPEN_TRANSIENT_DB || type == LIBCU_OPEN_WAL);
	winFile *file = (winFile *)id;
	assert(file);
	memset(file, 0, sizeof(winFile));
	file->h = INVALID_HANDLE_VALUE;
#if LIBCU_OS_WINRT
	if (!utf8Name && !_libcu_temp_directory)
		_log(RC_ERROR, "sqlite3_temp_directory variable should be set for WinRT");
#endif

	// If argument zPath is a NULL pointer, this function is required to open a temporary file. Use this buffer to store the file name in.
	char *tmpname = nullptr; // For temporary filename, if necessary.

	// If the second argument to this function is NULL, generate a temporary file name to use
	if (!utf8Name) {
		assert(isDelete && !isOpenJournal);
		rc = winGetTempname(sys, &tmpname);
		if (rc != RC_OK) {
			OSTRACE(("OPEN name=%s, rc=%s", utf8Name, runtimeErrName(rc)));
			return rc;
		}
		utf8Name = tmpname;
	}

	// Database filenames are double-zero terminated if they are not URIs with parameters.  Hence, they can always be passed into sqlite3_uri_parameter().
	assert((type != SYSTEM_OPEN_MAIN_DB) || (flags & SYSTEM_OPEN_URI) || !utf8Name[strlen(utf8Name)+1]);

	// Convert the filename to the system encoding.
	void *converted = winConvertFromUtf8Filename(utf8Name); // Filename in OS encoding
	if (!converted) {
		mfree(tmpname);
		OSTRACE(("OPEN name=%s, rc=LIBCU_IOERR_NOMEM", utf8Name));
		return RC_IOERR_NOMEM_BKPT;
	}
	if (winIsDir(converted)) {
		mfree(converted);
		mfree(tmpname);
		OSTRACE(("OPEN name=%s, rc=LIBCU_CANTOPEN_ISDIR", utf8Name));
		return RC_CANTOPEN_ISDIR;
	}

	DWORD dwDesiredAccess = isReadWrite ? GENERIC_READ|GENERIC_WRITE : GENERIC_READ;
	// LIBCU_OPEN_EXCLUSIVE is used to make sure that a new file is created. Libcu doesn't use it to indicate "exclusive access" as it is usually understood.
	DWORD dwCreationDisposition;
	if (isExclusive) dwCreationDisposition = CREATE_NEW; // Creates a new file, only if it does not already exist. If the file exists, it fails.
	else if (isCreate) dwCreationDisposition = OPEN_ALWAYS; // Open existing file, or create if it doesn't exist
	else dwCreationDisposition = OPEN_EXISTING; // Opens a file, only if it exists.
	DWORD dwShareMode = FILE_SHARE_READ | FILE_SHARE_WRITE;
	DWORD dwFlagsAndAttributes = 0;
#if LIBCU_OS_WINCE
	bool isTemp = false;
#endif
	if (isDelete) {
#if LIBCU_OS_WINCE
		dwFlagsAndAttributes = FILE_ATTRIBUTE_HIDDEN;
		isTemp = true;
#else
		dwFlagsAndAttributes = FILE_ATTRIBUTE_TEMPORARY|FILE_ATTRIBUTE_HIDDEN|FILE_FLAG_DELETE_ON_CLOSE;
#endif
	}
	else dwFlagsAndAttributes = FILE_ATTRIBUTE_NORMAL;
	// Reports from the internet are that performance is always better if FILE_FLAG_RANDOM_ACCESS is used.  Ticket #2699.
#if LIBCU_OS_WINCE
	dwFlagsAndAttributes |= FILE_FLAG_RANDOM_ACCESS;
#endif

	HANDLE h;
	int cnt = 0;
	DWORD lastErrno = 0;
	if (osIsNT()) {
#if LIBCU_OS_WINRT
		CREATEFILE2_EXTENDED_PARAMETERS extendedParameters;
		extendedParameters.dwSize = sizeof(CREATEFILE2_EXTENDED_PARAMETERS);
		extendedParameters.dwFileAttributes = dwFlagsAndAttributes & FILE_ATTRIBUTE_MASK;
		extendedParameters.dwFileFlags = dwFlagsAndAttributes & FILE_FLAG_MASK;
		extendedParameters.dwSecurityQosFlags = SECURITY_ANONYMOUS;
		extendedParameters.lpSecurityAttributes = NULL;
		extendedParameters.hTemplateFile = NULL;
		while ((h = osCreateFile2((LPCWSTR)converted, dwDesiredAccess, dwShareMode, dwCreationDisposition, &extendedParameters)) == INVALID_HANDLE_VALUE && winRetryIoerr(&cnt, &lastErrno)) { } // Noop
#else
		while ((h = osCreateFileW((LPCWSTR)converted, dwDesiredAccess, dwShareMode, NULL, dwCreationDisposition, dwFlagsAndAttributes, NULL)) == INVALID_HANDLE_VALUE && winRetryIoerr(&cnt, &lastErrno)) { } // Noop
#endif
	}
#ifdef LIBCU_WIN32_HAS_ANSI
	else while ((h = osCreateFileA((LPCSTR)converted, dwDesiredAccess, dwShareMode, NULL, dwCreationDisposition, dwFlagsAndAttributes, NULL)) == INVALID_HANDLE_VALUE && winRetryIoerr(&cnt, &lastErrno)) { } // Noop
#endif
	winLogIoerr(cnt, __LINE__);

	OSTRACE(("OPEN file=%p, name=%s, access=%lx, rc=%s\n", h, utf8Name, dwDesiredAccess, h == INVALID_HANDLE_VALUE ? "failed" : "ok"));

	if (h == INVALID_HANDLE_VALUE) {
		file->lastErrno = lastErrno;
		winLogError(RC_CANTOPEN, file->lastErrno, "winOpen", utf8Name);
		mfree(converted);
		mfree(tmpname);
		if (isReadWrite && !isExclusive) return winOpen(sys, name, id, ((flags|SYSTEM_OPEN_READONLY) & ~(SYSTEM_OPEN_CREATE|SYSTEM_OPEN_READWRITE)), outFlags);
		else return RC_CANTOPEN_BKPT;
	}

	if (outFlags) {
		if (isReadWrite) *outFlags = SYSTEM_OPEN_READWRITE;
		else *outFlags = SYSTEM_OPEN_READONLY;
	}

	OSTRACE(("OPEN file=%p, name=%s, access=%lx, pOutFlags=%p, *pOutFlags=%d, rc=%s\n", h, utf8Name, dwDesiredAccess, outFlags, outFlags ? *outFlags : 0, h == INVALID_HANDLE_VALUE ? "failed" : "ok"));

	winVfsAppData *appData = (winVfsAppData *)sys->appData;

#if LIBCU_OS_WINCE
	{
		if (isReadWrite && type == SYSTEM_OPEN_MAIN_DB && (appData == NULL || !appData->noLock) && (rc = winceCreateLock(name, file)) != RC_OK) {
			osCloseHandle(h);
			mfree(converted);
			mfree(tmpname);
			OSTRACE(("OPEN-CE-LOCK name=%s, rc=%s\n", name, runtimeErrName(rc)));
			return rc;
		}
	}
	if (isTemp) 
		file->deleteOnClose = converted;
	else
#endif
	{
		mfree(converted);
	}

	mfree(tmpname);
	file->method = appData ? appData->method : &_winIoMethod;
	file->sys = sys;
	file->h = h;
	if (isReadonly)
		file->ctrlFlags |= WINFILE_RDONLY;
	if (convert_uriboolean(name, "psow", LIBCU_POWERSAFE_OVERWRITE))
		file->ctrlFlags |= WINFILE_PSOW;
	file->lastErrno = NO_ERROR;
	file->path = name;
#if LIBCU_MAXMMAPSIZE>0
	file->hMap = NULL;
	file->mapRegion = 0;
	file->mmapSize = 0;
	file->mmapSizeActual = 0;
	file->mmapSizeMax = _runtimeConfig.sizeMmap;
#endif

	OpenCounter(+1);
	return rc;
}

/*
** Delete the named file.
**
** Note that Windows does not allow a file to be deleted if some other process has it open.  Sometimes a virus scanner or indexing program
** will open a journal file shortly after it is created in order to do whatever it does.  While this other process is holding the
** file open, we will be unable to delete it.  To work around this problem, we delay 100 milliseconds and try to delete again.  Up
** to MX_DELETION_ATTEMPTs deletion attempts are run before giving up and returning an error.
*/
static RC winDelete(vsystem *sys, const char *filename, int syncDir)
{
	UNUSED_SYMBOL(sys);
	UNUSED_SYMBOL(syncDir);
	SimulateIOError(return RC_IOERR_DELETE);
	OSTRACE(("DELETE name=%s, syncDir=%d\n", filename, syncDir));

	void *converted = winConvertFromUtf8Filename(filename);
	if (!converted) {
		OSTRACE(("DELETE name=%s, rc=LIBCU_IOERR_NOMEM\n", filename));
		return RC_IOERR_NOMEM_BKPT;
	}

	RC rc;
	DWORD attr;
	DWORD lastErrno = 0;
	int cnt = 0;
	if (osIsNT()) {
		do {
#if LIBCU_OS_WINRT
			WIN32_FILE_ATTRIBUTE_DATA sAttrData;
			memset(&sAttrData, 0, sizeof(sAttrData));
			if (osGetFileAttributesExW(converted, GetFileExInfoStandard, &sAttrData))
				attr = sAttrData.dwFileAttributes;
			else {
				lastErrno = osGetLastError();
				if (lastErrno == ERROR_FILE_NOT_FOUND || lastErrno == ERROR_PATH_NOT_FOUND) rc = RC_IOERR_DELETE_NOENT; // Already gone?
				else rc = RC_ERROR;
				break;
			}
#else
			attr = osGetFileAttributesW(converted);
#endif
			if (attr == INVALID_FILE_ATTRIBUTES) {
				lastErrno = osGetLastError();
				if (lastErrno == ERROR_FILE_NOT_FOUND || lastErrno == ERROR_PATH_NOT_FOUND) rc = RC_IOERR_DELETE_NOENT; // Already gone?
				else rc = RC_ERROR;
				break;
			}
			if (attr&FILE_ATTRIBUTE_DIRECTORY) {
				rc = RC_ERROR; // Files only.
				break;
			}
			if (osDeleteFileW(converted)) {
				rc = RC_OK; // Deleted OK.
				break;
			}
			if (!winRetryIoerr(&cnt, &lastErrno)) {
				rc = RC_ERROR; // No more retries.
				break;
			}
		} while(1);
	}
#ifdef LIBCU_WIN32_HAS_ANSI
	else {
		do {
			attr = osGetFileAttributesA(converted);
			if (attr == INVALID_FILE_ATTRIBUTES) {
				lastErrno = osGetLastError();
				if (lastErrno==ERROR_FILE_NOT_FOUND || lastErrno==ERROR_PATH_NOT_FOUND) rc = RC_IOERR_DELETE_NOENT; // Already gone?
				else rc = RC_ERROR;
				break;
			}
			if (attr&FILE_ATTRIBUTE_DIRECTORY) {
				rc = RC_ERROR; // Files only.
				break;
			}
			if (osDeleteFileA(converted)) {
				rc = RC_OK; // Deleted OK.
				break;
			}
			if (!winRetryIoerr(&cnt, &lastErrno)) {
				rc = RC_ERROR; // No more retries.
				break;
			}
		} while(1);
	}
#endif
	if (rc && rc != RC_IOERR_DELETE_NOENT)
		rc = winLogError(RC_IOERR_DELETE, lastErrno, "winDelete", filename);
	else
		winLogIoerr(cnt, __LINE__);
	mfree(converted);
	OSTRACE(("DELETE name=%s, rc=%s\n", filename, runtimeErrName(rc)));
	return rc;
}

/* Check the existence and status of a file. */
static RC winAccess(vsystem *sys, const char *filename, int flags, int *resOut)
{
	UNUSED_SYMBOL(sys);
	SimulateIOError(return RC_IOERR_ACCESS;);
	OSTRACE(("ACCESS name=%s, flags=%x, pResOut=%p\n", filename, flags, resOut));

	void *converted = winConvertFromUtf8Filename(filename);
	if (!converted) {
		OSTRACE(("ACCESS name=%s, rc=LIBCU_IOERR_NOMEM\n", filename));
		return RC_IOERR_NOMEM_BKPT;
	}
	DWORD attr;
	RC rc = 0;
	DWORD lastErrno = 0;
	if (osIsNT()) {
		int cnt = 0;
		WIN32_FILE_ATTRIBUTE_DATA sAttrData;
		memset(&sAttrData, 0, sizeof(sAttrData));
		while (!(rc = osGetFileAttributesExW((LPCWSTR)converted, GetFileExInfoStandard, &sAttrData)) && winRetryIoerr(&cnt, &lastErrno)) { }
		if (rc) {
			// For an SYSTEM_ACCESS_EXISTS query, treat a zero-length file as if it does not exist.
			if (flags == SYSTEM_ACCESS_EXISTS && !sAttrData.nFileSizeHigh && !sAttrData.nFileSizeLow) attr = INVALID_FILE_ATTRIBUTES;
			else attr = sAttrData.dwFileAttributes;
		}
		else {
			winLogIoerr(cnt, __LINE__);
			if (lastErrno != ERROR_FILE_NOT_FOUND && lastErrno != ERROR_PATH_NOT_FOUND) {
				mfree(converted);
				return winLogError(RC_IOERR_ACCESS, lastErrno, "winAccess", filename);
			}
			else attr = INVALID_FILE_ATTRIBUTES;
		}
	}
#ifdef LIBCU_WIN32_HAS_ANSI
	else attr = osGetFileAttributesA((char *)converted);
#endif
	mfree(converted);
	switch (flags) {
	case SYSTEM_ACCESS_READ:
	case SYSTEM_ACCESS_EXISTS:
		rc = attr != INVALID_FILE_ATTRIBUTES;
		break;
	case SYSTEM_ACCESS_READWRITE:
		rc = attr != INVALID_FILE_ATTRIBUTES && (attr & FILE_ATTRIBUTE_READONLY) == 0;
		break;
	default:
		assert(!"Invalid flags argument");
	}
	*resOut = rc;
	OSTRACE(("ACCESS name=%s, pResOut=%p, *pResOut=%d, rc=RC_OK\n", filename, resOut, *resOut));
	return RC_OK;
}

/* Returns non-zero if the specified path name starts with a drive letter followed by a colon character. */
static BOOL winIsDriveLetterAndColon(const char *pathname)
{
	return (isalpha(pathname[0]) && pathname[1]==':');
}

/*
** Returns non-zero if the specified path name should be used verbatim.  If non-zero is returned from this function, the calling function must simply
** use the provided path name verbatim -OR- resolve it into a full path name using the GetFullPathName Win32 API function (if available).
*/
static BOOL winIsVerbatimPathname(const char *pathname)
{
	// If the path name starts with a forward slash or a backslash, it is either a legal UNC name, a volume relative path, or an absolute path name in the
	// "Unix" format on Windows.  There is no easy way to differentiate between the final two cases; therefore, we return the safer return value of TRUE
	// so that callers of this function will simply use it verbatim.
	if (winIsDirSep(pathname[0]))
		return TRUE;
	// If the path name starts with a letter and a colon it is either a volume relative path or an absolute path.  Callers of this function must not
	// attempt to treat it as a relative path name (i.e. they should simply use it verbatim).
	if (winIsDriveLetterAndColon(pathname))
		return TRUE;
	// If we get to this point, the path name should almost certainly be a purely relative one (i.e. not a UNC name, not absolute, and not volume relative).
	return FALSE;
}

/* Turn a relative pathname into a full pathname.  Write the full pathname into zOut[].  zOut[] will be at least sys->mxPathname bytes in size. */
static RC winFullPathname(vsystem *sys, const char *relative, int fullLength, char *full)
{
#if !LIBCU_OS_WINCE && !LIBCU_OS_WINRT && !defined(__CYGWIN__)
	DWORD bytes;
	void *converted;
	char *out;
#endif

	// If this path name begins with "/X:", where "X" is any alphabetic character, discard the initial "/" from the pathname.
	if (relative[0] == '/' && winIsDriveLetterAndColon(relative+1))
		relative++;

#if defined(__CYGWIN__)
	SimulateIOError(return RC_ERROR);
	UNUSED_SYMBOL(fullLength);
	assert(fullLength >= sys->maxPathname);
	if (_libcu_data_directory && !winIsVerbatimPathname(relative)) {
		/*
		** NOTE: We are dealing with a relative path name and the data directory has been set.  Therefore, use it as the basis
		**       for converting the relative path name to an absolute one by prepending the data directory and a slash.
		*/
		char *out = sqlite3MallocZero( sys->mxPathname+1 );
		if (!out)
			return RC_IOERR_NOMEM_BKPT;
		if (cygwin_conv_path((osIsNT() ? CCP_POSIX_TO_WIN_W : CCP_POSIX_TO_WIN_A)|CCP_RELATIVE, relative, out, sys->maxPathname+1) < 0) {
			mfree(out);
			return winLogError(RC_CANTOPEN_CONVPATH, (DWORD)errno, "winFullPathname1", relative);
		}
		else {
			char *utf8 = winConvertToUtf8Filename(out);
			if (!utf8) {
				mfree(out);
				return RC_IOERR_NOMEM_BKPT;
			}
			snprintf(full, MIN(fullLength, sys->maxPathname), full, "%s%c%s", _libcu_data_directory, winGetDirSep(), utf8);
			mfree(utf8);
			mfree(out);
		}
	}
	else {
		char *out = allocZero(sys->maxPathname+1);
		if (!out)
			return RC_IOERR_NOMEM_BKPT;
		if (cygwin_conv_path((osIsNT() ? CCP_POSIX_TO_WIN_W : CCP_POSIX_TO_WIN_A), relative, out, sys->maxPathname+1) < 0) {
			mfree(out);
			return winLogError(RC_CANTOPEN_CONVPATH, (DWORD)errno, "winFullPathname2", relative);
		}
		else {
			char *utf8 = winConvertToUtf8Filename(out);
			if (!utf8) {
				mfree(out);
				return RC_IOERR_NOMEM_BKPT;
			}
			snprintf(full, MIN(fullLength, sys->maxPathname), "%s", utf8);
			mfree(utf8);
			mfree(out);
		}
	}
	return RC_OK;
#endif

#if (LIBCU_OS_WINCE || LIBCU_OS_WINRT) && !defined(__CYGWIN__)
	SimulateIOError(return RC_ERROR);
	// WinCE has no concept of a relative pathname, or so I am told.
	// WinRT has no way to convert a relative path to an absolute one.
	/*
	** NOTE: We are dealing with a relative path name and the data directory has been set.  Therefore, use it as the basis
	**       for converting the relative path name to an absolute one by prepending the data directory and a backslash.
	*/
	if (_libcu_data_directory && !winIsVerbatimPathname(relative))snprintf(full, _MIN(fullLength, sys->maxPathname), full, "%s%c%s", _libcu_data_directory, winGetDirSep(), relative);
	else snprintf(full, _MIN(fullLenth, sys->maxPathname), full, "%s", relative);
	return RC_OK;
#endif

#if !LIBCU_OS_WINCE && !LIBCU_OS_WINRT && !defined(__CYGWIN__)
	// It's odd to simulate an io-error here, but really this is just using the io-error infrastructure to test that Libcu handles this
	// function failing. This function could fail if, for example, the current working directory has been unlinked.
	SimulateIOError(return RC_ERROR);
	if (_libcu_data_directory && !winIsVerbatimPathname(relative)) {
		/*
		** NOTE: We are dealing with a relative path name and the data directory has been set.  Therefore, use it as the basis
		**       for converting the relative path name to an absolute one by prepending the data directory and a backslash.
		*/
		snprintf(full, _MIN(fullLength, sys->maxPathname), full, "%s%c%s", _libcu_data_directory, winGetDirSep(), relative);
		return RC_OK;
	}
	converted = winConvertFromUtf8Filename(relative);
	if (!converted)
		return RC_IOERR_NOMEM_BKPT;
	if (osIsNT()) {
		LPWSTR temp;
		bytes = osGetFullPathNameW((LPCWSTR)converted, 0, 0, 0);
		if (!bytes) {
			mfree(converted);
			return winLogError(RC_CANTOPEN_FULLPATH, osGetLastError(), "winFullPathname1", relative);
		}
		bytes += 3;
		temp = allocZero(bytes*sizeof(temp[0]));
		if (!temp){
			mfree(converted);
			return RC_IOERR_NOMEM_BKPT;
		}
		bytes = osGetFullPathNameW((LPCWSTR)converted, bytes, temp, 0);
		if (!bytes) {
			mfree(converted);
			mfree(temp);
			return winLogError(RC_CANTOPEN_FULLPATH, osGetLastError(), "winFullPathname2", relative);
		}
		mfree(converted);
		out = winUnicodeToUtf8(temp);
		mfree(temp);
	}
#ifdef LIBCU_WIN32_HAS_ANSI
	else {
		char *temp;
		bytes = osGetFullPathNameA((char*)converted, 0, 0, 0);
		if (!bytes) {
			mfree(converted);
			return winLogError(LIBCU_CANTOPEN_FULLPATH, osGetLastError(), "winFullPathname3", relative);
		}
		bytes += 3;
		temp = allocZero(bytes*sizeof(temp[0]));
		if (!temp) {
			mfree(converted);
			return RC_IOERR_NOMEM_BKPT;
		}
		bytes = osGetFullPathNameA((char *)converted, bytes, temp, 0);
		if (!bytes) {
			mfree(converted);
			mfree(temp);
			return winLogError(RC_CANTOPEN_FULLPATH, osGetLastError(), "winFullPathname4", relative);
		}
		mfree(converted);
		out = winMbcsToUtf8(temp, osAreFileApisANSI());
		mfree(temp);
	}
#endif
	if (out) {
		snprintf(full, _MIN(fullLength, sys->maxPathname), full, "%s", out);
		mfree(out);
		return RC_OK;
	}
	else return RC_IOERR_NOMEM_BKPT;
#endif
}

#ifndef LIBCU_OMIT_LOAD_EXTENSION
/* Interfaces for opening a shared library, finding entry points within the shared library, and closing the shared library. */
static void *winDlOpen(vsystem *sys, const char *filename)
{
#if defined(__CYGWIN__)
	int fullLength = sys->mxPathname+1;
	char *full = allocZero(fullLength);
	void *converted = nullptr;
	if (!full) {
		OSTRACE(("DLOPEN name=%s, handle=%p\n", filename, (void *)nullptr));
		return nullptr;
	}
	if (winFullPathname(sys, filename, fullLength, full) != RC_OK) {
		mfree(full);
		OSTRACE(("DLOPEN name=%s, handle=%p\n", filename, (void *)nullptr));
		return nullptr;
	}
	converted = winConvertFromUtf8Filename(full);
	mfree(full);
#else
	void *converted = winConvertFromUtf8Filename(filename);
	UNUSED_SYMBOL(sys);
#endif
	if (!converted) {
		OSTRACE(("DLOPEN name=%s, handle=%p\n", filename, (void *)nullptr));
		return nullptr;
	}
	HANDLE h;
	if (osIsNT())
#if LIBCU_OS_WINRT
		h = osLoadPackagedLibrary((LPCWSTR)converted, 0);
#else
		h = osLoadLibraryW((LPCWSTR)converted);
#endif
#ifdef LIBCU_WIN32_HAS_ANSI
	else h = osLoadLibraryA((char *)converted);
#endif
	OSTRACE(("DLOPEN name=%s, handle=%p\n", filename, (void *)h));
	mfree(converted);
	return (void *)h;
}
static void winDlError(vsystem *sys, int bufLength, char *bufOut)
{
	UNUSED_SYMBOL(sys);
	winGetLastErrorMsg(osGetLastError(), bufLength, bufOut);
}
static void (*winDlSym(vsystem*sys,void*h,const char*sym))()
{
	UNUSED_SYMBOL(sys);
	FARPROC proc = osGetProcAddressA((HANDLE)h, sym);
	OSTRACE(("DLSYM handle=%p, symbol=%s, address=%p\n", (void *)h, sym, (void *)proc));
	return (void(*)(void))proc;
}
static void winDlClose(vsystem *sys, void *h)
{
	UNUSED_SYMBOL(sys);
	osFreeLibrary((HANDLE)h);
	OSTRACE(("DLCLOSE handle=%p\n", (void *)h));
}
#else // if LIBCU_OMIT_LOAD_EXTENSION is defined:
#define winDlOpen  nullptr
#define winDlError nullptr
#define winDlSym   nullptr
#define winDlClose nullptr
#endif

/* State information for the randomness gatherer. */
typedef struct EntropyGatherer EntropyGatherer;
struct EntropyGatherer {
	unsigned char *a;   // Gather entropy into this buffer
	int aLength;        // Size of a[] in bytes
	int i;              // XOR next input into a[i]
	int xors;           // Number of XOR operations done
};

#if !defined(LIBCU_TEST) && !defined(LIBCU_OMIT_RANDOMNESS)
/* Mix size bytes of entropy into p. */
static void xorMemory(EntropyGatherer *p, unsigned char *x, int size)
{
	int j, k;
	for (j = 0, k = p->i; j < size; j++) {
		p->a[k++] ^= x[j];
		if (k >= p->aLength) k = 0;
	}
	p->i = k;
	p->xors += size;
}
#endif

/* Write up to nBuf bytes of randomness into zBuf. */
static int winRandomness(vsystem *sys, int bufLength, char *buf)
{
#if defined(LIBCU_TEST) || defined(LIBCU_OMIT_RANDOMNESS)
	UNUSED_SYMBOL(sys);
	memset(buf, 0, bufLength);
	return bufLength;
#else
	EntropyGatherer e;
	UNUSED_SYMBOL(sys);
	memset(buf, 0, bufLength);
#if defined(_MSC_VER) && _MSC_VER>=1400 && !LIBCU_OS_WINCE
	rand_s((unsigned int *)buf); // rand_s() is not available with MinGW
#endif
	e.a = (unsigned char *)buf;
	e.aLength = bufLength;
	e.xors = 0;
	e.i = 0;
	{
		SYSTEMTIME x;
		osGetSystemTime(&x);
		xorMemory(&e, (unsigned char *)&x, sizeof(SYSTEMTIME));
	}
	{
		DWORD pid = osGetCurrentProcessId();
		xorMemory(&e, (unsigned char *)&pid, sizeof(DWORD));
	}
#if LIBCU_OS_WINRT
	{
		ULONGLONG cnt = osGetTickCount64();
		xorMemory(&e, (unsigned char *)&cnt, sizeof(ULONGLONG));
	}
#else
	{
		DWORD cnt = osGetTickCount();
		xorMemory(&e, (unsigned char *)&cnt, sizeof(DWORD));
	}
#endif
	{
		LARGE_INTEGER i;
		osQueryPerformanceCounter(&i);
		xorMemory(&e, (unsigned char *)&i, sizeof(LARGE_INTEGER));
	}
#if !LIBCU_OS_WINCE && !LIBCU_OS_WINRT && LIBCU_WIN32_USE_UUID
	{
		UUID id;
		memset(&id, 0, sizeof(UUID));
		osUuidCreate(&id);
		xorMemory(&e, (unsigned char *)&id, sizeof(UUID));
		memset(&id, 0, sizeof(UUID));
		osUuidCreateSequential(&id);
		xorMemory(&e, (unsigned char *)&id, sizeof(UUID));
	}
#endif
	return e.xors > bufLength ? bufLength : e.xors;
#endif
}

/* Sleep for a little while.  Return the amount of time slept. */
static int winSleep(vsystem *sys, int microsec)
{
	sqlite3_win32_sleep((microsec+999)/1000);
	UNUSED_SYMBOL(sys);
	return ((microsec+999)/1000)*1000;
}

/* The following variable, if set to a non-zero value, is interpreted as the number of seconds since 1970 and is used to set the result of sqlite3OsCurrentTime() during testing. */
#ifdef LIBCU_TEST
int sqlite3_current_time = 0; // Fake system time in seconds since 1970.
#endif

/*
** Find the current time (in Universal Coordinated Time).  Write into *piNow the current time and date as a Julian Day number times 86_400_000.  In
** other words, write into *piNow the number of milliseconds since the Julian epoch of noon in Greenwich on November 24, 4714 B.C according to the
** proleptic Gregorian calendar.
**
** On success, return RC_OK.  Return LIBCU_ERROR if the time and date cannot be found.
*/
static RC winCurrentTimeInt64(vsystem *sys, int64_t *now)
{
	// FILETIME structure is a 64-bit value representing the number of 100-nanosecond intervals since January 1, 1601 (= JD 2305813.5).
	FILETIME ft;
	static const int64_t winFiletimeEpoch = 23058135*(int64_t)8640000;
#ifdef LIBCU_TEST
	static const int64_t unixEpoch = 24405875*(int64_t)8640000;
#endif
	// 2^32 - to avoid use of LL and warnings in gcc
	static const int64_t max32BitValue = (int64_t)2000000000 + (int64_t)2000000000 + (int64_t)294967296;
#if LIBCU_OS_WINCE
	SYSTEMTIME time;
	osGetSystemTime(&time);
	// if SystemTimeToFileTime() fails, it returns zero.
	if (!osSystemTimeToFileTime(&time, &ft))
		return RC_ERROR;
#else
	osGetSystemTimeAsFileTime(&ft);
#endif
	*now = winFiletimeEpoch + ((((int64_t)ft.dwHighDateTime)*max32BitValue) + (int64_t)ft.dwLowDateTime)/(int64_t)10000;
#ifdef LIBCU_TEST
	if (sqlite3_current_time)
		*nNow = 1000*(int64_t)sqlite3_current_time + unixEpoch;
#endif
	UNUSED_SYMBOL(sys);
	return RC_OK;
}

/*
** Find the current time (in Universal Coordinated Time).  Write the current time and date as a Julian Day number into *prNow and
** return 0.  Return 1 if the time and date cannot be found.
*/
static RC winCurrentTime(vsystem *sys, double *now)
{
	int64_t i;
	RC rc = winCurrentTimeInt64(sys, &i);
	if (!rc)
		*now = i/86400000.0;
	return rc;
}

/*
** The idea is that this function works like a combination of GetLastError() and FormatMessage() on Windows (or errno and
** strerror_r() on Unix). After an error is returned by an OS function, Libcu calls this function with zBuf pointing to
** a buffer of nBuf bytes. The OS layer should populate the buffer with a nul-terminated UTF-8 encoded error message
** describing the last IO error to have occurred within the calling thread.
**
** If the error message is too large for the supplied buffer, it should be truncated. The return value of xGetLastError
** is zero if the error message fits in the buffer, or non-zero otherwise (if the message was truncated). If non-zero is returned,
** then it is not necessary to include the nul-terminator character in the output buffer.
**
** Not supplying an error message will have no adverse effect on Libcu. It is fine to have an implementation that never
** returns an error message:
**
**   int xGetLastError(vsystem *sys, int nBuf, char *zBuf){
**     assert(zBuf[0]=='\0');
**     return 0;
**   }
**
** However if an error message is supplied, it will be incorporated by sqlite into the error message available to the user using
** sqlite3_errmsg(), possibly making IO errors easier to debug.
*/
static int winGetLastError(vsystem *sys, int bufLength, char *buf)
{
	DWORD e = osGetLastError();
	UNUSED_SYMBOL(sys);
	if (bufLength > 0) winGetLastErrorMsg(e, bufLength, buf);
	return e;
}

/* Initialize and deinitialize the operating system interface. */
RC sqlite3_os_init()
{
	static vsystem winVfs = {
		3,						// iVersion
		sizeof(winFile),		// szOsFile
		LIBCU_WIN32_MAX_PATH_BYTES, // mxPathname
		nullptr,				// pNext
		"win32",				// zName
		&winAppData,			// pAppData
		winOpen,				// xOpen
		winDelete,				// xDelete
		winAccess,				// xAccess
		winFullPathname,		// xFullPathname
		winDlOpen,				// xDlOpen
		winDlError,				// xDlError
		winDlSym,				// xDlSym
		winDlClose,				// xDlClose
		winRandomness,			// xRandomness
		winSleep,				// xSleep
		winCurrentTime,			// xCurrentTime
		winGetLastError,		// xGetLastError
		winCurrentTimeInt64,	// xCurrentTimeInt64
		winSetSystemCall,		// xSetSystemCall
		winGetSystemCall,		// xGetSystemCall
		winNextSystemCall,		// xNextSystemCall
	};
#if defined(LIBCU_WIN32_HAS_WIDE)
	static vsystem winLongPathVfs = {
		3,						// iVersion
		sizeof(winFile),		// szOsFile
		LIBCU_WINNT_MAX_PATH_BYTES, // mxPathname
		nullptr,				// pNext
		"win32-longpath",		// zName
		&winAppData,			// pAppData
		winOpen,				// xOpen
		winDelete,				// xDelete
		winAccess,				// xAccess
		winFullPathname,		// xFullPathname
		winDlOpen,				// xDlOpen
		winDlError,				// xDlError
		winDlSym,				// xDlSym
		winDlClose,				// xDlClose
		winRandomness,			// xRandomness
		winSleep,				// xSleep
		winCurrentTime,			// xCurrentTime
		winGetLastError,		// xGetLastError
		winCurrentTimeInt64,	// xCurrentTimeInt64
		winSetSystemCall,		// xSetSystemCall
		winGetSystemCall,		// xGetSystemCall
		winNextSystemCall,		// xNextSystemCall
	};
#endif
	static vsystem winNolockVfs = {
		3,						// iVersion
		sizeof(winFile),		// szOsFile
		LIBCU_WIN32_MAX_PATH_BYTES, // mxPathname
		nullptr,                // pNext
		"win32-none",			// zName
		&winNolockAppData,		// pAppData
		winOpen,				// xOpen
		winDelete,				// xDelete
		winAccess,				// xAccess
		winFullPathname,		// xFullPathname
		winDlOpen,				// xDlOpen
		winDlError,				// xDlError
		winDlSym,				// xDlSym
		winDlClose,				// xDlClose
		winRandomness,			// xRandomness
		winSleep,				// xSleep
		winCurrentTime,			// xCurrentTime
		winGetLastError,		// xGetLastError
		winCurrentTimeInt64,	// xCurrentTimeInt64
		winSetSystemCall,		// xSetSystemCall
		winGetSystemCall,		// xGetSystemCall
		winNextSystemCall,		// xNextSystemCall
	};
#if defined(LIBCU_WIN32_HAS_WIDE)
	static vsystem winLongPathNolockVfs = {
		3,						// iVersion
		sizeof(winFile),		// szOsFile
		LIBCU_WINNT_MAX_PATH_BYTES, // mxPathname
		nullptr,                // pNext
		"win32-longpath-none",	// zName
		&winNolockAppData,		// pAppData
		winOpen,				// xOpen
		winDelete,				// xDelete
		winAccess,				// xAccess
		winFullPathname,		// xFullPathname
		winDlOpen,				// xDlOpen
		winDlError,				// xDlError
		winDlSym,				// xDlSym
		winDlClose,				// xDlClose
		winRandomness,			// xRandomness
		winSleep,				// xSleep
		winCurrentTime,			// xCurrentTime
		winGetLastError,		// xGetLastError
		winCurrentTimeInt64,	// xCurrentTimeInt64
		winSetSystemCall,		// xSetSystemCall
		winGetSystemCall,		// xGetSystemCall
		winNextSystemCall,		// xNextSystemCall
	};
#endif
	// Double-check that the Syscalls[] array has been constructed correctly.  See ticket [bb3a86e890c8e96ab]
	assert(_LENGTHOF(Syscalls) == 80);
	// get memory map allocation granularity
	memset(&winSysInfo, 0, sizeof(SYSTEM_INFO));
#if LIBCU_OS_WINRT
	osGetNativeSystemInfo(&winSysInfo);
#else
	osGetSystemInfo(&winSysInfo);
#endif
	assert(winSysInfo.dwAllocationGranularity > 0);
	assert(winSysInfo.dwPageSize > 0);
	vsystemRegister(&winVfs, 1);
#if defined(LIBCU_WIN32_HAS_WIDE)
	vsystemRegister(&winLongPathVfs, 0);
#endif
	vsystemRegister(&winNolockVfs, 0);
#if defined(LIBCU_WIN32_HAS_WIDE)
	vsystemRegister(&winLongPathNolockVfs, 0);
#endif
	return RC_OK;
}

int sqlite3_os_end()
{
#if LIBCU_OS_WINRT
	if (sleepObj != NULL) {
		osCloseHandle(sleepObj);
		sleepObj = NULL;
	}
#endif
	return RC_OK;
}

#endif /* LIBCU_OS_WIN */
#pragma endregion
