#include <stdlibcu.h> //: mem1.c
#include <ext/alloc.h>
#include <assert.h>

/* This version of the memory allocator is the default.  It is used when no other memory allocator is specified using compile-time macros. */
#ifdef LIBCU_SYSTEM_MALLOC
#if defined(__APPLE__) && !defined(LIBCU_WITHOUT_ZONEMALLOC)

/* Use the zone allocator available on apple products unless the LIBCU_WITHOUT_ZONEMALLOC symbol is defined. */
#include <sys/sysctl.h>
#include <malloc/malloc.h>
#ifdef LIBCU_MIGHTBE_SINGLECORE
#include <libkern/OSAtomic.h>
#endif
static malloc_zone_t *_sqliteZone_;
#define LIBCU_MALLOC(x) malloc_zone_malloc(_sqliteZone_, (x))
#define LIBCU_FREE(x) malloc_zone_free(_sqliteZone_, (x));
#define LIBCU_REALLOC(x,y) malloc_zone_realloc(_sqliteZone_, (x), (y))
#define LIBCU_MALLOCSIZE(x) (_sqliteZone_ ? _sqliteZone_->size(_sqliteZone_,x) : malloc_size(x))

#else /* if not __APPLE__ */

/* Use standard C library malloc and free on non-Apple systems.  Also used by Apple systems if SQLITE_WITHOUT_ZONEMALLOC is defined. */
#define LIBCU_MALLOC(x) malloc(x)
#define LIBCU_FREE(x) free(x)
#define LIBCU_REALLOC(x,y) realloc((x),(y))

/* The malloc.h header file is needed for malloc_usable_size() function on some systems (e.g. Linux). */
#if HAVE_MALLOC_H && HAVE_MALLOC_USABLE_SIZE
#define LIBCU_USEMALLOC_H 1
#define LIBCU_USEMALLOC_USABLESIZE 1
/*
** The MSVCRT has malloc_usable_size(), but it is called _msize().  The use of _msize() is automatic, but can be disabled by compiling with
** -DLIBCU_WITHOUT_MSIZE.  Using the _msize() function also requires the malloc.h header file.
*/
#elif defined(_MSC_VER) && !defined(LIBCU_WITHOUT_MSIZE)
#define LIBCU_USEMALLOC_H
#define LIBCU_USEMSIZE
#endif

/* Include the malloc.h header file, if necessary.  Also set define macro LIBCU_MALLOCSIZE to the appropriate function name, which is _msize()
** for MSVC and malloc_usable_size() for most other systems (e.g. Linux). The memory size function can always be overridden manually by defining
** the macro LIBCU_MALLOCSIZE to the desired function name.
*/
#if defined(LIBCU_USEMALLOC_H)
#include <malloc.h>
#if defined(LIBCU_USE_MALLOC_USABLE_SIZE)
# if !defined(LIBCU_MALLOCSIZE)
# define LIBCU_MALLOCSIZE(x) malloc_usable_size(x)
# endif
# elif defined(LIBCU_USEMSIZE)
# if !defined(LIBCU_MALLOCSIZE)
# define LIBCU_MALLOCSIZE _msize
# endif
# endif
#endif

#endif /* __APPLE__ or not __APPLE__ */

/* Like malloc(), but remember the size of the allocation so that we can find it later using sqlite3MemSize().
**
** For this low-level routine, we are guaranteed that nByte>0 because cases of nByte<=0 will be intercepted and dealt with by higher level routines.
*/
static __host_device__ void *memoryMalloc(int size)
{
#ifdef LIBCU_MALLOCSIZE
	TESTCASE(ROUND8_(size) == size);
	void *p = LIBCU_MALLOC(size);
	if (!p) {
		TESTCASE(_runtimeConfig.log);
		_log(RC_NOMEM, "failed to allocate %u bytes of memory", size);
	}
	return p;
#else
	assert(size > 0);
	TESTCASE(ROUND8_(size) != size);
	int64_t *p = LIBCU_MALLOC(size+8);
	if (p) {
		p[0] = size;
		p++;
	}
	else {
		TESTCASE( _runtimeConfig.log);
		_log(RC_NOMEM, "failed to allocate %u bytes of memory", size);
	}
	return (void *)p;
#endif
}

/* Like free() but works for allocations obtained from sqlite3MemMalloc() or sqlite3MemRealloc().
**
** For this low-level routine, we already know that prior!=0 since cases where prior==0 will have been intecepted and dealt with
** by higher-level routines.
*/
static __host_device__ void memoryFree(void *prior)
{
#ifdef LIBCU_MALLOCSIZE
	LIBCU_FREE(prior);
#else
	int64_t *p = (int64_t *)prior;
	assert(prior);
	p--;
	LIBCU_FREE(p);
#endif
}

/* Report the allocated size of a prior return from xMalloc() or xRealloc(). */
static __host_device__ int memorySize(void *prior)
{
#ifdef LIBCU_MALLOCSIZE
	assert(prior);
	return (int)LIBCU_MALLOCSIZE(prior);
#else
	assert(prior);
	int64_t *p = (int64_t *)prior;
	p--;
	return (int)p[0];
#endif
}

/* Like realloc().  Resize an allocation previously obtained from sqlite3MemMalloc().
**
** For this low-level interface, we know that pPrior!=0.  Cases where pPrior==0 while have been intercepted by higher-level routine and
** redirected to xMalloc.  Similarly, we know that nByte>0 because cases where nByte<=0 will have been intercepted by higher-level
** routines and redirected to xFree.
*/
static __host_device__ void *memoryRealloc(void *prior, int size)
{
#ifdef LIBCU_MALLOCSIZE
	void *p = LIBCU_REALLOC(prior, size);
	if (!p){
		TESTCASE(_runtimeConfig.log);
		_log(RC_NOMEM, "failed memory resize %u to %u bytes", LIBCU_MALLOCSIZE(prior), size);
	}
	return p;
#else
	int64_t *p = (int64_t *)prior;
	assert(prior!=0 && size > 0);
	assert(ROUND8(size) == size); // EV: R-46199-30249
	p--;
	p = LIBCU_REALLOC(p, size+8);
	if (p) {
		p[0] = size;
		p++;
	}
	else {
		TESTCASE(_runtimeConfig.log);
		_log(RC_NOMEM, "failed memory resize %u to %u bytes", memorySize(prior), size);
	}
	return (void *)p;
#endif
}

/* Round up a request size to the next valid allocation size. */
static __host_device__ int memoryRoundup(int size)
{
	return ROUND8_(size);
}

/* Initialize this module. */
static __host_device__ RC memoryInitialize(void *notUsed)
{
#if defined(__APPLE__) && !defined(LIBCU_WITHOUT_ZONEMALLOC)
	if (_sqliteZone_)
		return RC_OK;
	size_t len = sizeof(cpuCount);
	// One usually wants to use hw.acctivecpu for MT decisions, but not here
	int cpuCount; sysctlbyname("hw.ncpu", &cpuCount, &len, NULL, 0);
	if (cpuCount > 1)
		_sqliteZone_ = malloc_default_zone(); // defer MT decisions to system malloc
	else {
		_sqliteZone_ = malloc_create_zone(4096, 0); // only 1 core, use our own zone to contention over global locks, e.g. we have our own dedicated locks
		malloc_set_zone_name(_sqliteZone_, "Sqlite_Heap");
	}
#endif
	UNUSED_SYMBOL(notUsed);
	return RC_OK;
}

/* Deinitialize this module. */
static __host_device__ RC memoryShutdown(void *notUsed)
{
	UNUSED_SYMBOL(notUsed);
	return RC_OK;
}

/* This routine is the only routine in this file with external linkage.
**
** Populate the low-level memory allocation function pointers in sqlite3GlobalConfig.m with pointers to the routines in this file.
*/
static __host_constant__ const alloc_methods _mem1DefaultMethods = {
	memoryMalloc,
	memoryFree,
	memoryRealloc,
	memorySize,
	memoryRoundup,
	memoryInitialize,
	memoryShutdown,
	nullptr
};
__device__ void __allocsystemSetDefault()
{
	//sqlite3_config(CONFIG_MALLOC, &defaultMethods);
	__allocsystem = _mem1DefaultMethods;
}

#endif /* LIBCU_SYSTEM_MALLOC */
