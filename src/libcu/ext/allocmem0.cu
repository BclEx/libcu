#include <ext/alloc.h>
#include <assert.h>

/* This version of the memory allocator is the default.  It is used when no other memory allocator is specified using compile-time macros. */
#ifdef LIBCU_ZERO_MALLOC

/* No-op versions of all memory allocation routines */
static __host_device__ void *memoryAlloc(int size) { return nullptr; }
static __host_device__ void memoryFree(void *prior) { }
static __host_device__ void *memoryRealloc(void *prior, int newSize) { return nullptr; }
static __host_device__ int memorySize(void *prior){ return 0; }
static __host_device__ int memoryRoundup(int size) { return n; }
static __host_device__ RC memoryInitialize(void *notUsed) { return RC_OK; }
static __host_device__ RC memoryShutdown(void *notUsed) { return RC_OK; }

/*
** This routine is the only routine in this file with external linkage.
**
** Populate the low-level memory allocation function pointers in _.alloc with pointers to the routines in this file.
*/
static __host_constant__ const alloc_methods _defaultMethods = {
	memoryAlloc,
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
	//sqlite3_config(SQLITE_CONFIG_MALLOC, &defaultMethods);
	__allocsystem = _defaultMethods;
}

#endif /* LIBCU_ZERO_MALLOC */
