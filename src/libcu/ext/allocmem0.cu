/* This version of the memory allocator is the default.  It is used when no other memory allocator is specified using compile-time macros. */
#ifdef LIBCU_ZERO_MALLOC

static __host_device__ void *memMalloc(int size) { return 0; }
static __host_device__ void memFree(void *prior) { }
static __host_device__ void *memRealloc(void *prior, int size) { return 0; }
static __host_device__ size_t memSize(void *prior ){ return 0; }
static __host_device__ size_t memRoundup(int size) { return n; }
static __host_device__ RC memInit(void *notUsed1) { return RC_OK; }
static __host_device__ RC memShutdown(void *notUsed1) { return RC_OK; }

// This routine is the only routine in this file with external linkage.
//
// Populate the low-level memory allocation function pointers in sqlite3GlobalConfig.m with pointers to the routines in this file.
static __host_constant__ const alloc_methods _defaultMethods = {
	memMalloc,
	memFree,
	memRealloc,
	memSize,
	memRoundup,
	memInititalize,
	memShutdown,
	nullptr
};
__device__ void __allocsystem_setdefault()
{
	//sqlite3_config(SQLITE_CONFIG_MALLOC, &defaultMethods);
	__allocsystem = _defaultMethods;
}

#endif /* LIBCU_ZERO_MALLOC */
