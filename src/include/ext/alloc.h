/*
alloc.h - xxx
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
#ifndef _EXT_ALLOC_H
#define _EXT_ALLOC_H
__BEGIN_DECLS;

/*
** EVIDENCE-OF: R-25715-37072 Memory allocation statistics are enabled by default unless SQLite is compiled with SQLITE_DEFAULT_MEMSTATUS=0 in
** which case memory allocation statistics are disabled by default.
*/
#if !defined(LIBCU_DEFAULTMEMSTATUS)
#define LIBCU_DEFAULTMEMSTATUS true
#endif

/*
** Exactly one of the following macros must be defined in order to specify which memory allocation subsystem to use.
**
**     LIBCU_SYSTEM_MALLOC          // Use normal system malloc()
**     LIBCU_WIN32_MALLOC           // Use Win32 native heap API
**     LIBCU_ZERO_MALLOC            // Use a stub allocator that always fails
**     LIBCU_MEMDEBUG               // Debugging version of system malloc()
**
** On Windows, if the SQLITE_WIN32_MALLOC_VALIDATE macro is defined and the assert() macro is enabled, each call into the Win32 native heap subsystem
** will cause HeapValidate to be called.  If heap validation should fail, an assertion will be triggered.
**
** If none of the above are defined, then set LIBCU_SYSTEM_MALLOC as the default.
*/
#if defined(LIBCU_SYSTEM_MALLOC) + defined(LIBCU_WIN32_MALLOC) + defined(LIBCU_ZERO_MALLOC) + defined(LIBCU_MEMDEBUG) > 1
#error "Two or more of the following compile-time configuration options are defined but at most one is allowed: LIBCU_SYSTEM_MALLOC, LIBCU_WIN32_MALLOC, LIBCU_MEMDEBUG, LIBCU_ZERO_MALLOC"
#endif
#if defined(LIBCU_SYSTEM_MALLOC) + defined(LIBCU_WIN32_MALLOC) + defined(LIBCU_ZERO_MALLOC) + defined(LIBCU_MEMDEBUG)==0
#define LIBCU_SYSTEM_MALLOC 1
#endif

/* If LIBCU_MALLOCSOFTLIMIT is not zero, then try to keep the sizes of memory allocations below this value where possible. */
#if !defined(LIBCU_MALLOCSOFTLIMIT)
# define LIBCU_MALLOCSOFTLIMIT 1024
#endif

// CAPI3REF: Memory Allocation Routines
typedef struct alloc_methods alloc_methods;
struct alloc_methods {
	void *(*alloc)(int);			// Memory allocation function
	void (*free)(void *);			// Free a prior allocation
	void *(*realloc)(void *, int);	// Resize an allocation
	int (*size)(void *);			// Return the size of an allocation
	int (*roundup)(int);			// Round up request size to allocation size
	int (*initialize)(void *);		// Initialize the memory allocator
	int (*shutdown)(void *);		// Deinitialize the memory allocator
	void *appData;					// Argument to _.init() and _.shutdown()
};
#define __allocsystem _runtimeConfig.allocSystem

// CAPI3REF: Attempt To Free Heap Memory
__host_device__ int alloc_releasememory(int size);

// CAPI3REF: Free Memory Used By A Tag Object
//__host_device__ int alloc__tagreleasememory(tagbase_t *);

// CAPI3REF: Impose A Limit On Heap Size
__host_device__ int64_t alloc_softheaplimit64(int64_t size);

// CAPI3REF: Memory Allocation Subsystem
__host_device__ void *alloc32(int size);
__host_device__ void *alloc64(uint64_t size);
__host_device__ void *alloc_realloc32(void *prior, int newSize);
__host_device__ void *alloc_realloc64(void *prior, uint64_t newSize);
__host_device__ void mfree(void *p);
__host_device__ uint64_t alloc_msize(void *p);

// CAPI3REF: Memory Allocator Statistics
__host_device__ int64_t alloc_memoryused();
__host_device__ int64_t alloc_memoryhighwater(bool resetFlag);

/* Access to mutexes used by sqlite3_status() */
__host_device__ mutex *allocMutex();
__host_device__ mutex *pcacheMutex();

__host_device__ RC allocInitialize();
__host_device__ RC allocShutdown();
__host_device__ void *alloc(uint64_t size);
__host_device__ void *allocZero(uint64_t size);
__host_device__ void *tagallocZero(tagbase_t *tag, uint64_t size);
__host_device__ void *tagallocRaw(tagbase_t *tag, uint64_t size);
__host_device__ void *tagallocRawNN(tagbase_t *tag, uint64_t size);
__host_device__ char *tagstrdup(tagbase_t *tag, const char *z);
__host_device__ char *tagstrndup(tagbase_t *tag, const char *z, uint64_t size);
__host_device__ void *allocRealloc(void *prior, uint64_t newSize);
__host_device__ void *tagreallocOrFree(tagbase_t *tag, void *p, uint64_t newSize);
__host_device__ void *tagrealloc(tagbase_t *tag, void *p, uint64_t newSize);
__host_device__ void tagfree(tagbase_t *tag, void *p);
__host_device__ void tagfreeNN(tagbase_t *tag, void *p);
__host_device__ int allocSize(void *p);
__host_device__ int tagallocSize(tagbase_t *tag, void *p);
__host_device__ void *scratchAlloc(int size);
__host_device__ void scratchFree(void *p);
//__host_device__ void *sqlite3PageMalloc(int);
//__host_device__ void sqlite3PageFree(void*);
__host_device__ void __allocsystemSetDefault();
#ifndef LIBCU_UNTESTABLE
__host_device__ void allocBenignHook(void (*)(void), void (*)(void));
#endif
__host_device__ bool allocHeapNearlyFull();
__host_device__ void tagOomFault(tagbase_t *tag);
__host_device__ void tagOomClear(tagbase_t *tag);
__host_device__ RC tagApiExit(tagbase_t *tag, RC rc);

/* Available fault injectors.  Should be numbered beginning with 0. */
#define LIBCU_FAULTINJECTOR_MALLOC     0
#define LIBCU_FAULTINJECTOR_COUNT      1

/* The interface to the code in fault.c used for identifying "benign" malloc failures. This is only present if LIBCU_UNTESTABLE is not defined. */
#ifndef LIBCU_UNTESTABLE
__host_device__ void allocBenignBegin();
__host_device__ void allocBenignEnd();
#else
#define allocBenignBegin()
#define allocBenignEnd()
#endif

/*
** On systems with ample stack space and that support alloca(), make use of alloca() to obtain space for large automatic objects.  By default,
** obtain space from malloc().
**
** The alloca() routine never returns NULL.  This will cause code paths that deal with sqlite3StackAlloc() failures to be unreachable.
*/
#ifdef LIBCU_USEALLOCA
#define tagstackAllocRaw(tag, size) alloca(size)
#define tagstackAllocZero(tag, size) memset(alloca(size), 0, size)
#define tagstackFree(tag, ptr)
#else
#define tagstackAllocRaw(tag, size) tagallocRaw(tag, size)
#define tagstackAllocZero(tag, size) tagallocZero(tag, size)
#define tagstackFree(tag, ptr) tagfree(tag, ptr)
#endif

/* Do not allow both MEMSYS5 and MEMSYS3 to be defined together.  If they are, disable MEMSYS3 */
#ifdef LIBCU_ENABLE_MEMSYS5
const alloc_methods *sqlite3MemGetMemsys5();
#undef LIBCU_ENABLE_MEMSYS3
#endif
#ifdef LIBCU_ENABLE_MEMSYS3
const alloc_methods *sqlite3MemGetMemsys3();
#endif

/*
** These routines are available for the mem2.c debugging memory allocator only.  They are used to verify that different "types" of memory
** allocations are properly tracked by the system.
**
** memdbg_settype() sets the "type" of an allocation to one of the MEMTYPE_* macros defined below.  The type must be a bitmask with
** a single bit set.
**
** memdbg_hastype() returns true if any of the bits in its second argument match the type set by the previous memdbg_settype().
** memdbg_hastype() is intended for use inside assert() statements.
**
** memdbg_notype() returns true if none of the bits in its second argument match the type set by the previous memdbg_settype().
**
** Perhaps the most important point is the difference between MEMTYPE_HEAP and MEMTYPE_LOOKASIDE.  If an allocation is MEMTYPE_LOOKASIDE, that means
** it might have been allocated by lookaside, except the allocation was too large or lookaside was already full.  It is important to verify
** that allocations that might have been satisfied by lookaside are not passed back to non-lookaside free() routines.  Asserts such as the
** example above are placed on the non-lookaside free() routines to verify this constraint.
**
** All of this is no-op for a production build.  It only comes into play when the MEMDEBUG compile-time option is used.
*/
#ifdef MEMDEBUG
void memdbg_settype(void *, uint8_t);
int memdbg_hastype(void *, uint8_t);
int memdbg_nottype(void *, uint8_t);
#else
#define memdbg_settype(X, Y) /* no-op */
#define memdbg_hastype(X, Y) 1
#define memdbg_nottype(X, Y) 1
#endif
#define MEMTYPE_HEAP       0x01  // General heap allocations
#define MEMTYPE_LOOKASIDE  0x02  // Heap that might have been lookaside
#define MEMTYPE_SCRATCH    0x04  // Scratch allocations
#define MEMTYPE_PCACHE     0x08  // Page cache allocations

__END_DECLS;
#endif	/* _EXT_ALLOC_H */