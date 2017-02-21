#include <cuda_runtime.h>
#include <stdio.h>
#include <assert.h>

#define _ROUND8(x) (((x)+7)&~7)
#define panic(fmt, ...) { printf(fmt, __VA_ARGS__); asm("trap;"); }

#ifdef	__cplusplus
#define __BEGIN_DECLS extern "C" {
#define __END_DECLS }
#else
#define __BEGIN_DECLS
#define __END_DECLS
#endif

__BEGIN_DECLS;

#ifndef _WIN64
#define MALLOCSIZETYPE long int
#else
#define MALLOCSIZETYPE long long int
#endif
__device__ void *malloc_(size_t size)
{
	assert(size > 0);
	size = _ROUND8(size);
	MALLOCSIZETYPE *p = (MALLOCSIZETYPE *)malloc(sizeof(MALLOCSIZETYPE) + size);
	if (p)
		p[0] = size;
	else panic("failed to allocate %u bytes of memory", size);
	return (void *)(p+1);
}

__device__ void *calloc_(size_t nmemb, size_t size)
{
	return malloc(size);
}

__device__ void free_(void *ptr)
{
	assert(ptr);
	MALLOCSIZETYPE *p = (MALLOCSIZETYPE *)ptr;
	free(p-1);
}

__device__ void *realloc_(void *ptr, size_t size)
{
	assert(size > 0);
	size = _ROUND8(size);
	MALLOCSIZETYPE *p = (MALLOCSIZETYPE *)malloc(sizeof(MALLOCSIZETYPE) + size);
	if (p)
		p[0] = size;
	else panic("failed to allocate %u bytes of memory", size);
	if (ptr)
	{ 
		MALLOCSIZETYPE *p2 = (MALLOCSIZETYPE *)ptr;
		size_t ptrSize = (size_t)p2[0];
		if (ptrSize) memcpy(p+1, p2+1, ptrSize);
		free(p2-1);
	}
	return (void *)(p+1);
}

__END_DECLS;
