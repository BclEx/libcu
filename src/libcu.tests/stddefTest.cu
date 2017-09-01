#include <cuda_runtime.h>
#include <stdiocu.h>
#include <xstddefcu.h>
#include <assert.h>

static __global__ void g_stddef_test1()
{
	printf("stddef_test1\n");

	//// PRINTF ////
	//_Check_return_opt_ _CRTIMP int __cdecl printf(_In_z_ _Printf_format_string_ const char *_Format, ...);

	//// TAGALLOC, TAGFREE, TAGREALLOC ////
	//__forceinline __device__ void *tagalloc(void *tag, size_t size) { return nullptr; }
	//__forceinline __device__ void tagfree(void *tag, void *p) { }
	//__forceinline __device__ void *tagrealloc(void *tag, void *old, size_t size) { return nullptr; }
}
cudaError_t stddef_test1() { g_stddef_test1<<<1, 1>>>(); return cudaDeviceSynchronize(); }

