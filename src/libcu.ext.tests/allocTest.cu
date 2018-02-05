#include <stdiocu.h>
#include <crtdefscu.h>
#include <ext\convert.h>
#include <assert.h>

static __global__ void g_alloc_test1()
{
	printf("alloc_test1\n");
}
cudaError_t alloc_test1() { g_alloc_test1<<<1, 1>>>(); return cudaDeviceSynchronize(); }