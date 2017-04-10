#include <cuda_runtime.h>
#include <stdiocu.h>
#include <sentinel.h>
#include <assert.h>

static __global__ void g_sentinel_test1()
{
	printf("sentinel_test1\n");
}
cudaError_t sentinel_test1() { g_sentinel_test1<<<1, 1>>>(); return cudaDeviceSynchronize(); }
