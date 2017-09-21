#include <cuda_runtime.h>
#include <stdiocu.h>
#include <xstddefcu.h>
#include <assert.h>

static __global__ void g_stddef_test1()
{
	printf("stddef_test1\n");
}
cudaError_t stddef_test1() { g_stddef_test1<<<1, 1>>>(); return cudaDeviceSynchronize(); }

