#include <cuda_runtime.h>
#include <stdiocu.h>
#include <timecu.h>
#include <assert.h>

static __global__ void g_time_test1()
{
	printf("time_test1\n");
}
cudaError_t time_test1() { g_time_test1<<<1, 1>>>(); return cudaDeviceSynchronize(); }
