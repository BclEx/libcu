#include <cuda_runtime.h>
#include <stdiocu.h>
#include <fcntlcu.h>
#include <assert.h>

static __global__ void g_fcntl_test1()
{
	printf("fcntl_test1\n");
}
cudaError_t fcntl_test1() { g_fcntl_test1<<<1, 1>>>(); return cudaDeviceSynchronize(); }
