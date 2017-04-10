#include <cuda_runtime.h>
#include <stdiocu.h>
#include <unistdcu.h>
#include <assert.h>

static __global__ void g_unistd_test1()
{
	printf("unistd_test1\n");
}
cudaError_t unistd_test1() { g_unistd_test1<<<1, 1>>>(); return cudaDeviceSynchronize(); }
