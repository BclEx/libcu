#include <cuda_runtime.h>
#include <stdiocu.h>
#include <grpcu.h>
#include <assert.h>

static __global__ void g_grp_test1()
{
	printf("grp_test1\n");
}
cudaError_t grp_test1() { g_grp_test1<<<1, 1>>>(); return cudaDeviceSynchronize(); }
