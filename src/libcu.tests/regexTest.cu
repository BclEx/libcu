#include <cuda_runtime.h>
#include <stdiocu.h>
#include <regexcu.h>
#include <assert.h>

static __global__ void g_regex_test1()
{
	printf("regex_test1\n");
}
cudaError_t regex_test1() { g_regex_test1<<<1, 1>>>(); return cudaDeviceSynchronize(); }
