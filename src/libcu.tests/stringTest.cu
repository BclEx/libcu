#include <cuda_runtimecu.h>
#include <stdiocu.h>
#include <stringcu.h>
#include <assert.h>

static __global__ void g_string_test1()
{
	printf("string_test1\n");
}
cudaError_t string_test1() { g_string_test1<<<1, 1>>>(); return cudaDeviceSynchronize(); }