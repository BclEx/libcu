#include <stdiocu.h>
#include <crtdefscu.h>
#include <ext\global.h>
#include <assert.h>

static __global__ void g_util_test1()
{
	printf("util_test1\n");
}
cudaError_t util_test1() { g_util_test1<<<1, 1>>>(); return cudaDeviceSynchronize(); }