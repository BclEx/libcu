#include <stdiocu.h>
#include <crtdefscu.h>
#include <ext\global.h>
#include <assert.h>

static __global__ void g_math_test1()
{
	printf("math_test1\n");
}
cudaError_t math_test1() { g_math_test1<<<1, 1>>>(); return cudaDeviceSynchronize(); }