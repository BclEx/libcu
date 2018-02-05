#include <stdiocu.h>
#include <crtdefscu.h>
#include <ext\global.h>
#include <assert.h>

static __global__ void g_vsystem_test1()
{
	printf("vsystem_test1\n");
}
cudaError_t vsystem_test1() { g_vsystem_test1<<<1, 1>>>(); return cudaDeviceSynchronize(); }