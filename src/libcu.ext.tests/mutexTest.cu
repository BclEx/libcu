#include <stdiocu.h>
#include <crtdefscu.h>
#include <ext\global.h>
#include <assert.h>

static __global__ void g_mutex_test1()
{
	printf("mutex_test1\n");
}
cudaError_t mutex_test1() { g_mutex_test1<<<1, 1>>>(); return cudaDeviceSynchronize(); }