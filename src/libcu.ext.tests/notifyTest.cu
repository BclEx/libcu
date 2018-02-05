#include <stdiocu.h>
#include <crtdefscu.h>
#include <ext\global.h>
#include <assert.h>

static __global__ void g_notify_test1()
{
	printf("notify_test1\n");
}
cudaError_t notify_test1() { g_notify_test1<<<1, 1>>>(); return cudaDeviceSynchronize(); }