#include <stdiocu.h>
#include <crtdefscu.h>
#include <ext\global.h>
#include <assert.h>

static __global__ void g_status_test1()
{
	printf("status_test1\n");
}
cudaError_t status_test1() { g_status_test1<<<1, 1>>>(); return cudaDeviceSynchronize(); }