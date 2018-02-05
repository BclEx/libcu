#include <stdiocu.h>
#include <crtdefscu.h>
#include <ext\global.h>
#include <assert.h>

static __global__ void g_utf_test1()
{
	printf("utf_test1\n");
}
cudaError_t utf_test1() { g_utf_test1<<<1, 1>>>(); return cudaDeviceSynchronize(); }