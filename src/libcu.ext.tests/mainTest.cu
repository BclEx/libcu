#include <stdiocu.h>
#include <crtdefscu.h>
#include <ext\global.h>
#include <assert.h>

static __global__ void g_main_test1()
{
	printf("main_test1\n");
}
cudaError_t main_test1() { g_main_test1<<<1, 1>>>(); return cudaDeviceSynchronize(); }