#include <stdiocu.h>
#include <crtdefscu.h>
#include <ext\global.h>
#include <assert.h>

static __global__ void g_random_test1()
{
	printf("random_test1\n");
}
cudaError_t random_test1() { g_random_test1<<<1, 1>>>(); return cudaDeviceSynchronize(); }