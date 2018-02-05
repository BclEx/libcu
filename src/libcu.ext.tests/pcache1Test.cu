#include <stdiocu.h>
#include <crtdefscu.h>
#include <ext\global.h>
#include <assert.h>

static __global__ void g_pcache_test1()
{
	printf("pcache_test1\n");
}
cudaError_t pcache_test1() { g_pcache_test1<<<1, 1>>>(); return cudaDeviceSynchronize(); }