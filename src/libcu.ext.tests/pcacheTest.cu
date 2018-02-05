#include <stdiocu.h>
#include <crtdefscu.h>
#include <ext\global.h>
#include <assert.h>

static __global__ void g_pcache1_test1()
{
	printf("pcache1_test1\n");
}
cudaError_t pcache1_test1() { g_pcache1_test1<<<1, 1>>>(); return cudaDeviceSynchronize(); }