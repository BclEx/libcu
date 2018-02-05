#include <stdiocu.h>
#include <crtdefscu.h>
#include <ext\global.h>
#include <assert.h>

static __global__ void g_printf_test1()
{
	printf("printf_test1\n");
}
cudaError_t printf_test1() { g_printf_test1<<<1, 1>>>(); return cudaDeviceSynchronize(); }