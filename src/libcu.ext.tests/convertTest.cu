#include <stdiocu.h>
#include <crtdefscu.h>
#include <ext\convert.h>
#include <assert.h>

static __global__ void g_convert_test1()
{
	printf("convert_test1\n");
}
cudaError_t convert_test1() { g_convert_test1<<<1, 1>>>(); return cudaDeviceSynchronize(); }