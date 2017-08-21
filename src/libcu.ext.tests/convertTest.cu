#include <stdiocu.h>
#include <crtdefscu.h>
#include <ext\convert.h>
#include <assert.h>

static __global__ void g_ext_convert_test1()
{
	printf("ext_convert_test1\n");
}
cudaError_t ext_convert_test1() { g_ext_convert_test1<<<1, 1>>>(); return cudaDeviceSynchronize(); }