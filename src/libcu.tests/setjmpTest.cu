#include <cuda_runtime.h>
#include <stdiocu.h>
#include <setjmpcu.h>
#include <assert.h>

static __global__ void g_setjmp_test1()
{
	printf("setjmp_test1\n");
}
cudaError_t setjmp_test1() { g_setjmp_test1<<<1, 1>>>(); return cudaDeviceSynchronize(); }
