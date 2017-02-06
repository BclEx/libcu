#define LIBCUFORCE
#include <cuda_runtimecu.h>
#include <errnocu.h>
#include <assert.h>

static __global__ void g_errno_test1()
{
	printf("errno_test1\n");
	//int a0 = errno; assert(a0);

	//_set_errno(3);
	//int b0 = errno; assert(b0 == 3);
	//int b1 = _get_errno(nullptr); assert(b1 == 3);
	//int b1a, b1b = _get_errno(&b1a); assert(b1a == b1b == 3);
}
cudaError_t errno_test1() { g_errno_test1<<<1, 1>>>(); return cudaDeviceSynchronize(); }
