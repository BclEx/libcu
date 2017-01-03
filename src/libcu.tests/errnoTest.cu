#include <stdiocu.h>
#include <errnocu.h>
#include <cuda_runtimecu.h>
#include <assert.h>

static __global__ void errno_test1()
{
	printf("errno_test1\n");
}

void errno_()
{
	errno_test1<<<1, 1>>>();
}
