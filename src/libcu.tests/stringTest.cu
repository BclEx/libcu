#include <cuda_runtimecu.h>
#include <assert.h>

static __global__ void string_test1()
{
	printf("string_test1\n");
}

void string_()
{
	string_test1<<<1, 1>>>();
}