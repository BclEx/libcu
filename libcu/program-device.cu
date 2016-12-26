#include <stdlibcu.h>
//#include <stdiocu.h>
#include <cuda_runtimecu.h>
#include <stdio.h>

__global__ void addKernel(int *c, const int *a, const int *b)
{
	//const char buf[100] = {0};
	//snprintf(buf, 100, "test");
	//int test = atoi("51236");
	printf("%d\n", atoi("51236"));
	printf("%f\n", atof("1.2"));

    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}
