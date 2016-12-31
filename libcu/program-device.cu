#include <stdlibcu.h>
#include <stdiocu.h>
#include <cuda_runtimecu.h>
#include <common_functions.h>

__global__ void addKernel(int *c, const int *a, const int *b)
{
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
	if (i != 1)
		return;

	//printf("%d %s\n", 2, "sky morey");

	FILE *f = fopen("C:\\T_\\fopen.txt", "w");
	_fprintf(f, "The quick brown fox jumps over the lazy dog");
	//fwrite("test", 4, 1, f);
	fflush(f);
	fclose(f);

	//const char buf[100] = {0};
	//snprintf(buf, 100, "test");
	//printf("%s\n", buf);
	//printf("%d\n", atoi("51236"));
	//printf("%f\n", atof("1.2"));
}
