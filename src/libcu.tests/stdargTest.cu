#include <cuda_runtimecu.h>
#include <assert.h>

static __global__ void stdarg_test1()
{
#ifdef __CUDA_ARCH__
	printf("stdargs_test1\n");
	va_list2<const char*, int> va;
	va_start(va, "Name", 4);
	char *a0 = va_arg(va, char*); assert(a0 == "Name");
	int a1 = va_arg(va, int); assert(a1 == 4);
	va_end(va);
#endif
}

void stdarg_()
{
	stdarg_test1<<<1, 1>>>();
}
