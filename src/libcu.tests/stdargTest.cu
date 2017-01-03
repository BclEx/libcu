#include <stdiocu.h>
#include <stdargcu.h>
#include <cuda_runtimecu.h>
#include <assert.h>

static __global__ void stdarg_test1()
{
	printf("stdargs_test1\n");
	va_list2<const char*, int> args;
	va_start(args, "Name", 4);
	char *a0 = va_arg(args, char*); assert(a0 == "Name");
	int a1 = va_arg(args, int); assert(a1 == 4);
	va_end(args);
}

void stdarg_()
{
	stdarg_test1<<<1, 1>>>();
}
