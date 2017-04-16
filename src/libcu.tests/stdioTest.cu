#include <cuda_runtime.h>
#include <stdiocu.h>
#include <assert.h>

static __global__ void g_stdio_test1()
{
	printf("stdio_test1\n");

	//bool f0a = ISDEVICEFILE(stdio); bool f0b = ISDEVICEFILE(stdio+2); bool f0c = ISDEVICEFILE(stdio-1); assert(!f0a && f0b && f0c);

}
cudaError_t stdio_test1() { g_stdio_test1<<<1, 1>>>(); return cudaDeviceSynchronize(); }

static __global__ void g_stdio_64bit()
{
	printf("stdio_64bit\n");
	/*
	unsigned long long val = -1;
	void *ptr = (void *)-1;
	printf("%p\n", ptr);

	sscanf("123456789", "%Lx", &val);
	printf("val = %Lx\n", val);
	*/
}
cudaError_t stdio_64bit() { g_stdio_64bit<<<1, 1>>>(); return cudaDeviceSynchronize(); }

static __global__ void g_stdio_scanf()
{
	printf("stdio_scanf\n");
	/*
	const char *buf = "hello world";
	char *ps = NULL, *pc = NULL;
	char s[6], c;

	/ Check that %[...]/%c work. /
	sscanf(buf, "%[a-z] %c", s, &c);
	/ Check that %m[...]/%mc work. /
	sscanf(buf, "%m[a-z] %mc", &ps, &pc);

	if (strcmp(ps, "hello") != 0 || *pc != 'w' || strcmp(s, "hello") != 0 || c != 'w')
	return 1;

	free(ps);
	free(pc);

	return 0;
	*/
}
cudaError_t stdio_scanf() { g_stdio_scanf<<<1, 1>>>(); return cudaDeviceSynchronize(); }