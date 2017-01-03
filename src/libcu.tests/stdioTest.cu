#include <stdiocu.h>
#include <cuda_runtimecu.h>
#include <assert.h>

static __global__ void stdio_test1()
{
	printf("stdio_test1\n");
}

/*
static __global__ void stdio_64bit()
{
	printf("stdio_64bit\n");
	unsigned long long val = -1;
	void *ptr = (void *)-1;
	printf("%p\n", ptr);

	sscanf("123456789", "%Lx", &val);
	printf("val = %Lx\n", val);
	return 0;
}

static __global__ void stdio_scanf()
{
	printf("stdio_scanf\n");
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
}
*/

void stdio_()
{
	stdio_test1<<<1, 1>>>();
}