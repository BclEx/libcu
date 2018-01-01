#include <stdiocu.h>
#include <stdargcu.h>
#include <assert.h>

static __global__ void g_stdarg_parse()
{
#ifdef __CUDA_ARCH__
	printf("stdarg_parse\n");
	va_list2<const char*, int> va;
	va_start(va, "Name", 4);
	char *a0 = va_arg(va, char*); assert(a0 == "Name");
	int a1 = va_arg(va, int); assert(a1 == 4);
	va_end(va);
#endif
}
cudaError_t stdarg_parse() { g_stdarg_parse<<<1, 1>>>(); return cudaDeviceSynchronize(); }

__device__ void methodVoid_(int cnt, va_list va)
{
	int value;
	for (int i = 1; i <= cnt; i++)
		assert((value = va_arg(va, int)) == i);
}
STDARG1void(methodVoid, methodVoid_(cnt, va), int cnt);
STDARG2void(methodVoid, methodVoid_(cnt, va), int cnt);
STDARG3void(methodVoid, methodVoid_(cnt, va), int cnt);

__device__ int methodRet_(int cnt, va_list va)
{
	int value = 0;
	for (int i = 1; i <= cnt; i++)
		assert((value = va_arg(va, int)) == i);
	return value;
}
STDARG1(int, methodRet, methodRet_(cnt, va), int cnt);
STDARG2(int, methodRet, methodRet_(cnt, va), int cnt);
STDARG3(int, methodRet, methodRet_(cnt, va), int cnt);

static __global__ void g_stdarg_call()
{
	printf("stdarg_call\n");

	methodVoid(0);
	methodVoid(1, 1);
	methodVoid(2, 1, 2);
	methodVoid(3, 1, 2, 3);
	methodVoid(4, 1, 2, 3, 4);
	methodVoid(5, 1, 2, 3, 4, 5);
	methodVoid(6, 1, 2, 3, 4, 5, 6);
	methodVoid(7, 1, 2, 3, 4, 5, 6, 7);
	methodVoid(8, 1, 2, 3, 4, 5, 6, 7, 8);
	methodVoid(9, 1, 2, 3, 4, 5, 6, 7, 8, 9);
	methodVoid(10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
	methodVoid(11, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11);
	methodVoid(12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);
	methodVoid(13, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13);
	methodVoid(14, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14);
	methodVoid(15, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
	methodVoid(16, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
	methodVoid(17, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17);
	methodVoid(18, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18);
	methodVoid(19, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19);
	methodVoid(20, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20);

	assert(methodRet(0) == 0);
	assert(methodRet(1, 1) == 1);
	assert(methodRet(2, 1, 2) == 2);
	assert(methodRet(3, 1, 2, 3) == 3);
	assert(methodRet(4, 1, 2, 3, 4) == 4);
	assert(methodRet(5, 1, 2, 3, 4, 5) == 5);
	assert(methodRet(6, 1, 2, 3, 4, 5, 6) == 6);
	assert(methodRet(7, 1, 2, 3, 4, 5, 6, 7) == 7);
	assert(methodRet(8, 1, 2, 3, 4, 5, 6, 7, 8) == 8);
	assert(methodRet(9, 1, 2, 3, 4, 5, 6, 7, 8, 9) == 9);
	assert(methodRet(10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10) == 10);
	assert(methodRet(11, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11) == 11);
	assert(methodRet(12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12) == 12);
	assert(methodRet(13, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13) == 13);
	assert(methodRet(14, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14) == 14);
	assert(methodRet(15, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15) == 15);
	assert(methodRet(16, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16) == 16);
	assert(methodRet(17, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17) == 17);
	assert(methodRet(18, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18) == 18);
	assert(methodRet(19, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19) == 19);
	assert(methodRet(20, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20) == 20);
}
cudaError_t stdarg_call() { g_stdarg_call<<<1, 1>>>(); return cudaDeviceSynchronize(); }
