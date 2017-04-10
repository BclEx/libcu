#include <cuda_runtime.h>
#include <stdiocu.h>
#include <pwdcu.h>
#include <assert.h>

static __global__ void g_pwd_test1()
{
	printf("pwd_test1\n");
}
cudaError_t pwd_test1() { g_pwd_test1<<<1, 1>>>(); return cudaDeviceSynchronize(); }
