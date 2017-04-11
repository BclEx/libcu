#include <cuda_runtime.h>
#include <stdiocu.h>
//#include "../libcu/fsystem.h"
#include <assert.h>

static __global__ void g_fsystem_test1()
{
	printf("fsystem_test1\n");
}
cudaError_t fsystem_test1() { g_fsystem_test1<<<1, 1>>>(); return cudaDeviceSynchronize(); }
