#include <cuda_runtime.h>
#include <stdiocu.h>
#include <direntcu.h>
#include <assert.h>

static __global__ void g_dirent_test1()
{
	printf("dirent_test1\n");
}
cudaError_t dirent_test1() { g_dirent_test1<<<1, 1>>>(); return cudaDeviceSynchronize(); }
