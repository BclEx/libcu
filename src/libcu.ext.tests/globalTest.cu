#include <stdiocu.h>
#include <crtdefscu.h>
#include <ext\global.h>
#include <assert.h>

static __global__ void gGLOBAL__test1()
{
	printf("global_test1\n");
}
cudaError_t global_test1() { gGLOBAL__test1<<<1, 1>>>(); return cudaDeviceSynchronize(); }
