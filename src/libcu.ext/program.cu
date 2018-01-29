#include <cuda_runtimecu.h>
#include <sentinel.h>
#include <stdlibcu.h>
#include <stdiocu.h>
#include <ext\global.h>

static __global__ void g_test1()
{
	printf("test1\n");

	void *a = alloc32(10);
	mfree(a);
}
cudaError_t test1() { g_test1<<<1, 1>>>(); return cudaDeviceSynchronize(); }

#if _HASPAUSE
#define mainPause(fmt) { printf(fmt"\n"); char c; scanf("%c", &c); }
#else
#define mainPause(fmt) { printf(fmt"\n"); }
#endif

int main(int argc, char ** argv)
{
	int testId = argv[1] ? atoi(argv[1]) : 1;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaError_t cudaStatus = cudaSetDevice(gpuGetMaxGflopsDevice());
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?\n");
		goto Error;
	}
	cudaErrorCheck(cudaDeviceSetLimit(cudaLimitStackSize, 1024*5));
	sentinelServerInitialize();

	// Launch test
	switch (testId)
	{
	case 0: mainPause("Press any key to continue."); break;
	case 1: cudaStatus = test1(); break;
	default: break;
	}
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "failed! %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// finish
	mainPause("SUCCESS");

Error:
	sentinelServerShutdown();

	// close
	if (cudaStatus != cudaSuccess) {
		// finish
		mainPause("ERROR");
		return 1;
	}

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!\n");
		return 1;
	}

	return 0;
}
