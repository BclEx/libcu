#include <cuda_runtimecu.h>
#include <sentinel.h>
#include <stdlibcu.h>
#include <stdiocu.h>
#include <ext\global.h>

cudaError_t alloc_test1();
cudaError_t bitvec_test1();
cudaError_t convert_test1();
cudaError_t global_test1();
cudaError_t main_test1();
cudaError_t math_test1();
cudaError_t mutex_test1();
cudaError_t notify_test1();
cudaError_t pcache_test1();
cudaError_t pcache1_test1();
cudaError_t printf_test1();
cudaError_t random_test1();
cudaError_t status_test1();
cudaError_t utf_test1();
cudaError_t util_test1();
cudaError_t vsystem_test1();

#if _HASPAUSE
#define mainPause(fmt) { printf(fmt"\n"); char c; scanf("%c", &c); }
#else
#define mainPause(fmt) { printf(fmt"\n"); }
#endif

int main(int argc, char **argv)
{
	int testId = argv[1] ? atoi(argv[1]) : 0;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaError_t cudaStatus = cudaSetDevice(gpuGetMaxGflopsDevice());
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?\n");
		goto Error;
	}
	cudaErrorCheck(cudaDeviceSetLimit(cudaLimitStackSize, 1024*5));
	sentinelServerInitialize();
	runtimeInitialize();

	// Launch test
	switch (testId)
	{
	case 0: mainPause("Press any key to continue."); break;
	case 1: cudaStatus = alloc_test1(); break;
	case 2: cudaStatus = bitvec_test1(); break;
	case 3: cudaStatus = convert_test1(); break;
	case 4: cudaStatus = global_test1(); break;
	case 5: cudaStatus = math_test1(); break;
	case 6: cudaStatus = main_test1(); break;
	case 7: cudaStatus = mutex_test1(); break;
	case 8: cudaStatus = notify_test1(); break;
	case 9: cudaStatus = pcache_test1(); break;
	case 10: cudaStatus = pcache1_test1(); break;
	case 11: cudaStatus = printf_test1(); break;
	case 12: cudaStatus = random_test1(); break;
	case 13: cudaStatus = status_test1(); break;
	case 14: cudaStatus = utf_test1(); break;
	case 15: cudaStatus = util_test1(); break;
	case 16: cudaStatus = vsystem_test1(); break;
		// default
	default: cudaStatus = bitvec_test1(); break;
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
	runtimeShutdown();
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
