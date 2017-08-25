#include <cuda_runtimecu.h>
#include <sentinel.h>
#include <stdiocu.h>

cudaError_t crtdefs_test1();
cudaError_t ctype_test1();
cudaError_t dirent_test1();
cudaError_t errno_test1();
cudaError_t falloc_lauched_cuda_kernel();
cudaError_t falloc_alloc_with_getchunk();
cudaError_t falloc_alloc_with_getchunks();
cudaError_t falloc_alloc_with_context();
cudaError_t fcntl_test1(); // fails
cudaError_t fsystem_test1();
cudaError_t grp_test1();
cudaError_t pwd_test1();
cudaError_t regex_test1();
cudaError_t sentinel_test1();
cudaError_t setjmp_test1();
cudaError_t stdarg_parse();
cudaError_t stdarg_call();
cudaError_t stddef_test1();
cudaError_t stdio_test1(); // fails
cudaError_t stdio_64bit();
cudaError_t stdio_ganging();
cudaError_t stdio_scanf();
cudaError_t stdlib_test1(); // fails
cudaError_t stdlib_strtol();
cudaError_t stdlib_strtoq();
cudaError_t string_test1();
cudaError_t time_test1();
cudaError_t unistd_test1();

#define test stdlib_test1

int main()
{
	sentinelServerInitialize();

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaError_t cudaStatus = cudaSetDevice(gpuGetMaxGflopsDevice());
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?\n");
		goto Error;
	}
	cudaErrorCheck(cudaDeviceSetLimit(cudaLimitStackSize, 1024*5));

	// Launch test
	cudaStatus = test();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "test failed! %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "test launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

Error:
	sentinelServerShutdown();

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!\n");
		return 1;
	}

	// finish
	printf("\nPress any key to continue.\n");
	scanf("%c");

	return 0;
}
