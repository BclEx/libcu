#include <cuda_runtimecu.h>
#include <sentinel.h>
#include <stdlibcu.h>
#include <stdiocu.h>

cudaError_t crtdefs_test1();
cudaError_t ctype_test1();
cudaError_t dirent_test1();
cudaError_t errno_test1();
cudaError_t falloc_lauched_cuda_kernel();
cudaError_t falloc_alloc_with_getchunk();
cudaError_t falloc_alloc_with_getchunks();
cudaError_t falloc_alloc_with_context();
cudaError_t fcntl_test1();
cudaError_t fsystem_test1();
cudaError_t grp_test1();
cudaError_t pwd_test1();
cudaError_t regex_test1();
cudaError_t sentinel_test1();
cudaError_t setjmp_test1();
cudaError_t stdarg_parse();
cudaError_t stdarg_call();
cudaError_t stddef_test1();
cudaError_t stdio_test1();
cudaError_t stdio_64bit();
cudaError_t stdio_ganging();
cudaError_t stdio_scanf();
cudaError_t stdlib_test1();
cudaError_t stdlib_strtol();
cudaError_t stdlib_strtoq();
cudaError_t string_test1();
cudaError_t time_test1();
cudaError_t unistd_test1();

#if _HASPAUSE
#define mainPause(fmt) { printf(fmt"\n"); char c; scanf("%c", &c); }
#else
#define mainPause(fmt) { printf(fmt"\n"); }
#endif

int main(int argc, char ** argv)
{
	int testId = argv[1] ? atoi(argv[1]) : 25; //0; //18; //25;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaError_t cudaStatus = cudaSetDevice(gpuGetMaxGflopsDevice());
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?\n");
		goto Error;
	}
	cudaErrorCheck(cudaDeviceSetLimit(cudaLimitStackSize, 1024*5));
	sentinelServerInitialize();
	sentinelRegisterFileUtils();

	// Launch test
	switch (testId)
	{
	case 0: mainPause("Press any key to continue."); break;
	case 1: cudaStatus = crtdefs_test1(); break;
	case 2: cudaStatus = ctype_test1(); break;
	case 3: cudaStatus = dirent_test1(); break;
	case 4: cudaStatus = errno_test1(); break;
	case 5: cudaStatus = falloc_lauched_cuda_kernel(); break;
	case 6: cudaStatus = falloc_alloc_with_getchunk(); break; // memory access
	case 7: cudaStatus = falloc_alloc_with_getchunks(); break;
	case 8: cudaStatus = falloc_alloc_with_context(); break; // memory access
	case 9: cudaStatus = fcntl_test1(); break;
	case 10: cudaStatus = grp_test1(); break;
	case 11: cudaStatus = pwd_test1(); break;
	case 12: cudaStatus = regex_test1(); break;
	case 13: cudaStatus = sentinel_test1(); break;
	case 14: cudaStatus = setjmp_test1(); break;
	case 15: cudaStatus = stdarg_parse(); break;
	case 16: cudaStatus = stdarg_call(); break;
	case 17: cudaStatus = stddef_test1(); break;
	case 18: cudaStatus = stdio_test1(); break; // assert
	case 19: cudaStatus = stdio_64bit(); break;
	case 20: cudaStatus = stdio_ganging(); break;
	case 21: cudaStatus = stdio_scanf(); break;
	case 22: cudaStatus = stdlib_test1(); break; // assert
	case 23: cudaStatus = stdlib_strtol(); break;
	case 24: cudaStatus = stdlib_strtoq(); break;
	case 25: cudaStatus = string_test1(); break;
	case 26: cudaStatus = time_test1(); break;
	case 27: cudaStatus = unistd_test1(); break; // missing device, throws on fast run
		// default
	default: cudaStatus = crtdefs_test1(); break;
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
