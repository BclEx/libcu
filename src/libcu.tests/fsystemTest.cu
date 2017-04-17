#include <cuda_runtime.h>
#include <stdiocu.h>
#include <stringcu.h>
#include "../libcu/fsystem.h"
#include <assert.h>

static __global__ void g_fsystem_test1()
{
	printf("fsystem_test1\n");
	char newPath[MAX_PATH]; 
	strcpy(__cwd, ":\\Test");
	expandPath(":\\one", newPath); printf("%s\n", newPath);
	expandPath(":\\one\\", newPath); printf("%s\n", newPath);
	expandPath(":\\one\\.", newPath); printf("%s\n", newPath);
	expandPath(":\\one\.\\", newPath); printf("%s\n", newPath);
}
cudaError_t fsystem_test1() { g_fsystem_test1<<<1, 1>>>(); return cudaDeviceSynchronize(); }
