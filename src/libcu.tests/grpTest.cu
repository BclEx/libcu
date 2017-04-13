#include <cuda_runtime.h>
#include <stdiocu.h>
#include <grpcu.h>
#include <assert.h>

static __global__ void g_grp_test1()
{
	printf("grp_test1\n");

	struct group *a0a = getgrgid(0); struct group *a0b = getgrgid(1); bool a0c = !strcmp(a0b->gr_name, "std"); assert(!a0a && a0b && a0c);

	struct group *b0a = getgrnam(nullptr);
	struct group *b0b = getgrnam("");
	struct group *b0c = getgrnam("abc"); struct group *b0d = getgrnam("std"); bool b0e = !strcmp(b0d->gr_name, "std"); assert(!b0a && !b0b && !b0c && b0d && b0e);

}
cudaError_t grp_test1() { g_grp_test1<<<1, 1>>>(); return cudaDeviceSynchronize(); }
