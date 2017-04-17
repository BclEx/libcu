#include <cuda_runtime.h>
#include <stdiocu.h>
#include <regexcu.h>
#include <assert.h>

static __device__ void exact()
{
	regex_t re;
	assert(!regcomp(&re, "sam", 0));

	regmatch_t pm;
	char str[128] = "onces sam lived with samle to win samile hehe sam hoho sam\0";
	int a = regexec(&re, &str[0], 1, &pm, REG_EXTENDED);
	assert(a == REG_NOERROR);

	int idx = 0; int offset = 0; int offsets[5];
	while (a == REG_NOERROR) {
		printf("%s match at %d\n", offset ? "next" : "first", offset + pm.rm_so);
		offsets[idx++] = offset + pm.rm_so;
		offset += pm.rm_eo;
		a = regexec(&re, &str[0] + offset, 1, &pm, 0);
	}
	assert(idx == 5);
	assert(offsets[0] == 6 && offsets[1] == 21 && offsets[2] == 34 && offsets[3] == 46 && offsets[4] == 55);
}

static __global__ void g_regex_test1()
{
	printf("regex_test1\n");
	exact();
}
cudaError_t regex_test1() { g_regex_test1<<<1, 1>>>(); return cudaDeviceSynchronize(); }


