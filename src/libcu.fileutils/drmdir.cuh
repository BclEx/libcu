#include <unistdcu.h>

__device__ __managed__ int m_drmdir_rc;
__global__ void g_drmdir(char *str)
{
	m_drmdir_rc = rmdir(str);
}
int drmdir(char *str)
{
	size_t strLength = strlen(str) + 1;
	char *d_str;
	cudaMalloc(&d_str, strLength);
	cudaMemcpy(d_str, str, strLength, cudaMemcpyHostToDevice);
	g_drmdir<<<1,1>>>(d_str);
	cudaFree(d_str);
	return m_drmdir_rc;
}
