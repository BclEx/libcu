#include <sys/statcu.h>

__device__ __managed__ int m_dmkdir_rc;
__global__ void g_dmkdir(char *str, unsigned short mode)
{
	m_dmkdir_rc = mkdir(str, mode);
}
int dmkdir(char *str, unsigned short mode)
{
	size_t strLength = strlen(str) + 1;
	char *d_str;
	cudaMalloc(&d_str, strLength);
	cudaMemcpy(d_str, str, strLength, cudaMemcpyHostToDevice);
	g_dmkdir<<<1,1>>>(d_str, mode);
	cudaFree(d_str);
	return m_dmkdir_rc;
}
