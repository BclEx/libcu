#include <unistdcu.h>

__device__ __managed__ int m_dcd_rc;
__global__ void g_dcd(char *str)
{
	m_dcd_rc = 0;
}
int dcd(char *str)
{
	size_t strLength = strlen(str) + 1;
	char *d_str;
	cudaMalloc(&d_str, strLength);
	cudaMemcpy(d_str, str, strLength, cudaMemcpyHostToDevice);
	g_dcd<<<1,1>>>(d_str);
	cudaFree(d_str);
	return m_dcd_rc;
}