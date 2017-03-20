#include <unistdcu.h>

__device__ __managed__ int m_dpwd_rc;
__global__ void g_dpwd(char *str)
{
	getcwd(str, MAX_PATH);
	m_dpwd_rc = 0;
}
int dpwd(char *str)
{
	char *d_str;
	cudaMalloc(&d_str, MAX_PATH);
	g_dpwd<<<1,1>>>(d_str);
	cudaMemcpy(str, d_str, MAX_PATH, cudaMemcpyDeviceToHost);
	cudaFree(d_str);
	return m_dcat_rc;
}