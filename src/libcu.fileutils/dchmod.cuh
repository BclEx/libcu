#include <sys/statcu.h>
#include <unistdcu.h>

__device__ __managed__ int m_dchmod_rc;
__global__ void g_dchmod(char *str, int mode)
{
	m_dchmod_rc = (chmod(str, mode) < 0);
}
int dchmod(char *str, int mode)
{
	size_t strLength = strlen(str) + 1;
	char *d_str;
	cudaMalloc(&d_str, strLength);
	cudaMemcpy(d_str, str, strLength, cudaMemcpyHostToDevice);
	g_dchmod<<<1,1>>>(d_str, mode);
	cudaFree(d_str);
	return m_dchmod_rc;
}
