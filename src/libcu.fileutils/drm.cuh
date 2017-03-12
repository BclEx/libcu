#include <sys/statcu.h>

__device__ __managed__ int m_drm_rc;
__global__ void g_drm(char *str)
{
	struct stat sbuf;
	m_drm_rc = (!lstat(str, &sbuf) && unlink(str));
}
int drm(char *str)
{
	size_t strLength = strlen(str) + 1;
	char *d_str;
	cudaMalloc(&d_str, strLength);
	cudaMemcpy(d_str, str, strLength, cudaMemcpyHostToDevice);
	g_drm<<<1,1>>>(d_str);
	cudaFree(d_str);
	return m_drm_rc;
}
