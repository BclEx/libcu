#include <unistdcu.h>

__device__ __managed__ int m_drmdir_rc;
__global__ void g_drmdir(char *str)
{
	m_drmdir_rc = rmdir(str);
}

int drmdir(char *str)
{
	g_drmdir<<<1,1>>>(str);
	return m_drmdir_rc;
}
