#include <sys/statcu.h>

__device__ __managed__ int m_dmkdir_rc;
__global__ void g_dmkdir(char *str, unsigned short mode)
{
	m_dmkdir_rc = mkdir(str, mode);
}

int dmkdir(char *str, unsigned short mode)
{
	g_dmkdir<<<1,1>>>(str, mode);
	return m_dmkdir_rc;
}
