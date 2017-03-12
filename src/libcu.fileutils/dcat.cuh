#include <stdiocu.h>
#include <errnocu.h>

#define CAT_BUF_SIZE 4096
__device__ void dumpfile(FILE *f)
{
	size_t nred;
	char readbuf[CAT_BUF_SIZE];
	while ((nred = fread(readbuf, 1, CAT_BUF_SIZE, f)) > 0)
		fwrite(readbuf, nred, 1, stdout);
}

__device__ __managed__ int m_dcat_rc;
__global__ void g_dcat(char *str)
{
	FILE *f = fopen(str, "r");
	if (!f)
		m_dcat_rc = errno;
	else {
		dumpfile(f);
		fclose(f);
		m_dcat_rc = 0;
	}
}
int dcat(char *str)
{
	size_t strLength = strlen(str) + 1;
	char *d_str;
	cudaMalloc(&d_str, strLength);
	cudaMemcpy(d_str, str, strLength, cudaMemcpyHostToDevice);
	g_dcat<<<1,1>>>(d_str);
	cudaFree(d_str);
	return m_dcat_rc;
	//int rc; cudaMemcpyFromSymbol(&rc, "d_dcat_rc", sizeof(rc), 0, cudaMemcpyDeviceToHost); return rc;
}