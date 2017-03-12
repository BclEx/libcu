//#include <sentinel.h>
//#include "futils.h"
//#include <sys/types.h>
//#include <sys/stat.h>
//#include <fcntl.h>
//#include <signal.h>
//#include <errno.h>

#define BUF_SIZE 1024 

// Copy one file to another, while possibly preserving its modes, times, and modes.  Returns TRUE if successful, or FALSE on a failure with an
// error message output.  (Failure is not indicted if the attributes cannot be set.)
__device__ bool copyFile(char *srcName, char *destName, bool setModes)
{
	struct stat statbuf1;
	if (stat(srcName, &statbuf1) < 0) {
		perror(srcName);
		return false;
	}
	struct stat statbuf2;
	if (stat(destName, &statbuf2) < 0) {
		statbuf2.st_ino = -1;
		statbuf2.st_dev = -1;
	}
	if (statbuf1.st_dev == statbuf2.st_dev && statbuf1.st_ino == statbuf2.st_ino) {
		fprintf(stderr, "Copying file \"%s\" to itself\n", srcName);
		return false;
	}
	//
	FILE *rfd = fopen(srcName, "r");
	if (!rfd) {
		perror(srcName);
		return false;
	}
	FILE *wfd = fopen(destName, statbuf1.st_mode);
	if (!wfd) {
		perror(destName);
		fclose(rfd);
		return false;
	}
	//
	char *buf = (char *)malloc(BUF_SIZE);
	int rcc;
	while ((rcc = fread(buf, 1, BUF_SIZE, rfd)) > 0) {
		char *bp = buf;
		while (rcc > 0) {
			int wcc = fwrite(bp, 1, rcc, wfd);
			if (wcc < 0) {
				perror(destName);
				goto error_exit;
			}
			bp += wcc;
			rcc -= wcc;
		}
	}
	if (rcc < 0) {
		perror(srcName);
		goto error_exit;
	}
	free(buf);
	fclose(rfd);
	if (fclose(wfd) < 0) {
		perror(destName);
		return false;
	}
	if (setModes) {
		chmod(destName, statbuf1.st_mode);
		chown(destName, statbuf1.st_uid, statbuf1.st_gid);
		struct utimbuf times;
		times.actime = statbuf1.st_atime;
		times.modtime = statbuf1.st_mtime;
		utime(destName, &times);
	}
	return true;

error_exit:
	free(buf);
	fclose(rfd);
	fclose(wfd);
	return false;
}

__device__ __managed__ int m_dcp_rc;
__global__ void g_dcp(char *srcName, char *destName, bool setModes)
{
	m_dcp_rc = copyFile(srcName, destName, setModes);
}
int dcp(char *str, char *str2, bool setModes)
{
	size_t strLength = strlen(str) + 1;
	size_t str2Length = strlen(str2) + 1;
	char *d_str;
	char *d_str2;
	cudaMalloc(&d_str, strLength);
	cudaMalloc(&d_str2, str2Length);
	cudaMemcpy(d_str, str, strLength, cudaMemcpyHostToDevice);
	cudaMemcpy(d_str2, str2, str2Length, cudaMemcpyHostToDevice);
	g_dcp<<<1,1>>>(d_str, d_str2);
	cudaFree(d_str);
	cudaFree(d_str2);
	return m_dcp_rc;
}
