//#include <sentinel.h>
//#include "futils.h"
//
//#include <sys/types.h>
#include <sys/statcu.h>
//#include <unistd.h>
//#include <fcntl.h>
//#include <signal.h>
//#include <pwd.h>
//#include <grp.h>
//#include <utime.h>
//#include <errno.h>

#define BUF_SIZE 1024 
//typedef	struct chunk CHUNK;
//#define	CHUNKINITSIZE 4
//struct chunk {
//	CHUNK *next;
//	char data[CHUNKINITSIZE]; // actually of varying length
//};
//static CHUNK *_chunkList;

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
	int rfd = open(srcName, 0);
	if (rfd < 0) {
		perror(srcName);
		return false;
	}
	int wfd = creat(destName, statbuf1.st_mode);
	if (wfd < 0) {
		perror(destName);
		close(rfd);
		return false;
	}
	//
	char *buf = (char *)malloc(BUF_SIZE);
	int rcc;
	while ((rcc = read(rfd, buf, BUF_SIZE)) > 0) {
		char *bp = buf;
		while (rcc > 0) {
			int wcc = write(wfd, bp, rcc);
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
	close(rfd);
	if (close(wfd) < 0) {
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
	close(rfd);
	close(wfd);
	return false;
}

__device__ __managed__ int m_dmv_rc;
__global__ void g_dmv(char *srcName, char *destName)
{
	m_dmv_rc = false;
	if (access(srcName, 0) < 0) {
		perror(srcName);
		return;
	}
	if (rename(srcName, destName) >= 0)
		return;
	if (errno != EXDEV) {
		perror(destName);
		return;
	}
	if (!copyFile(srcName, destName, true))
		return;
	if (unlink(srcName) < 0)
		perror(srcName);
	m_dmv_rc = 1;
}
int dmv(char *str, char *str2)
{
	size_t strLength = strlen(str) + 1;
	size_t str2Length = strlen(str2) + 1;
	char *d_str;
	char *d_str2;
	cudaMalloc(&d_str, strLength);
	cudaMalloc(&d_str2, str2Length);
	cudaMemcpy(d_str, str, strLength, cudaMemcpyHostToDevice);
	cudaMemcpy(d_str2, str2, str2Length, cudaMemcpyHostToDevice);
	g_dmv<<<1,1>>>(d_str, d_str2);
	cudaFree(d_str);
	cudaFree(d_str2);
	return m_dmv_rc;
}
