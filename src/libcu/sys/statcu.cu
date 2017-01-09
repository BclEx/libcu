#include <sys/statcu.h>

/* Get file attributes for FILE and put them in BUF.  */
__device__ int stat(const char *__restrict file, struct stat *__restrict buf)
{
	return 0;
}

/* Get file attributes for the file, device, pipe, or socket that file descriptor FD is open on and put them in BUF.  */
__device__ int fstat(int fd, struct stat *buf)
{
	return 0;
}

#ifdef __USE_LARGEFILE64
__device__ int stat64(const char *__restrict file, struct stat64 *__restrict buf)
{
	return 0;
}
__device__ int fstat64(int fd, struct stat64 *buf)
{
	return 0;
}
#endif

/* Set file access permissions for FILE to MODE. If FILE is a symbolic link, this affects its target instead.  */
__device__ int chmod(const char *file, mode_t mode)
{
	return 0;
}

/* Set the file creation mask of the current process to MASK, and return the old creation mask.  */
__device__ mode_t umask(mode_t mask)
{
	return 0;
}

/* Create a new directory named PATH, with permission bits MODE.  */
__device__ int mkdir(const char *path, mode_t mode)
{
	return 0;
}

/* Create a new FIFO named PATH, with permission bits MODE.  */
__device__ int mkfifo(const char *path, mode_t mode)
{
	return 0;
}
