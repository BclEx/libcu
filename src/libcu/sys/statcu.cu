#include <sys/statcu.h>
#include <sentinel-fcntlmsg.h>
#include "../fsystem.h"

/* Get file attributes for FILE and put them in BUF.  */
__device__ int stat_(const char *__restrict file, struct stat *__restrict buf, bool lstat)
{
	if (!ISDEVICEPATH(file)) {
		fcntl_stat msg(file, buf, lstat); return msg.RC;
	}
	panic("Not Implemented");
	return 0;
}

/* Get file attributes for the file, device, pipe, or socket that file descriptor FD is open on and put them in BUF.  */
__device__ int fstat_device(int fd, struct stat *buf)
{
	panic("Not Implemented");
	return 0;
}

#ifdef __USE_LARGEFILE64
/* Get file attributes for FILE and put them in BUF.  */
__device__ int stat64_(const char *__restrict file, struct stat64 *__restrict buf, bool lstat)
{
	if (!ISDEVICEPATH(file)) {
		fcntl_stat64 msg(file, buf, lstat); return msg.RC;
	}
	panic("Not Implemented");
	return 0;
}

/* Get file attributes for the file, device, pipe, or socket that file descriptor FD is open on and put them in BUF.  */
__device__ int fstat64_device(int fd, struct stat64 *buf)
{
	panic("Not Implemented");
	return 0;
}
#endif

/* Set file access permissions for FILE to MODE. If FILE is a symbolic link, this affects its target instead.  */
__device__ int chmod_(const char *file, mode_t mode)
{
	panic("Not Implemented");
	return 0;
}

/* Set the file creation mask of the current process to MASK, and return the old creation mask.  */
__device__ mode_t umask_(mode_t mask)
{
	panic("Not Implemented");
	return 0;
}

/* Create a new directory named PATH, with permission bits MODE.  */
__device__ int mkdir_(const char *path, mode_t mode)
{
	if (!ISDEVICEPATH(path)) {
		fcntl_mkdir msg(path, mode); return msg.RC;
	}
	int r; fsystemMkdir(path, mode, &r); return r;
}

/* Create a new FIFO named PATH, with permission bits MODE.  */
__device__ int mkfifo_(const char *path, mode_t mode)
{
	panic("Not Implemented");
	return 0;
}
