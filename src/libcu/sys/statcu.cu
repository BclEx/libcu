#include <sys/statcu.h>
#include <sentinel-statmsg.h>
#include "../fsystem.h"

/* Get file attributes for FILE and put them in BUF.  */
__device__ int stat_(const char *__restrict file, struct stat *__restrict buf)
{
	if (file[0] != ':') {
		stat_stat msg(file, buf); return msg.RC;
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
__device__ int stat64_(const char *__restrict file, struct stat64 *__restrict buf)
{
	return 0;
}
__device__ int fstat64_device(int fd, struct stat64 *buf)
{
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
	if (path[0] != ':') {
		stat_mkdir msg(path, mode); return msg.RC;
	}
	fsystemMkdir(path, mode);
	return 0;
}

/* Create a new FIFO named PATH, with permission bits MODE.  */
__device__ int mkfifo_(const char *path, mode_t mode)
{
	panic("Not Implemented");
	return 0;
}
