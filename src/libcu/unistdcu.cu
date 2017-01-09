#include <unistdcu.h>
#include <cuda_runtimecu.h>

/* Test for access to NAME using the real UID and real GID.  */
__device__ int access(const char *name, int type)
{
	return 0;
}

/* Move FD's file position to OFFSET bytes from the beginning of the file (if WHENCE is SEEK_SET),
the current position (if WHENCE is SEEK_CUR), or the end of the file (if WHENCE is SEEK_END).
Return the new file position.  */
__device__ off_t lseek(int fd, off_t offset, int whence)
{
	return 0;
}

#ifdef __USE_LARGEFILE64
__device__ off64_t lseek64(int fd, off64_t offset, int whence)
{
	return 0;
}
#endif

/* Close the file descriptor FD.  */
__device__ int close(int fd)
{
	return 0;
}

/* Read NBYTES into BUF from FD.  Return the number read, -1 for errors or 0 for EOF.  */
__device__ size_t read(int fd, void *buf, size_t nbytes)
{
	return 0;
}

/* Write N bytes of BUF to FD.  Return the number written, or -1.  */
__device__ size_t write(int fd, const void *buf, size_t n)
{
	return 0;
}

/* Make the process sleep for SECONDS seconds, or until a signal arrives and is not ignored.  The function returns the number of seconds less
than SECONDS which it actually slept (thus zero if it slept the full time). If a signal handler does a `longjmp' or modifies the handling of the
SIGALRM signal while inside `sleep' call, the handling of the SIGALRM signal afterwards is undefined.  There is no return value to indicate
error, but if `sleep' returns SECONDS, it probably didn't work.  */
__device__ void usleep(unsigned long milliseconds)
{
	clock_t start = clock();
	clock_t end = milliseconds * 10;
	for (;;) {
		clock_t now = clock();
		clock_t cycles = (now > start ? now - start : now + (0xffffffff - start));
		if (cycles >= end) break;
	}
}

/* Change the process's working directory to PATH.  */
__device__ int chdir(const char *path)
{
	return 0;
}

/* Get the pathname of the current working directory, and put it in SIZE bytes of BUF.  Returns NULL if the
directory couldn't be determined or SIZE was too small. If successful, returns BUF.  In GNU, if BUF is NULL,
an array is allocated with `malloc'; the array is SIZE bytes long, unless SIZE == 0, in which case it is as
big as necessary.  */
__device__ char *getcwd(char *buf, size_t size)
{
	return "GPU";
}

/* Duplicate FD, returning a new file descriptor on the same file.  */
__device__ int dup(int fd)
{
	return 0;
}

/* Duplicate FD to FD2, closing FD2 and making it open on the same file.  */
__device__ int dup2(int fd, int fd2)
{
	return 0;
}

/* NULL-terminated array of "NAME=VALUE" environment variables.  */
__device__ char *__environ_device[3] = { "HOME=", "PATH=", nullptr }; // pointer to environment table

__BEGIN_DECLS;
extern __device__ char **__environ = (char **)__environ_device;
__END_DECLS;