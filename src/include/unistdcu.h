/*
unistd.h - Symbolic Constants
The MIT License

Copyright (c) 2016 Sky Morey

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#pragma once

#if !__CUDACC__
#include <unistd.h>
#elif !defined(_UNISTD_H)
#define	_UNISTD_H
#include <featurescu.h>

__BEGIN_DECLS;

/* Values for the second argument to access.
These may be OR'd together.  */
#define	R_OK	4		/* Test for read permission.  */
#define	W_OK	2		/* Test for write permission.  */
#define	X_OK	1		/* Test for execute permission.  */
#define	F_OK	0		/* Test for existence.  */

/* Test for access to NAME using the real UID and real GID.  */
extern __device__ int access(const char *name, int type);

/* Values for the WHENCE argument to lseek.  */
#ifndef	_STDIO_H		/* <stdio.h> has the same definitions.  */
# define SEEK_SET	0	/* Seek from beginning of file.  */
# define SEEK_CUR	1	/* Seek from current position.  */
# define SEEK_END	2	/* Seek from end of file.  */
#endif

/* Move FD's file position to OFFSET bytes from the beginning of the file (if WHENCE is SEEK_SET),
the current position (if WHENCE is SEEK_CUR), or the end of the file (if WHENCE is SEEK_END).
Return the new file position.  */
#ifndef __USE_FILE_OFFSET64
extern off_t lseek(int fd, off_t offset, int whence);
#else
#define lseek lseek64
#endif
#ifdef __USE_LARGEFILE64
extern off64_t lseek64(int fd, off64_t offset, int whence);
#endif

/* Close the file descriptor FD.  */
extern __device__ int close(int fd);

/* Read NBYTES into BUF from FD.  Return the number read, -1 for errors or 0 for EOF.  */
extern __device__ size_t read(int fd, void *buf, size_t nbytes);

/* Write N bytes of BUF to FD.  Return the number written, or -1.  */
extern __device__ size_t write(int fd, const void *buf, size_t n);

/* Create a one-way communication channel (pipe). If successful, two file descriptors are stored in PIPEDES;
bytes written on PIPEDES[1] can be read from PIPEDES[0]. Returns 0 if successful, -1 if not.  */
//nosupport: extern __device__ int pipe(int pipedes[2]);

/* Schedule an alarm.  In SECONDS seconds, the process will get a SIGALRM. If SECONDS is zero, any currently scheduled alarm will be cancelled.
The function returns the number of seconds remaining until the last alarm scheduled would have signaled, or zero if there wasn't one.
There is no return value to indicate an error, but you can set `errno' to 0 and check its value after calling `alarm', and this might tell you.
The signal may come late due to processor scheduling.  */
//nosupport: extern __device__ unsigned int alarm(unsigned int seconds);

/* Make the process sleep for SECONDS seconds, or until a signal arrives and is not ignored.  The function returns the number of seconds less
than SECONDS which it actually slept (thus zero if it slept the full time). If a signal handler does a `longjmp' or modifies the handling of the
SIGALRM signal while inside `sleep' call, the handling of the SIGALRM signal afterwards is undefined.  There is no return value to indicate
error, but if `sleep' returns SECONDS, it probably didn't work.  */
extern __device__ unsigned int sleep(unsigned int seconds);

/* Suspend the process until a signal arrives. This always returns -1 and sets `errno' to EINTR.  */
//nosupport: extern int pause(void);

/* Change the owner and group of FILE.  */
//extern __device__ int chown(const char *file, uid_t owner, gid_t group);

/* Change the process's working directory to PATH.  */
extern __device__ int chdir(const char *path);

/* Get the pathname of the current working directory, and put it in SIZE bytes of BUF.  Returns NULL if the
directory couldn't be determined or SIZE was too small. If successful, returns BUF.  In GNU, if BUF is NULL,
an array is allocated with `malloc'; the array is SIZE bytes long, unless SIZE == 0, in which case it is as
big as necessary.  */
extern __device__ char *getcwd(char *buf, size_t size);

/* Duplicate FD, returning a new file descriptor on the same file.  */
extern __device__ int dup(int fd);

/* Duplicate FD to FD2, closing FD2 and making it open on the same file.  */
extern __device__ int dup2(int fd, int fd2);

/* NULL-terminated array of "NAME=VALUE" environment variables.  */
extern __device__ char **__environ;

/* Terminate program execution with the low-order 8 bits of STATUS.  */
//nosupport: extern __device__ void _exit(int status);

/* Get file-specific configuration information about PATH.  */
//nosupport: extern __device__ long int pathconf(const char *path, int name);

/* Get file-specific configuration about descriptor FD.  */
//nosupport: extern __device__ long int fpathconf(int fd, int name);

__END_DECLS;

#endif  /* _UNISTD_H */
