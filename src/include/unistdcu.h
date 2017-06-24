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

//#pragma once
#ifndef _UNISTDCU_H
#define _UNISTDCU_H
#include <featurescu.h>

#include <_dirent.h>
#include <_unistd.h>
#include <sys/types.h>
typedef short gid_t;
typedef short uid_t;

#if defined(__CUDA_ARCH__) || defined(LIBCUFORCE)
#include <sentinel-unistdmsg.h>
__BEGIN_DECLS;

#undef access
#undef dup2
//#undef execve
//#undef ftruncate
#undef unlink
//#undef fileno
#undef getcwd
#undef chdir
//#undef isatty
#undef lseek
#undef sleep

/* Test for access to NAME using the real UID and real GID.  */
extern __device__ int access_device(const char *name, int type);
__forceinline __device__ int access_(const char *name, int type) { if (ISDEVICEPATH(name)) return access_device(name, type); unistd_access msg(name, type); return msg.RC; }
#define access access_

/* Move FD's file position to OFFSET bytes from the beginning of the file (if WHENCE is SEEK_SET),
the current position (if WHENCE is SEEK_CUR), or the end of the file (if WHENCE is SEEK_END).
Return the new file position.  */
#ifndef __USE_FILE_OFFSET64
extern __device__ off_t lseek_device(int fd, off_t offset, int whence);
__forceinline __device__ off_t lseek_(int fd, off_t offset, int whence) { if (ISDEVICEHANDLE(fd)) return lseek_device(fd, offset, whence); unistd_lseek msg(fd, offset, whence); return msg.RC; }
#define lseek lseek_
#else
#define lseek lseek64
#endif
#ifdef __USE_LARGEFILE64
extern __device__ off64_t lseek64_device(int fd, off64_t offset, int whence);
__forceinline __device__ off64_t lseek64_(int fd, off64_t offset, int whence) { if (ISDEVICEHANDLE(fd)) return lseek64_device(fd, offset, whence); unistd_lseek64 msg(fd, offset, whence); return msg.RC; }
#define lseek64 lseek64_
#endif

/* Close the file descriptor FD.  */
extern __device__ int close_device(int fd);
__forceinline __device__ int close_(int fd) { if (ISDEVICEHANDLE(fd)) return close_device(fd); unistd_close msg(fd); return msg.RC; }
#define close close_

/* Read NBYTES into BUF from FD.  Return the number read, -1 for errors or 0 for EOF.  */
extern __device__ size_t read_device(int fd, void *buf, size_t nbytes);
__forceinline __device__ size_t read_(int fd, void *buf, size_t nbytes, bool wait = true) { if (ISDEVICEHANDLE(fd)) return read_device(fd, buf, nbytes); unistd_read msg(wait, fd, buf, nbytes); return msg.RC; }
#define read read_

/* Write N bytes of BUF to FD.  Return the number written, or -1.  */
extern __device__ size_t write_device(int fd, const void *buf, size_t nbytes);
__forceinline __device__ size_t write_(int fd, void *buf, size_t nbytes, bool wait = true) { if (ISDEVICEHANDLE(fd)) return write_device(fd, buf, nbytes); unistd_write msg(wait, fd, buf, nbytes); return msg.RC; }
#define write write_

/* Create a one-way communication channel (pipe). If successful, two file descriptors are stored in PIPEDES;
bytes written on PIPEDES[1] can be read from PIPEDES[0]. Returns 0 if successful, -1 if not.  */
//nosupport: extern __device__ int pipe_(int pipedes[2]);
//#define pipe pipe_

/* Schedule an alarm.  In SECONDS seconds, the process will get a SIGALRM. If SECONDS is zero, any currently scheduled alarm will be cancelled.
The function returns the number of seconds remaining until the last alarm scheduled would have signaled, or zero if there wasn't one.
There is no return value to indicate an error, but you can set `errno' to 0 and check its value after calling `alarm', and this might tell you.
The signal may come late due to processor scheduling.  */
//nosupport: extern __device__ unsigned int alarm_(unsigned int seconds);
//#define alarm alarm_

/* Make the process sleep for SECONDS seconds, or until a signal arrives and is not ignored.  The function returns the number of seconds less
than SECONDS which it actually slept (thus zero if it slept the full time). If a signal handler does a `longjmp' or modifies the handling of the
SIGALRM signal while inside `sleep' call, the handling of the SIGALRM signal afterwards is undefined.  There is no return value to indicate
error, but if `sleep' returns SECONDS, it probably didn't work.  */
extern __device__ void usleep_(unsigned long milliseconds);
#define usleep usleep_
__forceinline __device__ void sleep_(unsigned int seconds) { usleep_(seconds * 1000); }
#define sleep sleep_

/* Suspend the process until a signal arrives. This always returns -1 and sets `errno' to EINTR.  */
//nosupport: extern __device__int pause_(void);
//#define pause pause_

/* Change the owner and group of FILE.  */
extern __device__ int chown_device(const char *file, uid_t owner, gid_t group);
__forceinline __device__ int chown_(const char *file, uid_t owner, gid_t group) { if (ISDEVICEPATH(file)) return chown_device(file, owner, group); __cwd[0] = 0; unistd_chown msg(file, owner, group); return msg.RC; }
#define chown chown_

/* Change the process's working directory to PATH.  */
extern __device__ int chdir_device(const char *path);
__forceinline __device__ int chdir_(const char *path) { if (ISDEVICEPATH(path)) return chdir_device(path); __cwd[0] = 0; unistd_chdir msg(path); return msg.RC; }
#define chdir chdir_

/* Get the pathname of the current working directory, and put it in SIZE bytes of BUF.  Returns NULL if the
directory couldn't be determined or SIZE was too small. If successful, returns BUF.  In GNU, if BUF is NULL,
an array is allocated with `malloc'; the array is SIZE bytes long, unless SIZE == 0, in which case it is as
big as necessary.  */
extern __device__ char *getcwd_device(char *buf, size_t size);
__forceinline __device__ char *getcwd_(char *buf, size_t size) { if (__cwd[0]) return getcwd_device(buf, size); unistd_getcwd msg(buf, size); return msg.RC; }
#define getcwd getcwd_

/* Duplicate FD, returning a new file descriptor on the same file.  */
extern __device__ int dup_device(int fd, int fd2, bool dup1);
__forceinline __device__ int dup_(int fd) { if (ISDEVICEHANDLE(fd)) return dup_device(fd, -1, true); unistd_dup msg(fd, -1, true); return msg.RC; }
#define dup dup_

/* Duplicate FD to FD2, closing FD2 and making it open on the same file.  */
__forceinline __device__ int dup2_(int fd, int fd2) { if (ISDEVICEHANDLE(fd)) return dup_device(fd, fd2, false); unistd_dup msg(fd, fd2, false); return msg.RC; }
#define dup2 dup2_

/* NULL-terminated array of "NAME=VALUE" environment variables.  */
extern __device__ char **__environ_;
#define __environ __environ_

/* Terminate program execution with the low-order 8 bits of STATUS.  */
//duplicate: extern __device__ void exit_(int status);
//#define exit exit_

/* Get file-specific configuration information about PATH.  */
//nosupport: extern __device__ long int pathconf_(const char *path, int name);
//#define pathconf pathconf_

/* Get file-specific configuration about descriptor FD.  */
//nosupport: extern __device__ long int fpathconf_(int fd, int name);
//#define fpathconf fpathconf_

/* Remove the link FILENAME.  */
extern __device__ int unlink_device(const char *filename);
__forceinline __device__ int unlink_(const char *filename) { if (ISDEVICEPATH(filename)) return unlink_device(filename); unistd_unlink msg(filename); return msg.RC; }
#define unlink unlink_

/* Remove the directory PATH.  */
extern __device__ int rmdir_device(const char *path);
__forceinline __device__ int rmdir_(const char *path) { if (ISDEVICEPATH(path)) return rmdir_device(path); unistd_rmdir msg(path); return msg.RC; }
#define rmdir rmdir_

__END_DECLS;
#else
#define __environ ((char **)nullptr)
#define chown(f,o,g) 0
//#define chgrp
#endif  /* __CUDA_ARCH__ */

#endif  /* _UNISTDCU_H */
