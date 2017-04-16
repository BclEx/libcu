/*
stat.h - File Characteristics
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
#ifndef _SYS_STATCU_H
#define	_SYS_STATCU_H
#include <featurescu.h>

#include <sys/stat.h>
//#define S_ISREG(m) (((m) & _S_IFMT) == _S_IFREG)
//#define S_ISDIR(m) (((m) & _S_IFMT) == _S_IFDIR)

//#include <bits/libcu_stat.h>
typedef int mode_t;
//
///* Test macros for file types.	*/
//#define	__S_ISTYPE(mode, mask)	(((mode) & __S_IFMT) == (mask))
//
//#define	S_ISDIR(mode)	 __S_ISTYPE((mode), __S_IFDIR)
//#define	S_ISCHR(mode)	 __S_ISTYPE((mode), __S_IFCHR)
//#define	S_ISBLK(mode)	 __S_ISTYPE((mode), __S_IFBLK)
//#define	S_ISREG(mode)	 __S_ISTYPE((mode), __S_IFREG)
//
///* Protection bits.  */
//#define	S_ISUID __S_ISUID	/* Set user ID on execution.  */
//#define	S_ISGID	__S_ISGID	/* Set group ID on execution.  */
//
//#define	S_IRUSR	__S_IREAD	/* Read by owner.  */
//#define	S_IWUSR	__S_IWRITE	/* Write by owner.  */
//#define	S_IXUSR	__S_IEXEC	/* Execute by owner.  */
///* Read, write, and execute by owner.  */
//#define	S_IRWXU	(__S_IREAD|__S_IWRITE|__S_IEXEC)
//
//#define	S_IRGRP	(S_IRUSR >> 3)	/* Read by group.  */
//#define	S_IWGRP	(S_IWUSR >> 3)	/* Write by group.  */
//#define	S_IXGRP	(S_IXUSR >> 3)	/* Execute by group.  */
///* Read, write, and execute by group.  */
//#define	S_IRWXG	(S_IRWXU >> 3)
//
//#define	S_IROTH	(S_IRGRP >> 3)	/* Read by others.  */
//#define	S_IWOTH	(S_IWGRP >> 3)	/* Write by others.  */
//#define	S_IXOTH	(S_IXGRP >> 3)	/* Execute by others.  */
///* Read, write, and execute by others.  */
//#define	S_IRWXO	(S_IRWXG >> 3)

#if defined(__CUDA_ARCH__) || defined(LIBCUFORCE)
__BEGIN_DECLS;

__END_DECLS;
#include <sentinel-fcntlmsg.h>
__BEGIN_DECLS;

#ifndef __USE_FILE_OFFSET64
/* Get file attributes about FILE and put them in BUF. If FILE is a symbolic link, do not follow it.  */
extern __device__ int stat_device(const char *__restrict file, struct stat *__restrict buf, bool lstat = false);
__forceinline __device__ int stat_(const char *__restrict file, struct stat *__restrict buf, bool lstat = false) { if (ISDEVICEPATH(file)) return stat_device(file, buf, lstat); fcntl_stat msg(file, buf, lstat); return msg.RC; }
#define stat(file, buf) stat_(file, buf, false)
#define lstat(file, buf) stat_(file, buf, true)
/* Get file attributes for the file, device, pipe, or socket that file descriptor FD is open on and put them in BUF.  */
extern __device__ int fstat_device(int fd, struct stat *buf);
__forceinline __device__ int fstat_(int fd, struct stat *buf) { if (ISDEVICEHANDLE(fd)) return fstat_device(fd, buf); fcntl_fstat msg(fd, buf); return msg.RC; }
#define fstat(fd, buf) fstat_(fd, buf)
#else
#define stat(file, buf) stat64_(file, buf)
#define lstat(file, buf) lstat64_(file, buf)
#define fstat(fd, buf) fstat64_device(fd, buf)
#endif
#ifdef __USE_LARGEFILE64
extern __device__ int stat64_device(const char *__restrict file, struct stat64 *__restrict buf, bool lstat = false);
__forceinline __device__ int stat64_(const char *__restrict file, struct stat64 *__restrict buf, bool lstat = false) { if (ISDEVICEPATH(file)) return stat64_device(file, buf, lstat); fcntl64_stat msg(file, buf, lstat); return msg.RC; }
#define stat64(file, buf) stat64_(file, buf, false)
#define lstat64(file, buf) stat64_(file, buf, true)
extern __device__ int fstat64_device(int fd, struct stat64 *buf);
__forceinline __device__ int fstat64_(int fd, struct stat64 *buf) { if (ISDEVICEHANDLE(fd)) return fstat64_device(fd, buf); fcntl_fstat msg(fd, buf); return msg.RC; }
#define fstat64(fd, buf) fstat64_(fd, buf)
#endif

/* Set file access permissions for FILE to MODE. If FILE is a symbolic link, this affects its target instead.  */
extern __device__ int chmod_device(const char *file, mode_t mode);
__forceinline __device__ int chmod_(const char *file, mode_t mode) { if (ISDEVICEPATH(file)) return chmod_device(file, mode); fcntl_chmod msg(file, mode); return msg.RC; }
#define chmod chmod_

/* Set the file creation mask of the current process to MASK, and return the old creation mask.  */
extern __device__ mode_t umask_(mode_t mask);
#define umask umask_

/* Create a new directory named PATH, with permission bits MODE.  */
extern __device__ int mkdir_device(const char *path, mode_t mode);
__forceinline __device__ int mkdir_(const char *path, mode_t mode) { if (ISDEVICEPATH(path)) return mkdir_device(path, mode); fcntl_mkdir msg(path, mode); return msg.RC; }
#define mkdir mkdir_

/* Create a new FIFO named PATH, with permission bits MODE.  */
extern __device__ int mkfifo_device(const char *path, mode_t mode);
__forceinline __device__ int mkfifo_(const char *path, mode_t mode) { if (ISDEVICEPATH(path)) return mkfifo_device(path, mode); fcntl_mkfifo msg(path, mode); return msg.RC; }
#define mkfifo mkfifo_

__END_DECLS;
#else
#include <direct.h>
#define mkdir(path, mode) _mkdir(path) 
#define lstat stat
#endif /* __CUDA_ARCH__  */

#endif /* _SYS_STATCU_H  */
