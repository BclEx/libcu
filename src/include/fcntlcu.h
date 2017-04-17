/*
fcntl.h - File control
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
#ifndef _FCNTLCU_H
#define _FCNTLCU_H
#include <featurescu.h>
#include <stdargcu.h>
#include <sys/statcu.h>

#include <fcntl.h>
#if defined(__CUDA_ARCH__) || defined(LIBCUFORCE)
#include <io.h>
#include <sentinel-fcntlmsg.h>
__BEGIN_DECLS;

/* Do the file control operation described by CMD on FD. The remaining arguments are interpreted depending on CMD. */
#ifndef __USE_FILE_OFFSET64
extern __device__ int fcntlv_device(int fd, int cmd, va_list va);
__forceinline __device__ int fcntlv_(int fd, int cmd, va_list va) { if (ISDEVICEHANDLE(fd)) return fcntlv_device(fd, cmd, va); fcntl_fcntl msg(fd, cmd, va.i?va_arg(va, int):0); return msg.RC; }
#define fcntl fcntl_
#else
#define fcntl fcntl64_
#endif
#ifdef __USE_LARGEFILE64
extern __device__ int fcntl64v_device(int fd, int cmd, va_list va);
__forceinline __device__ int fcntl64v_(int fd, int cmd, va_list va) { if (ISDEVICEHANDLE(fd)) return fcntl64v_device(fd, cmd, va); fcntl_fcntl msg(fd, cmd, va.i?va_arg(va, int):0); return msg.RC; }
#define fcntl64 fcntl64_
#endif

/* Open FILE and return a new file descriptor for it, or -1 on error. OFLAG determines the type of access used.  If O_CREAT is on OFLAG,
   the third argument is taken as a `mode_t', the mode of the created file. */
#ifndef __USE_FILE_OFFSET64
extern __device__ int openv_device(const char *file, int oflag, va_list va);
__forceinline __device__ int openv_(const char *file, int oflag, va_list va) { if (ISDEVICEPATH(file)) return openv_device(file, oflag, va); fcntl_open msg(file, oflag, va.i?va_arg(va, int):0); return msg.RC; }
#define open open_
#else
#define open open64_
#endif
#ifdef __USE_LARGEFILE64
extern __device__ int open64v_device(const char *file, int oflag, va_list va);
__forceinline __device__ int openv_(const char *file, int oflag, va_list va) { if (ISDEVICEPATH(file)) return openv_device(file, oflag, va); fcntl_open msg(file, oflag, va.i?va_arg(va, int):0); return msg.RC; }
#define open64 open64_
#endif

/* Create and open FILE, with mode MODE.  This takes an `int' MODE argument because that is what `mode_t' will be widened to.*/
#define creat(file, mode) open_(file, O_WRONLY|O_CREAT|O_TRUNC, mode)
#ifdef __USE_LARGEFILE64
#define creat64(file, mode) open64_(file, O_WRONLY|O_CREAT|O_TRUNC, mode)
#endif

__END_DECLS;

#ifndef __USE_FILE_OFFSET64
STDARG(int, fcntl_, fcntlv_(fd, cmd, va), int fd, int cmd);
STDARG(int, open_, openv_(file, oflag, va), const char *file, int oflag);
#endif
#ifdef __USE_LARGEFILE64
STDARG(int, fcntl64_, fcntl64v_(fd, cmd, va), int fd, int cmd);
STDARG(int, open64_, open64v_(file, oflag, va), const char *file, int oflag);
#endif

#else
#include <io.h>
#endif  /* __CUDA_ARCH__ */

#endif  /* _FCNTLCU_H */
