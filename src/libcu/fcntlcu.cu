#include "fsystem.h"
#include <sys/statcu.h>
#include <stdargcu.h>
#include <fcntlcu.h>
#include <sentinel-fcntlmsg.h>
#include <sentinel-unistdmsg.h>

__BEGIN_DECLS;

__device__ int fcntlv_(int fd, int cmd, va_list va)
{
#ifdef __CUDA_ARCH__
	if (ISHOSTHANDLE(fd)) { fcntl_fcntl msg(fd, cmd, va.i?va_arg(va, int):0); return msg.RC; }
#endif
	panic("Not Implemented");
	// (int fd, unsigned int cmd, unsigned long arg, struct file *filp)
	//	long err = -EINVAL;
	//	switch (cmd) {
	//	case F_DUPFD: err = f_dupfd(arg, filp, 0); break;
	//	case F_GETFD: err = get_close_on_exec(fd) ? FD_CLOEXEC : 0; break;
	//	case F_SETFD: err = 0; set_close_on_exec(fd, arg & FD_CLOEXEC); break;
	//	case F_GETFL: err = filp->f_flags; break;
	//	case F_SETFL: err = setfl(fd, filp, arg); break;
	//	case F_GETOWN: err = f_getown(filp); force_successful_syscall_return(); break;
	//	case F_SETOWN: f_setown(filp, arg, 1); err = 0; break;
	//	default:
	//		break;
	//	}
	//	return err;
	return 0;
}
#ifdef __USE_LARGEFILE64
__device__ int fcntl64v_(int fd, int cmd, va_list va)
{
#ifdef __CUDA_ARCH__
	if (ISHOSTHANDLE(fd)) { fcntl_fcntl msg(fd, cmd, va.i?va_arg(va, int):0); return msg.RC; }
#endif
	panic("Not Implemented");
}
#endif

__device__ int openv_(const char *file, int oflag, va_list va)
{
#ifdef __CUDA_ARCH__
	if (ISHOSTPATH(file)) { fcntl_open msg(file, oflag, va.i?va_arg(va, int):0); return msg.RC; }
#endif
	int fd; fsystemOpen(file, oflag, &fd); return fd;
}
#ifdef __USE_LARGEFILE64
__device__ int openv64_(const char *file, int oflag, va_list va)
{
#ifdef __CUDA_ARCH__
	if (ISHOSTPATH(file)) { fcntl_open msg(file, oflag, va.i?va_arg(va, int):0); return msg.RC; }
#endif
	int fd; fsystemOpen(file, oflag, &fd); return fd;
}
#endif

/* Close the file descriptor FD.  */
__device__ int close_(int fd)
{
	if (ISHOSTHANDLE(fd)) { unistd_close msg(fd); return msg.RC; }
	fsystemClose(fd);
	return 0;
}

__END_DECLS;
