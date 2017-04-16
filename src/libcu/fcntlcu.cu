#include <stddefcu.h>
#include <stdargcu.h>
#include <fcntlcu.h>
#include "fsystem.h"

__BEGIN_DECLS;

__device__ int fcntlv_device(int fd, int cmd, va_list va)
{
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

__device__ int openv_device(const char *file, int oflag, va_list va)
{
	int fd; int r; fsystemOpen(file, oflag, &fd, &r);
	return fd;
}

/* Close the file descriptor FD.  */
__device__ int close_device(int fd)
{
	fileFree(fd);
	return 0;
}

__END_DECLS;
