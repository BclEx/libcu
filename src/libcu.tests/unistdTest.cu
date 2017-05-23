#include <cuda_runtime.h>
#include <stdiocu.h>
#include <unistdcu.h>
#include <assert.h>

static __global__ void g_unistd_test1()
{
	printf("unistd_test1\n");
}
cudaError_t unistd_test1() { g_unistd_test1<<<1, 1>>>(); return cudaDeviceSynchronize(); }


//__forceinline __device__ int access_(const char *name, int type) { if (ISDEVICEPATH(name)) return access_device(name, type); unistd_access msg(name, type); return msg.RC; }
//__forceinline __device__ off_t lseek_(int fd, off_t offset, int whence) { if (ISDEVICEHANDLE(fd)) return lseek_device(fd, offset, whence); unistd_lseek msg(fd, offset, whence); return msg.RC; }
//__forceinline __device__ int close_(int fd) { if (ISDEVICEHANDLE(fd)) return close_device(fd); unistd_close msg(fd); return msg.RC; }
//__forceinline __device__ size_t read_(int fd, void *buf, size_t nbytes, bool wait = true) { if (ISDEVICEHANDLE(fd)) return read_device(fd, buf, nbytes); unistd_read msg(wait, fd, buf, nbytes); return msg.RC; }
//__forceinline __device__ size_t write_(int fd, void *buf, size_t nbytes, bool wait = true) { if (ISDEVICEHANDLE(fd)) return write_device(fd, buf, nbytes); unistd_write msg(wait, fd, buf, nbytes); return msg.RC; }
////nosupport: extern __device__ int pipe_(int pipedes[2]);
////nosupport: extern __device__ unsigned int alarm_(unsigned int seconds);
//extern __device__ void usleep_(unsigned long milliseconds);
//__device__ __forceinline void sleep_(unsigned int seconds) { usleep_(seconds * 1000); }
////nosupport: extern int pause_(void);
//__forceinline __device__ int chown_(const char *file, uid_t owner, gid_t group) { if (ISDEVICEPATH(file)) return chown_device(file, owner, group); __cwd[0] = 0; unistd_chown msg(file, owner, group); return msg.RC; }
//__forceinline __device__ int chdir_(const char *path) { if (ISDEVICEPATH(path)) return chdir_device(path); __cwd[0] = 0; unistd_chdir msg(path); return msg.RC; }
//__forceinline __device__ char *getcwd_(char *buf, size_t size) { if (__cwd[0]) return getcwd_device(buf, size); unistd_getcwd msg(buf, size); return msg.RC; }
//__forceinline __device__ int dup_(int fd) { if (ISDEVICEHANDLE(fd)) return dup_device(fd, -1, true); unistd_dup msg(fd, -1, true); return msg.RC; }
//__forceinline __device__ int dup2_(int fd, int fd2) { if (ISDEVICEHANDLE(fd)) return dup_device(fd, fd2, false); unistd_dup msg(fd, fd2, false); return msg.RC; }
//extern __device__ char **__environ_;
////nosupport: extern __device__ void exit_(int status);
////nosupport: extern __device__ long int pathconf_(const char *path, int name);
////nosupport: extern __device__ long int fpathconf_(int fd, int name);
//__forceinline __device__ int unlink_(const char *filename) { if (ISDEVICEPATH(filename)) return unlink_device(filename); unistd_unlink msg(filename); return msg.RC; }
//__forceinline __device__ int rmdir_(const char *path) { if (ISDEVICEPATH(path)) return rmdir_device(path); unistd_rmdir msg(path); return msg.RC; }
