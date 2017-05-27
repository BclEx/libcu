#include <cuda_runtime.h>
#include <stdiocu.h>
#include <unistdcu.h>
#include <assert.h>

static __global__ void g_unistd_test1()
{
	printf("unistd_test1\n");

	//// ACCESS, LSEEK, CLOSE ////
	//__forceinline __device__ int access_(const char *name, int type); #sentinel-branch
	//__forceinline __device__ off_t lseek_(int fd, off_t offset, int whence); #sentinel-branch
	//__forceinline __device__ int close_(int fd); #sentinel-branch

	//// READ, WRITE ////
	//__forceinline __device__ size_t read_(int fd, void *buf, size_t nbytes, bool wait = true); #sentinel-branch
	//__forceinline __device__ size_t write_(int fd, void *buf, size_t nbytes, bool wait = true); #sentinel-branch

	//// PIPE, ALARM ////
	////nosupport: extern __device__ int pipe_(int pipedes[2]); #notsupported
	////nosupport: extern __device__ unsigned int alarm_(unsigned int seconds); #notsupported

	//// USLEEP, SLEEP, PAUSE ////
	//extern __device__ void usleep_(unsigned long milliseconds);
	//__device__ __forceinline void sleep_(unsigned int seconds) { usleep_(seconds * 1000); }
	////nosupport: extern int pause_(void); #notsupported

	//// CHOWN ////
	//__forceinline __device__ int chown_(const char *file, uid_t owner, gid_t group); #sentinel-branch

	//// CHDIR, GETCWD ////
	//__forceinline __device__ int chdir_(const char *path); #sentinel-branch
	//__forceinline __device__ char *getcwd_(char *buf, size_t size); #sentinel-branch

	//// DUP, DUP2 ////
	//__forceinline __device__ int dup_(int fd); #sentinel-branch
	//__forceinline __device__ int dup2_(int fd, int fd2); #sentinel-branch

	//// EXIT ////
	//extern __device__ char **__environ_;
	////nosupport: extern __device__ void exit_(int status); #notsupported

	//// PATHCONF, FPATHCONF ////
	////nosupport: extern __device__ long int pathconf_(const char *path, int name);
	////nosupport: extern __device__ long int fpathconf_(int fd, int name);

	//// UNLINK ////
	//__forceinline __device__ int unlink_(const char *filename); #sentinel-branch

	//// RMDIR ////
	//__forceinline __device__ int rmdir_(const char *path); #sentinel-branch

}
cudaError_t unistd_test1() { g_unistd_test1<<<1, 1>>>(); return cudaDeviceSynchronize(); }
