#include <cuda_runtime.h>
#include <stdiocu.h>
#include <direntcu.h>
#include <sys/statcu.h>
#include <unistdcu.h>
#include <assert.h>

static __device__ void testReading(DIR *d)
{
	//struct dirent *a0 = readdir(d); assert(a0);
	//bool b0 = !strcmp(a0->d_name, "dir0"); assert(!b0);
	//struct dirent *c0 = readdir(d); assert(!c0);
	//rewinddir(d);
	//struct dirent *d0 = readdir(d); assert(d0);
	//bool e0 = !strcmp(d0->d_name, "dir0"); assert(!e0);
}

#define HostDir "C:\\T_\\"
#define DeviceDir ":\\"
static __global__ void g_dirent_test1()
{
	printf("dirent_test1\n");

	///* Open a directory stream on NAME. Return a DIR stream on the directory, or NULL if it could not be opened. */
	///* Host Absolute */
	//DIR *a0a = opendir(HostDir"missing"); int a0b = closedir(a0a); assert(!a0a && a0b == -1);
	//mkdir(HostDir"test", 0); mkdir(HostDir"test\\dir0", 0);
	//DIR *a1a = opendir(HostDir"test"); bool a1b = !strcmp(a1a->ent.d_name, "test"); testReading(a0a); int a1c = closedir(a1a); assert(a1a && a1b && !a1c);

	///* Device Absolute */
	//DIR *b0a = opendir(DeviceDir":\\missing"); int b0b = closedir(b0a); assert(!b0a && b0b == -1);
	//mkdir(DeviceDir"test", 0); mkdir(DeviceDir"test\\dir0", 0);
	//DIR *b1a = opendir(DeviceDir"test"); bool b1b = !strcmp(b1a->ent.d_name, "test"); testReading(b0a); int b1c = closedir(b1a); assert(b1a && b1b && !b1c);

	///* Host Relative */
	//chdir(HostDir);
	//DIR *c0a = opendir("missing"); int c0b = closedir(c0a); assert(!c0a && c0b == -1);
	//mkdir("test", 0); mkdir("test\\dir0", 0);
	//DIR *c1a = opendir("test"); bool c1b = !strcmp(c1a->ent.d_name, "test"); testReading(c0a); int c1c = closedir(c1a); assert(c1a && c1b && !c1c);

	///* Device Relative */
	//chdir(DeviceDir);
	//DIR *d0a = opendir("missing"); int d0b = closedir(d0a); assert(!d0a && d0b == -1);
	//mkdir("test", 0); mkdir("test\\dir0", 0);
	//DIR *d1a = opendir("test"); bool d1b = !strcmp(d1a->ent.d_name, "test"); testReading(d0a); int d1c = closedir(d1a); assert(d1a && d1b && !d1c);

}
cudaError_t dirent_test1() { g_dirent_test1<<<1, 1>>>(); return cudaDeviceSynchronize(); }
