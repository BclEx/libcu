#include <cuda_runtime.h>
#include <stdiocu.h>
#include <fcntlcu.h>
#include <unistdcu.h>
#include <assert.h>

static __device__ void makeAFile(char *file)
{
	FILE *fp = fopen(file, "w");
	fprintf_(fp, "test");
	fclose(fp);
}

#define HostDir "C:\\T_\\"
#define DeviceDir ":\\"
static __global__ void g_fcntl_test1()
{
	printf("fcntl_test1\n");

	////////// Open Files //////////
	/* Host Absolute */
	int a0a = open(HostDir"missing.txt", O_RDONLY); assert(a0a < 0);
	makeAFile(HostDir"test.txt");
	int a1a = open(HostDir"test.txt", O_RDONLY); int a1b = close(a1a); assert(a1a && !a1b);

	/* Device Absolute */
	int b0a = open(DeviceDir"missing.txt", O_RDONLY); assert(b0a < 0);
	makeAFile(DeviceDir"test.txt");
	int b1a = open(DeviceDir"test.txt", O_RDONLY); int b1b = close(b1a); assert(b1a && !b1b);

	/* Host Relative */
	chdir(HostDir);
	int c0a = open("missing.txt", O_RDONLY); assert(c0a < 0);
	makeAFile("test.txt");
	int c1a = open("test.txt", O_RDONLY); int c1b = close(c1a); assert(c1a && !c1b);

	/* Device Relative */
	chdir(DeviceDir);
	int d0a = open("missing.txt", O_RDONLY); assert(d0a < 0);
	makeAFile("test.txt");
	int d1a = open("test.txt", O_RDONLY); int d1b = close(d1a); assert(d1a && !d1b);

	////////// CREATE FILES //////////
	/* Host Absolute */
	int e0a = creat(HostDir"missing.txt", O_RDONLY); assert(e0a < 0);
	makeAFile(HostDir"test.txt");
	int e1a = creat(HostDir"test.txt", O_RDONLY); int e1b = close(e1a); assert(e1a && !e1b);

	/* Device Absolute */
	int f0a = creat(DeviceDir"missing.txt", O_RDONLY); assert(f0a < 0);
	makeAFile(DeviceDir"test.txt");
	int f1a = creat(DeviceDir"test.txt", O_RDONLY); int f1b = close(f1a); assert(f1a && !f1b);

	/* Host Relative */
	chdir(HostDir);
	int g0a = creat("missing.txt", O_RDONLY); assert(g0a < 0);
	makeAFile("test.txt");
	int g1a = creat("test.txt", O_RDONLY); int g1b = close(g1a); assert(g1a && !g1b);

	/* Device Relative */
	chdir(DeviceDir);
	int h0a = creat("missing.txt", O_RDONLY); assert(h0a < 0);
	makeAFile("test.txt");
	int h1a = creat("test.txt", O_RDONLY); int h1b = close(h1a); assert(h1a && !h1b);

	// TEST fcntl
}
cudaError_t fcntl_test1() { g_fcntl_test1<<<1, 1>>>(); return cudaDeviceSynchronize(); }
