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

static __global__ void g_fcntl_test1()
{
	printf("fcntl_test1\n");

	////////// Open Files //////////
	/* Host Absolute */
	int a0a = open("C:\\missing.txt", O_RDONLY); int a0b = close(a0a); assert(!a0a && a0b == -1);
	makeAFile("C:\\test.txt");
	int a1a = open("C:\\test.txt", O_RDONLY); int a1b = close(a1a); assert(a1a && !a1b);

	/* Device Absolute */
	int b0a = open(":\\missing.txt", O_RDONLY); int b0b = close(b0a); assert(!b0a && b0b == -1);
	makeAFile(":\\test.txt");
	int b1a = open(":\\test.txt", O_RDONLY); int b1b = close(b1a); assert(b1a && !b1b);

	/* Host Relative */
	chdir("C:\\");
	int c0a = open("missing.txt", O_RDONLY); int c0b = close(c0a); assert(!c0a && c0b == -1);
	makeAFile("test.txt");
	int c1a = open("test.txt", O_RDONLY); int c1b = close(c1a); assert(c1a && !c1b);

	/* Device Relative */
	chdir(":\\");
	int d0a = open("missing.txt", O_RDONLY); int d0b = close(d0a); assert(!d0a && d0b == -1);
	makeAFile("test.txt");
	int d1a = open("test.txt", O_RDONLY); int d1b = close(d1a); assert(d1a && !d1b);

	////////// CREATE FILES //////////
	/* Host Absolute */
	int e0a = creat("C:\\missing.txt", O_RDONLY); int e0b = close(e0a); assert(!e0a && e0b == -1);
	makeAFile("C:\\test.txt");
	int e1a = creat("C:\\test.txt", O_RDONLY); int e1b = close(e1a); assert(e1a && !e1b);

	/* Device Absolute */
	int f0a = creat(":\\missing.txt", O_RDONLY); int f0b = close(f0a); assert(!f0a && f0b == -1);
	makeAFile(":\\test.txt");
	int f1a = creat(":\\test.txt", O_RDONLY); int f1b = close(f1a); assert(f1a && !f1b);

	/* Host Relative */
	chdir("C:\\");
	int g0a = creat("missing.txt", O_RDONLY); int g0b = close(g0a); assert(!g0a && g0b == -1);
	makeAFile("test.txt");
	int g1a = creat("test.txt", O_RDONLY); int g1b = close(g1a); assert(g1a && !g1b);

	/* Device Relative */
	chdir(":\\");
	int h0a = creat("missing.txt", O_RDONLY); int h0b = close(h0a); assert(!h0a && h0b == -1);
	makeAFile("test.txt");
	int h1a = creat("test.txt", O_RDONLY); int h1b = close(h1a); assert(h1a && !h1b);


	// TEST fcntl
}
cudaError_t fcntl_test1() { g_fcntl_test1<<<1, 1>>>(); return cudaDeviceSynchronize(); }
