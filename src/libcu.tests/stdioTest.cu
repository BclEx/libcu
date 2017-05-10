#include <cuda_runtime.h>
#include <stdiocu.h>
#include <assert.h>

#ifndef MAKEAFILE
#define MAKEAFILE
static __device__ void makeAFile(char *file)
{
	FILE *fp = fopen(file, "w");
	fprintf_(fp, "test");
	fclose(fp);
}
#endif


static __global__ void g_stdio_test1()
{
	printf("stdio_test1\n");

	//bool f0a = ISDEVICEFILE(stdio); bool f0b = ISDEVICEFILE(stdio+2); bool f0c = ISDEVICEFILE(stdio-1); assert(!f0a && f0b && f0c);

	////////// REMOVE FILE //////////
	/* Host Absolute */
	int a0a = remove(HostDir"missing.txt"); assert(a0a < 0);
	makeAFile(HostDir"test.txt");
	int a1a = remove(HostDir"test.txt"); assert(a1a);

	/* Device Absolute */
	int b0a = remove(DeviceDir"missing.txt"); assert(b0a < 0);
	makeAFile(DeviceDir"test.txt");
	int b1a = remove(DeviceDir"test.txt"); assert(b1a);

	/* Host Relative */
	chdir(HostDir);
	int c0a = remove("missing.txt"); assert(c0a < 0);
	makeAFile("test.txt");
	int c1a = remove("test.txt"); assert(c1a);

	/* Device Relative */
	chdir(DeviceDir);
	int d0a = remove("missing.txt"); assert(d0a < 0);
	makeAFile("test.txt");
	int d1a = remove("test.txt"); assert(d1a);

	////////// RENAME FILE //////////
	/* Host Absolute */
	int e0a = rename(HostDir"missing.txt", "missing.txt"); assert(e0a < 0);
	makeAFile(HostDir"test.txt");
	int e1a = rename(HostDir"test.txt", "test.txt"); assert(e1a);

	/* Device Absolute */
	int f0a = rename(DeviceDir"missing.txt", "missing.txt"); assert(f0a < 0);
	makeAFile(DeviceDir"test.txt");
	int f1a = rename(DeviceDir"test.txt", "test.txt"); assert(f1a);

	/* Host Relative */
	chdir(HostDir);
	int g0a = rename("missing.txt", "missing.txt"); assert(g0a < 0);
	makeAFile("test.txt");
	int g1a = rename("test.txt", "test.txt"); assert(g1a);

	/* Device Relative */
	chdir(DeviceDir);
	int h0a = rename("missing.txt", "missing.txt"); assert(h0a < 0);
	makeAFile("test.txt");
	int h1a = rename("test.txt", "test.txt"); assert(h1a);

	//// TMPFILE
	FILE *i0a = tmpfile();

	//// FCLOSE, FFLUSH, FREOPEN, FOPEN
	char buf[100];
	/* Host Absolute */
	FILE *j0a = fopen(HostDir"missing.txt", "r"); assert(j0a < 0);
	makeAFile(HostDir"test.txt");
	FILE *j1a = fopen(HostDir"test.txt", "r"); int j1b = fread(buf, 4, 1, j1a); FILE *j1c = freopen(HostDir"test.txt", "r", j1a); int j1d = fread(buf, 4, 1, j1c); int j1e = fclose(j1c); assert(j1a);
	FILE *j2a = fopen(HostDir"test.txt", "w"); int j2b = fprintf_(j2a, "test"); FILE *j2c = freopen(HostDir"test.txt", "w", j2a); int j2d = fprintf_(j2c, "test"); int j2e = fflush(j1c); int j2e = fclose(j2c); assert(j2a);

	/* Device Absolute */
	FILE *k0a = fopen(DeviceDir"missing.txt", "r"); assert(k0a < 0);
	makeAFile(DeviceDir"test.txt");
	FILE *k1a = fopen(DeviceDir"test.txt", "r"); int k1b = fread(buf, 4, 1, k1a); FILE *k1c = freopen(DeviceDir"test.txt", "r", k1a); int k1d = fread(buf, 4, 1, k1c); int k1e = fclose(k1c); assert(k1a);
	FILE *k2a = fopen(DeviceDir"test.txt", "w"); int k2b = fprintf_(k2a, "test"); FILE *k2c = freopen(DeviceDir"test.txt", "w", k2a); int k2d = fprintf_(k2c, "test"); int k2e = fflush(k1c); int k2e = fclose(k2c); assert(k2a);

	/* Host Relative */
	chdir(HostDir);
	FILE *l0a = fopen("missing.txt", "r"); assert(l0a < 0);
	makeAFile("test.txt");
	FILE *l1a = fopen("test.txt", "r"); int l1b = fread(buf, 4, 1, l1a); FILE *l1c = freopen("test.txt", "r", l1a); int l1d = fread(buf, 4, 1, l1c); int l1e = fclose(l1c); assert(l1a);
	FILE *l2a = fopen("test.txt", "w"); int l2b = fprintf_(l2a, "test"); FILE *l2c = freopen("test.txt", "w", l2a); int l2d = fprintf_(l2c, "test"); int l2e = fflush(l1c); int l2e = fclose(l2c); assert(l2a);

	/* Device Relative */
	chdir(DeviceDir);
	FILE *m0a = fopen("missing.txt", "r"); assert(m0a < 0);
	makeAFile("test.txt");
	FILE *m1a = fopen("test.txt", "r"); int m1b = fread(buf, 4, 1, m1a); FILE *m1c = freopen("test.txt", "r", m1a); int m1d = fread(buf, 4, 1, m1c); int m1e = fclose(m1c); assert(m1a);
	FILE *m2a = fopen("test.txt", "w"); int m2b = fprintf_(m2a, "test"); FILE *m2c = freopen("test.txt", "w", m2a); int m2d = fprintf_(m2c, "test"); int m2e = fflush(m1c); int m2e = fclose(m2c); assert(m2a);
}
cudaError_t stdio_test1() { g_stdio_test1<<<1, 1>>>(); return cudaDeviceSynchronize(); }

static __global__ void g_stdio_64bit()
{
	printf("stdio_64bit\n");
	/*
	unsigned long long val = -1;
	void *ptr = (void *)-1;
	printf("%p\n", ptr);

	sscanf("123456789", "%Lx", &val);
	printf("val = %Lx\n", val);
	*/
}
cudaError_t stdio_64bit() { g_stdio_64bit<<<1, 1>>>(); return cudaDeviceSynchronize(); }

static __global__ void g_stdio_scanf()
{
	printf("stdio_scanf\n");
	/*
	const char *buf = "hello world";
	char *ps = NULL, *pc = NULL;
	char s[6], c;

	/ Check that %[...]/%c work. /
	sscanf(buf, "%[a-z] %c", s, &c);
	/ Check that %m[...]/%mc work. /
	sscanf(buf, "%m[a-z] %mc", &ps, &pc);

	if (strcmp(ps, "hello") != 0 || *pc != 'w' || strcmp(s, "hello") != 0 || c != 'w')
	return 1;

	free(ps);
	free(pc);

	return 0;
	*/
}
cudaError_t stdio_scanf() { g_stdio_scanf<<<1, 1>>>(); return cudaDeviceSynchronize(); }