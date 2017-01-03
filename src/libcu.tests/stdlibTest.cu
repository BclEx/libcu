#include <crtdefscu.h>
#include <stdiocu.h>
#include <stdlibcu.h>
#include <cuda_runtimecu.h>
#include <assert.h>

static __global__ void stdlib_test1()
{
	printf("stdlib_test1\n");
}

#pragma region qsort

/*
static __device__ int qsortSelectFiles(const struct dirent *dirbuf)
{
return dirbuf->d_name[0] == '.' ? 0 : 1;
}

static __global__ void stdlib_qsort()
{
struct dirent **a;
struct dirent *dirbuf;

int i, numdir;

chdir("/");
numdir = scandir(".", &a, qsortSelectFiles, NULL);
printf("\nGot %d entries from scandir().\n", numdir);
for (i = 0; i < numdir; ++i) {
dirbuf = a[i];
printf("[%d] %s\n", i, dirbuf->d_name);
free(a[i]);
}
free(a);
numdir = scandir(".", &a, qsortSelectFiles, alphasort);
printf("\nGot %d entries from scandir() using alphasort().\n", numdir);
for (i = 0; i < numdir; ++i) {
dirbuf = a[i];
printf("[%d] %s\n", i, dirbuf->d_name);
}
printf("\nCalling qsort()\n");
// Even though some manpages say that alphasort should be int alphasort(const void *a, const void *b),
// in reality glibc and uclibc have const struct dirent** instead of const void*.
// Therefore we get a warning here unless we use a cast, which makes people think that alphasort prototype needs to be fixed in uclibc headers.
qsort(a, numdir, sizeof(struct dirent *), (void *)alphasort);
for (i = 0; i < numdir; ++i) {
dirbuf = a[i];
printf("[%d] %s\n", i, dirbuf->d_name);
free(a[i]);
}
free(a);
return 0;
}
*/

#pragma endregion

#pragma region strtol

__constant__ const char *_strtol_strings[] = {
	/* some simple stuff */
	"0", "1", "10",
	"100", "1000", "10000", "100000", "1000000",
	"10000000", "100000000", "1000000000",

	/* negative */
	"-0", "-1", "-10",
	"-100", "-1000", "-10000", "-100000", "-1000000",
	"-10000000", "-100000000", "-1000000000",

	/* test base>10 */
	"a", "b", "f", "g", "z",

	/* test hex */
	"0x0", "0x1", "0xa", "0xf", "0x10",

	/* test octal */
	"00", "01", "07", "08", "0a", "010",

	/* other */
	"0x8000000",

	/* check overflow cases: (for 32 bit) */
	"2147483645",
	"2147483646",
	"2147483647",
	"2147483648",
	"2147483649",
	"-2147483645",
	"-2147483646",
	"-2147483647",
	"-2147483648",
	"-2147483649",
	"4294967293",
	"4294967294",
	"4294967295",
	"4294967296",
	"4294967297",
	"-4294967293",
	"-4294967294",
	"-4294967295",
	"-4294967296",
	"-4294967297",

	/* bad input tests */
	"",
	"00",
	"0x",
	"0x0",
	"-",
	"+",
	" ",
	" -",
	" - 0",
};
int _strtol_ntests = _LENGTHOF(_strtol_strings);

static __device__ void strtol_test(int base)
{
	int i;
	long n;
	char *endptr;
	for (i = 0; i < _strtol_ntests; i++) {
		n = strtol(_strtol_strings[i], &endptr, base);
		printf("strtol(\"%s\",%d) len=%lu res=%ld\n", _strtol_strings[i], base, (unsigned long)(endptr - _strtol_strings[i]), n);
	}
}

static void strtol_utest(int base)
{
	int i;
	unsigned long n;
	char *endptr;
	for (i = 0;i < _strtol_ntests; i++) {
		n = strtoul(_strtol_strings[i], &endptr, base);
		printf("strtoul(\"%s\",%d) len=%lu res=%lu\n", _strtol_strings[i], base, (unsigned long)(endptr - _strtol_strings[i]), n);
	}
}

static __global__ void stdlib_strtol()
{
	strtol_test(0); strtol_utest(0);
	strtol_test(8); strtol_utest(8);
	strtol_test(10); strtol_utest(10);
	strtol_test(16); strtol_utest(16);
	strtol_test(36); strtol_utest(36);
}

#pragma endregion

#pragma region strtoq

__constant__ const char *_strtoq_strings[] = {
	/* some simple stuff */
	"0", "1", "10",
	"100", "1000", "10000", "100000", "1000000",
	"10000000", "100000000", "1000000000",

	/* negative */
	"-0", "-1", "-10",
	"-100", "-1000", "-10000", "-100000", "-1000000",
	"-10000000", "-100000000", "-1000000000",

	/* test base>10 */
	"a", "b", "f", "g", "z",

	/* test hex */
	"0x0", "0x1", "0xa", "0xf", "0x10",

	/* test octal */
	"00", "01", "07", "08", "0a", "010",

	/* other */
	"0x8000000",

	/* check overflow cases: (for 32 bit) */
	"2147483645",
	"2147483646",
	"2147483647",
	"2147483648",
	"2147483649",
	"-2147483645",
	"-2147483646",
	"-2147483647",
	"-2147483648",
	"-2147483649",
	"4294967293",
	"4294967294",
	"4294967295",
	"4294967296",
	"4294967297",
	"-4294967293",
	"-4294967294",
	"-4294967295",
	"-4294967296",
	"-4294967297",

	/* bad input tests */
	"",
	"00",
	"0x",
	"0x0",
	"-",
	"+",
	" ",
	" -",
	" - 0",
};
int _strtoq_ntests = _LENGTHOF(_strtoq_strings);

void strtoq_test(int base)
{
	int i;
	quad_t n;
	char *endptr;
	for (i = 0; i < _strtoq_ntests; i++) {
		n = strtoq(_strtoq_strings[i], &endptr, base);
		printf("strtoq(\"%s\",%d) len=%lu res=%qd\n", _strtoq_strings[i], base, (unsigned long)(endptr - _strtoq_strings[i]), n);
	}
}

static __global__ void stdlib_strtoq()
{
	strtoq_test(0);
	strtoq_test(8);
	strtoq_test(10);
	strtoq_test(16);
	strtoq_test(36);
}

#pragma endregion

void stdlib_()
{
	stdlib_test1<<<1, 1>>>();
}