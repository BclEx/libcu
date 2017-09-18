#include <cuda_runtimecu.h>
#include <stdiocu.h>
#include <stringcu.h>
#include <assert.h>

static __global__ void g_string_test1()
{
	printf("string_test1\n");

	char src[50] = "abcdefghijklmnopqrstuvwxyz";
	char dest[50]; memset(dest, 0, 50);

	//// MEMCPY, MEMMOVE, MEMSET, MEMCPY, MEMCHR ////
	//__forceinline __device__ void *memcpy_(void *__restrict dest, const void *__restrict src, size_t n);
	//extern __device__ void *memmove_(void *dest, const void *src, size_t n);
	//__forceinline __device__ void *memset_(void *s, int c, size_t n);
	//extern __device__ int memcmp_(const void *s1, const void *s2, size_t n);
	//extern __device__ void *memchr_(const void *s, int c, size_t n);
	void *a0a = memcpy(dest, src, 0); void *a0b = memcpy(dest+1, src, 1); assert(a0a == &dest[0] && ((char *)a0a)[0] == 0 && a0b == &dest[1] && ((char *)a0b)[0] == 'a');
	void *a1a = memmove(dest, src, 0); void *a1b = memmove(dest, src, 10); void *a1c = memmove(dest+5, dest, 5); assert(a1a == dest && !strncmp((char *)a1b, "abcde", 5) && !strncmp((char *)a1c, "abcde", 5));
	void *a2a = memset(dest, 0, 0); void *a2b = memset(dest+1, 0, 5); assert(a2a == dest && ((char *)a2a)[0] == 'a' && a2b == dest+1 && ((char *)a2b)[0] == 0);
	int a3a = memcmp(nullptr, nullptr, 0); int a3b = memcmp("abc", "abc", 2); int a3c = memcmp("abc", "abc", 10); int a3d = memcmp("abc", "axc", 10); assert(!a3a && !a3b && !a3c && a3d);
	void *a4a = memchr(src, 0, 0); void *a4b = memchr(src, 'b', 1); void *a4c = memchr(src, 'b', 3); void *a4d = memchr(src, 'z', 3); assert(!a4a && !a4b && a4c && !a4d);

	//// STRCPY, STRNCPY, STRCAT, STRNCAT ////
	//extern __device__ char *strcpy_(char *__restrict dest, const char *__restrict src);
	//extern __device__ char *strncpy_(char *__restrict dest, const char *__restrict src, size_t n);
	//extern __device__ char *strcat_(char *__restrict dest, const char *__restrict src);
	//extern __device__ char *strncat_(char *__restrict dest, const char *__restrict src, size_t n);
	strcpy(src, "abcd");
	char *b0a = strcpy(dest, src); assert(!strcmp(b0a, "abcd"));
	char *b1a = strncpy(dest+5, src, 10); assert(!strcmp(b1a, "abcd"));
	char *b2a = strcat(dest+10, "xyz"); assert(!strcmp(b2a, "xyz"));
	char *b3a = strncat(dest+15, "xyz", 2); assert(!strncmp(b3a, "xy", 2));

	//// STRCMP, STRICMP, STRNCMP, STRNICMP ///2
	//extern __device__ int strcmp_(const char *s1, const char *s2);
	//extern __device__ int stricmp_(const char *s1, const char *s2);
	//extern __device__ int strncmp_(const char *s1, const char *s2, size_t n);
	//extern __device__ int strnicmp_(const char *s1, const char *s2, size_t n);

	//// STRCOLL ////
	//extern __device__ int strcoll_(const char *s1, const char *s2);

	//// STRXFRM ////
	//extern __device__ size_t strxfrm_(char *__restrict dest, const char *__restrict src, size_t n);

	//// STRDUP, STRNDUP ////
	//extern __device__ char *strdup_(const char *s);
	//extern __device__ char *strndup_(const char *s, size_t n);

	//// STRCHR, STRRCHR, STRCSPN, STRSPN, STRPBRK, STRSTR, STRTOK ////
	//extern __device__ char *strchr_(const char *s, int c);
	//extern __device__ char *strrchr_(const char *s, int c);
	//extern __device__ size_t strcspn_(const char *s, const char *reject);
	//extern __device__ size_t strspn_(const char *s, const char *accept);
	//extern __device__ char *strpbrk_(const char *s, const char *accept);
	//extern __device__ char *strstr_(const char *haystack, const char *needle);
	//extern __device__ char *strtok_(char *__restrict s, const char *__restrict delim);

	//// MEMPCPY ////
	//extern __device__ void *mempcpy_(void *__restrict dest, const void *__restrict src, size_t n);

	//// STRLEN, STRLEN16, STRNLEN ////
	//extern __device__ size_t strlen_(const char *s);
	//__forceinline __device__ size_t strlen16(const void *s);
	//extern __device__ size_t strnlen_(const char *s, size_t maxlen);

	//// STRERROR ////
	//extern __device__ char *strerror_(int errnum);

}
cudaError_t string_test1() { g_string_test1<<<1, 1>>>(); return cudaDeviceSynchronize(); }
