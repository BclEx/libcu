#include <cuda_runtimecu.h>
#include <stdiocu.h>
#include <stringcu.h>
#include <assert.h>

static __global__ void g_string_test1()
{
	printf("string_test1\n");
}
cudaError_t string_test1() { g_string_test1<<<1, 1>>>(); return cudaDeviceSynchronize(); }


////builtin: extern void *__cdecl memset(void *, int, size_t);
////builtin: extern void *__cdecl memcpy(void *, const void *, size_t);
//__device__ __forceinline void *memcpy_(void *__restrict dest, const void *__restrict src, size_t n) { return (n ? memcpy(dest, src, n) : nullptr); }
//extern __device__ void *memmove_(void *dest, const void *src, size_t n);
//__device__ __forceinline void *memset_(void *s, int c, size_t n) { return (s ? memset(s, c, n) : nullptr); }
//extern __device__ int memcmp_(const void *s1, const void *s2, size_t n);
//extern __device__ void *memchr_(const void *s, int c, size_t n);
//extern __device__ char *strcpy_(char *__restrict dest, const char *__restrict src);
//extern __device__ char *strncpy_(char *__restrict dest, const char *__restrict src, size_t n);
//extern __device__ char *strcat_(char *__restrict dest, const char *__restrict src);
//extern __device__ char *strncat_(char *__restrict dest, const char *__restrict src, size_t n);
//extern __device__ int strcmp_(const char *s1, const char *s2);
//extern __device__ int stricmp_(const char *s1, const char *s2);
//extern __device__ int strncmp_(const char *s1, const char *s2, size_t n);
//extern __device__ int strnicmp_(const char *s1, const char *s2, size_t n);
//extern __device__ int strcoll_(const char *s1, const char *s2);
//extern __device__ size_t strxfrm_(char *__restrict dest, const char *__restrict src, size_t n);
//extern __device__ char *strdup_(const char *s);
//extern __device__ char *strndup_(const char *s, size_t n);
//extern __device__ char *strchr_(const char *s, int c);
//extern __device__ char *strrchr_(const char *s, int c);
//extern __device__ size_t strcspn_(const char *s, const char *reject);
//extern __device__ size_t strspn_(const char *s, const char *accept);
//extern __device__ char *strpbrk_(const char *s, const char *accept);
//extern __device__ char *strstr_(const char *haystack, const char *needle);
//extern __device__ char *strtok_(char *__restrict s, const char *__restrict delim);
//extern __device__ void *mempcpy_(void *__restrict dest, const void *__restrict src, size_t n);
//extern __device__ size_t strlen_(const char *s);
//__forceinline __device__ size_t strlen16(const void *s)
//extern __device__ size_t strnlen_(const char *s, size_t maxlen);
//extern __device__ char *strerror_(int errnum);
