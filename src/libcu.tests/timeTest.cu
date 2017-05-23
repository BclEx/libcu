#include <cuda_runtime.h>
#include <stdiocu.h>
#include <timecu.h>
#include <assert.h>

static __global__ void g_time_test1()
{
	printf("time_test1\n");
}
cudaError_t time_test1() { g_time_test1<<<1, 1>>>(); return cudaDeviceSynchronize(); }


////builtin: extern __device__ clock_t clock();
//extern __device__ time_t time_(time_t *timer);
//extern __device__ double difftime_(time_t time1, time_t time0);
//__forceinline __device__ time_t mktime_(struct tm *tp) { time_mktime msg(tp); return msg.RC; }
//__forceinline size_t strftime_(char *__restrict s, size_t maxsize, const char *__restrict format, const struct tm *__restrict tp) { time_strftime msg(s, maxsize, format, tp); return msg.RC; }
//extern __device__ struct tm *gmtime_(const time_t *timer);
//extern __device__ char *asctime_(const struct tm *tp);
//__forceinline __device__ char *ctime_(const time_t *timer) { return asctime(localtime(timer)); }
