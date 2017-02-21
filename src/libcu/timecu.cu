#include <timecu.h>

__BEGIN_DECLS;
#if defined(__CUDA_ARCH__) || defined(LIBCUFORCE)

// time
__device__ time_t time_(time_t *timer)
{
	clock_t start = clock();
	time_t epoch = 0;
	return epoch;
}

// gettimeofday
__device__ int gettimeofday_(struct timeval *tp, void *tz)
{
	time_t seconds = time(nullptr);
	tp->tv_usec = 0;
	tp->tv_sec = seconds;
	return 0;
	//if (tz)
	//	_abort();
	//tp->tv_usec = 0;
	//return (_time(&tp->tv_sec) == (time_t)-1 ? -1 : 0);
}

#endif
__END_DECLS;