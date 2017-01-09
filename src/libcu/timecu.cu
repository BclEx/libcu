//// time
//// TODO: BUILD
//__device__ time_t _time(time_t *timer)
//{
//	//clock_t start = clock();
//	time_t epoch = 0;
//	return epoch;
//}
//
//// gettimeofday
//// TODO: BUILD
//__device__ int _gettimeofday(struct timeval *tp, void *tz)
//{
//	time_t seconds = _time(nullptr);
//	tp->tv_usec = 0;
//	tp->tv_sec = seconds;
//	return 0;
//	//if (tz)
//	//	_abort();
//	//tp->tv_usec = 0;
//	//return (_time(&tp->tv_sec) == (time_t)-1 ? -1 : 0);
//}
