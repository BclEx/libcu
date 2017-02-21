#include <errnocu.h>

__device__ int errno_;
extern __device__ int *_errno_(void) { return &errno_; }
extern __device__ errno_t _set_errno_(int value) { return (errno_ = value); }
extern __device__ errno_t _get_errno_(int *value) { if (value) *value = errno_; return errno_; }
