#include <errnocu.h>
#include <cuda_runtimecu.h>

__device__ int errno_;
extern __device__ int *_errno(void) { return &errno_; }
extern __device__ errno_t _set_errno(int value) { return (errno_ = value); }
extern __device__ errno_t _get_errno(int *value) { if (value) *value = errno_; return errno_; }