#include <errnocu.h>

__BEGIN_DECLS;

__device__ int errno_;
extern __device__ int *_errno_(void) { return &errno_; }
extern __device__ int _set_errno_(int value) { return (errno_ = value); }
extern __device__ int _get_errno_(int *value) { if (value) *value = errno_; return errno_; }

__END_DECLS;
