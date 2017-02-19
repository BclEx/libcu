#include <cuda_runtimecu.h>
#include <setjmpcu.h>

__BEGIN_DECLS;

/* Store the calling environment in ENV, also saving the signal mask. Return 0.  */
__device__ int setjmp(jmp_buf env)
{
	panic("Not Implemented");
	return 0;
}

/* Jump to the environment saved in ENV, making the `setjmp' call there return VAL, or 1 if VAL is 0.  */
__device__ void longjmp(jmp_buf env, int val)
{
	panic("Not Implemented");
}

__END_DECLS;
