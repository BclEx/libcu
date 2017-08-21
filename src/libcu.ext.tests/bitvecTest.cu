#include <stdiocu.h>
#include <crtdefscu.h>
#include <ext\bitvec.h>
#include <assert.h>

static __global__ void g_ext_bitvec_test1()
{
	printf("ext_bitvec_test1\n");
	/* Create a new bitmap object able to handle bits between 0 and iSize, inclusive.  Return a pointer to the new object.  Return NULL if malloc fails. */
	//__device__ bitvec_t *bitvecNew(uint32_t size);
	/* Check to see if the i-th bit is set.  Return true or false. If p is NULL (if the bitmap has not been created) or if i is out of range, then return false. */
	//__device__ bool bitvecGet(bitvec_t *b, uint32_t index);
	/* Set the i-th bit.  Return 0 on success and an error code if anything goes wrong. */
	//__device__ bool bitvecSet(bitvec_t *b, uint32_t index);
	/* Clear the i-th bit. */
	//__device__ void bitvecClear(bitvec_t *b, uint32_t index, void *buffer);
	/* Destroy a bitmap object.  Reclaim all memory used. */
	//__device__ void bitvecDestroy(bitvec_t *b);
	/* Return the value of the iSize parameter specified when Bitvec *p was created. */
	//__device__ uint32_t bitvecSize(bitvec_t *b);

}
cudaError_t ext_bitvec_test1() { g_ext_bitvec_test1<<<1, 1>>>(); return cudaDeviceSynchronize(); }