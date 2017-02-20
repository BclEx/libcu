#undef __CUDA_RUNTIME_H__
#include <cuda_runtime.h>
#include <falloc.h>
#include <assert.h>

// launches cuda kernel
static __global__ void g_falloc_lauched_cuda_kernel()
{
	int gtid = blockIdx.x*blockDim.x + threadIdx.x;
	assert(gtid < 1);
}
cudaError_t falloc_lauched_cuda_kernel() { g_falloc_lauched_cuda_kernel<<<1, 1>>>(); return cudaDeviceSynchronize(); }

// alloc with get chunk
static __global__ void g_falloc_alloc_with_getchunk()
{
	// specific
	void *obj = fallocGetChunk(_defaultDeviceHeap);
	assert(obj != nullptr);
	fallocFreeChunk(obj, _defaultDeviceHeap);

	// default
	void *obj2 = fallocGetChunk();
	assert(obj2 != nullptr);
	fallocFreeChunk(obj2);
}
cudaError_t falloc_alloc_with_getchunk() { g_falloc_alloc_with_getchunk<<<1, 1>>>(); return cudaDeviceSynchronize(); }

// alloc with get chunks
static __global__ void g_falloc_alloc_with_getchunks()
{
	//void *obj = fallocGetBlocks(144 * 2);
	//assert(obj != nullptr);
	//fallocFreeChunks(obj);
	//
	//void *obj2 = fallocGetBlocks(144 * 2);
	//assert(obj2 != nullptr);
	//fallocFreeChunks(obj2);
}
cudaError_t falloc_alloc_with_getchunks() { g_falloc_alloc_with_getchunks<<<1, 1>>>(); return cudaDeviceSynchronize(); }

// alloc with context
static __global__ void g_falloc_alloc_with_context()
{
	fallocCtx *ctx = fallocCreateCtx();
	assert(ctx != nullptr);
	char *testString = (char *)falloc(ctx, 10);
	assert(testString != nullptr);
	int *testInteger = falloc<int>(ctx);
	assert(testInteger != nullptr);
	fallocDisposeCtx(ctx);
}
cudaError_t falloc_alloc_with_context() { g_falloc_alloc_with_context<<<1, 1>>>(); return cudaDeviceSynchronize(); }

// alloc with context as stack
static __global__ void g_falloc_alloc_with_context_as_stack()
{
	fallocCtx *ctx = fallocCreateCtx();
	assert(ctx != nullptr);
	fallocPush<int>(ctx, 1);
	fallocPush<int>(ctx, 2);
	int b = fallocPop<int>(ctx);
	int a = fallocPop<int>(ctx);
	assert(b == 2 && a == 1);
	fallocDisposeCtx(ctx);
}
cudaError_t falloc_alloc_with_context_as_stack() { g_falloc_alloc_with_context_as_stack<<<1, 1>>>(); return cudaDeviceSynchronize(); }
