/*
falloc.h - Forward-only memory allocator
The MIT License

Copyright (c) 2010 Sky Morey

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#pragma once

///////////////////////////////////////////////////////////////////////////////
// HOST SIDE
// External function definitions for host-side code
#pragma region HOST SIDE

typedef struct
{
	void *reserved;
	void *deviceHeap;
	size_t chunkSize;
	size_t chunksLength;
	size_t length;
} cudaDeviceFallocHeap;

//	cudaFallocSetDefaultHeap
extern "C" cudaError_t cudaFallocSetDefaultHeap(cudaDeviceFallocHeap &heap);

//	cudaDeviceFallocCreate
//
//	Call this to initialize a falloc heap. If the buffer size needs to be changed, call cudaDeviceFallocDestroy()
//	before re-calling cudaDeviceFallocCreate().
//
//	The default size for the buffer is 1 megabyte. The buffer is filled linearly and
//	is completely used.
//
//	Arguments:
//		length - Length, in bytes, of total space to reserve (in device global memory) for output.
//
//	Returns:
//		cudaDeviceFalloc if all is well.
//
// default 2k chunks, 1-meg heap
extern "C" cudaDeviceFallocHeap cudaDeviceFallocHeapCreate(size_t chunkSize = 2046, size_t length = 1048576, cudaError_t *error = nullptr, void *reserved = nullptr);

//	cudaDeviceFallocDestroy
//
//	Cleans up all memories allocated by cudaDeviceFallocCreate() for a heap.
//	Call this at exit, or before calling cudaDeviceFallocCreate() again.
//
//	Arguments:
//		heap - device heap as valuetype
//
//	Returns:
//		cudaSuccess if all is well.
extern "C" cudaError_t cudaDeviceFallocHeapDestroy(cudaDeviceFallocHeap &heap);

#pragma endregion

///////////////////////////////////////////////////////////////////////////////
// DEVICE SIDE
// External function definitions for device-side code
#pragma region DEVICE SIDE
#if __CUDACC__

typedef struct cuFallocDeviceHeap fallocDeviceHeap;
extern __constant__ fallocDeviceHeap *_defaultDeviceHeap;
extern "C" __device__ void *fallocGetChunk(fallocDeviceHeap *heap = nullptr);
extern "C" __device__ void fallocFreeChunk(void *obj, fallocDeviceHeap *heap = nullptr);
#if MULTIBLOCK
extern "C" __device__ void *fallocGetChunks(size_t length, size_t *allocLength = nullptr, fallocDeviceHeap *heap = nullptr);
extern "C" __device__ void fallocFreeChunks(void *obj, fallocDeviceHeap *heap = nullptr);
#endif

// CONTEXT
typedef struct cuFallocCtx fallocCtx;
extern "C" __device__ fallocCtx *fallocCreateCtx(fallocDeviceHeap *heap = nullptr);
extern "C" __device__ void fallocDisposeCtx(fallocCtx *ctx);
extern "C" __device__ void *falloc(fallocCtx *ctx, unsigned short bytes, bool alloc = true);
extern "C" __device__ void *fallocRetract(fallocCtx *ctx, unsigned short bytes);
extern "C" __device__ void fallocMark(fallocCtx *ctx, void *&mark, unsigned short &mark2);
extern "C" __device__ bool fallocAtMark(fallocCtx *ctx, void *mark, unsigned short mark2);
template <typename T> __forceinline__ __device__ T *falloc(fallocCtx *ctx) { return (T *)falloc(ctx, sizeof(T), true); }
template <typename T> __forceinline__ __device__ void fallocPush(fallocCtx *ctx, T t) { *((T *)falloc(ctx, sizeof(T), false)) = t; }
template <typename T> __forceinline__ __device__ T fallocPop(fallocCtx *ctx) { return *((T *)fallocRetract(ctx, sizeof(T))); }

#endif
#pragma endregion
