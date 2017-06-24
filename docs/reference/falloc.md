---
id: falloc
title: falloc.h
permalink: falloc.html
---

## #include <falloc.h>

## Host Side
Prototype | Description | Tags
--- | --- | :---:
```cudaError_t cudaFallocSetDefaultHeap(cudaDeviceFallocHeap &heap);``` | cudaFallocSetDefaultHeap
```cudaDeviceFallocHeap cudaDeviceFallocHeapCreate(size_t chunkSize = 2046, size_t length = 1048576, cudaError_t *error = nullptr, void *reserved = nullptr);``` | Call this to initialize a falloc heap. If the buffer size needs to be changed, call cudaDeviceFallocDestroy() before re-calling cudaDeviceFallocCreate().
```cudaError_t cudaDeviceFallocHeapDestroy(cudaDeviceFallocHeap &heap);``` | Cleans up all memories allocated by cudaDeviceFallocCreate() for a heap. Call this at exit, or before calling cudaDeviceFallocCreate() again.

## Device Side
Prototype | Description | Tags
--- | --- | :---:
```__device__ void *fallocGetChunk(fallocDeviceHeap *heap = nullptr);``` | xxxx
```__device__ void fallocFreeChunk(void *obj, fallocDeviceHeap *heap = nullptr);``` | xxxx
```__device__ void *fallocGetChunks(size_t length, size_t *allocLength = nullptr, fallocDeviceHeap *heap = nullptr);``` | xxxx | #multiblock
```__device__ void fallocFreeChunks(void *obj, fallocDeviceHeap *heap = nullptr);``` | xxxx | #multiblock

## Device Side, Context
Prototype | Description | Tags
--- | --- | :---:
```__device__ fallocCtx *fallocCreateCtx(fallocDeviceHeap *heap = nullptr);``` | xxxx
```__device__ void fallocDisposeCtx(fallocCtx *ctx);``` | xxxx
```__device__ void *falloc(fallocCtx *ctx, unsigned short bytes, bool alloc = true);``` | xxxx
```__device__ void *fallocRetract(fallocCtx *ctx, unsigned short bytes);``` | xxxx
```__device__ void fallocMark(fallocCtx *ctx, void *&mark, unsigned short &mark2);``` | xxxx
```__device__ bool fallocAtMark(fallocCtx *ctx, void *mark, unsigned short mark2);``` | xxxx
```__device__ T *falloc(fallocCtx *ctx);``` | xxxx | #template
```__device__ void fallocPush(fallocCtx *ctx, T t);``` | xxxx | #template
```__device__ T fallocPop(fallocCtx *ctx)``` | xxxx | #template
