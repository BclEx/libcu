/*
cuda_runtimecu.h - Inject in cuda_runtime
The MIT License

Copyright (c) 2016 Sky Morey

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

#ifndef __CUDA_RUNTIMECU_H__
#define __CUDA_RUNTIMECU_H__

#include <crtdefscu.h>

// http://stackoverflow.com/questions/29706730/which-headers-are-included-by-default-in-the-cu-source-file

#undef __CUDA_RUNTIME_H__
#include <cuda_runtime.h>

#undef va_start
#undef va_arg
#undef va_end
#define va_start _crt_va_start
#define va_restart _crt_va_restart
#define va_arg _crt_va_arg
#define va_end _crt_va_end

#define cudaErrorCheck(x) { gpuAssert((x), #x, __FILE__, __LINE__, true); }
#define cudaErrorCheckA(x) { gpuAssert((x), #x, __FILE__, __LINE__, false); }
#define cudaErrorCheckF(x, f) { if (!gpuAssert((x), #x, __FILE__, __LINE__, false)) f; }
#define cudaErrorCheckLast() { gpuAssert(cudaPeekAtLastError(), __FILE__, __LINE__); }
__forceinline bool gpuAssert(cudaError_t code, const char *action, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess) 
	{
		//fprintf(stderr, "GPUassert: %s [%s:%d]\n", cudaGetErrorString(code), file, line);
		//getchar();
		if (abort) exit(code);
		return false;
	}
	return true;
}

__forceinline int __convertSMVer2Cores(int major, int minor)
{
	typedef struct // Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
	{
		int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
		int Cores;
	} SMToCores;
	SMToCores gpuArchCoresPerSM[] = {
		{ 0x10,  8 }, // Tesla Generation (SM 1.0) G80 class
		{ 0x11,  8 }, // Tesla Generation (SM 1.1) G8x class
		{ 0x12,  8 }, // Tesla Generation (SM 1.2) G9x class
		{ 0x13,  8 }, // Tesla Generation (SM 1.3) GT200 class
		{ 0x20, 32 }, // Fermi Generation (SM 2.0) GF100 class
		{ 0x21, 48 }, // Fermi Generation (SM 2.1) GF10x class
		{ 0x30, 192}, // Kepler Generation (SM 3.0) GK10x class
		{ 0x35, 192}, // Kepler Generation (SM 3.5) GK11x class
		{   -1, -1 }
	};
	int index = 0;
	while (gpuArchCoresPerSM[index].SM != -1)
	{
		if (gpuArchCoresPerSM[index].SM == ((major << 4) + minor))
			return gpuArchCoresPerSM[index].Cores;
		index++;
	}
	//printf("MapSMtoCores SM %d.%d is undefined (please update to the latest SDK)!\n", major, minor);
	return -1;
}

__forceinline int gpuGetMaxGflopsDeviceId()
{
	int deviceCount = 0;
	cudaDeviceProp deviceProp;
	cudaGetDeviceCount(&deviceCount);
	if (deviceCount == 1) return 0;
	// Find the best major SM Architecture GPU device
	int bestMajor = 0;
	for (int i = 0; i < deviceCount; i++)
	{
		cudaGetDeviceProperties(&deviceProp, i);
		// If this GPU is not running on Compute Mode prohibited, then we can add it to the list
		if (deviceProp.computeMode != cudaComputeModeProhibited && deviceProp.major > 0 && deviceProp.major < 9999)
			bestMajor = (bestMajor > deviceProp.major ? bestMajor : deviceProp.major);
	}
	// Find the best CUDA capable GPU device
	int bestDevice = 0;
	unsigned long long basePerformace = 0;
	for (int i = 0; i < deviceCount; i++ )
	{
		cudaGetDeviceProperties(&deviceProp, i);
		// If this GPU is not running on Compute Mode prohibited, then we can add it to the list
		if (deviceProp.computeMode != cudaComputeModeProhibited)
		{
			int sm_per_multiproc = (deviceProp.major == 9999 && deviceProp.minor == 9999 ? 1 : __convertSMVer2Cores(deviceProp.major, deviceProp.minor));
			unsigned long long performace = (deviceProp.multiProcessorCount * sm_per_multiproc * deviceProp.clockRate);
			if (performace > basePerformace)
			{
				basePerformace = performace;
				bestDevice = i;
			}
		}
	}
	return bestDevice;
}

#endif /* __CUDA_RUNTIMECU_H__ */