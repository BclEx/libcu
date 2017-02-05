// stdafx.cpp : source file that includes just the standard includes
// libcu.tests.pch will be the pre-compiled header
// stdafx.obj will contain the pre-compiled type information

#include "stdafx.h"

void allClassInitialize(bool sentinel)
{
	return;
	if (sentinel)
		sentinelServerInitialize();

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaError_t cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess)
		throw gcnew System::InvalidOperationException("cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
}

void allClassCleanup(bool sentinel)
{
	return;
	if (sentinel)
		sentinelServerShutdown();

	// cudaDeviceReset must be called before exiting in order for profiling and tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaError_t cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess)
		throw gcnew System::InvalidOperationException("cudaDeviceReset failed!");
}

void allTestInitialize()
{
	cudaError_t cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess)
		throw gcnew System::InvalidOperationException("cudaDeviceReset failed!");
}

void allTestCleanup()
{
	return;
	// Check for any errors launching the kernel
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
		throw gcnew System::InvalidOperationException(System::String::Format("Kernel launch failed: {0}\n", gcnew System::String(cudaGetErrorString(cudaStatus))));

	// cudaDeviceSynchronize waits for the kernel to finish, and returns any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
		throw gcnew System::InvalidOperationException(System::String::Format("cudaDeviceSynchronize returned error code {0} after launching Kernel!\n", (int)cudaStatus));
}

