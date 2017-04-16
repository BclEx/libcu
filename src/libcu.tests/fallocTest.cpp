#include "stdafx.h"
#include <falloc.h>

using namespace System;
using namespace System::Text;
using namespace System::Collections::Generic;
using namespace Microsoft::VisualStudio::TestTools::UnitTesting;

cudaError_t falloc_lauched_cuda_kernel();
cudaError_t falloc_alloc_with_getchunk();
cudaError_t falloc_alloc_with_getchunks();
cudaError_t falloc_alloc_with_context();
namespace libcutests
{
	cudaDeviceFallocHeap _deviceFallocHeap;

	[TestClass]
	public ref class fallocTest
	{
	private:
		TestContext^ _testCtx;

	public: 
		property Microsoft::VisualStudio::TestTools::UnitTesting::TestContext^ TestContext
		{
			Microsoft::VisualStudio::TestTools::UnitTesting::TestContext^ get() { return _testCtx; }
			System::Void set(Microsoft::VisualStudio::TestTools::UnitTesting::TestContext^ value) { _testCtx = value; }
		}

#pragma region Initialize/Cleanup
		[ClassInitialize()] static void ClassInitialize(Microsoft::VisualStudio::TestTools::UnitTesting::TestContext^ testContext) { allClassInitialize(); }
		[ClassCleanup()] static void ClassCleanup() { allClassCleanup(); }
		[TestInitialize()] void TestInitialize() { allTestInitialize(); _deviceFallocHeap = cudaDeviceFallocHeapCreate(1024, 4098); cudaFallocSetDefaultHeap(_deviceFallocHeap); }
		[TestCleanup()] void TestCleanup() { allTestCleanup(); cudaDeviceFallocHeapDestroy(_deviceFallocHeap); }
#pragma endregion 

		[TestMethod, TestCategory("falloc")] void falloc_lauched_cuda_kernel() { Assert::AreEqual("no error", gcnew String(cudaGetErrorString(::falloc_lauched_cuda_kernel()))); }
		[TestMethod, TestCategory("falloc")] void falloc_alloc_with_getchunk() { Assert::AreEqual("no error", gcnew String(cudaGetErrorString(::falloc_alloc_with_getchunk()))); }
		[TestMethod, TestCategory("falloc")] void falloc_alloc_with_getchunks() { Assert::AreEqual("no error", gcnew String(cudaGetErrorString(::falloc_alloc_with_getchunks()))); }
		[TestMethod, TestCategory("falloc")] void falloc_alloc_with_context() { Assert::AreEqual("no error", gcnew String(cudaGetErrorString(::falloc_alloc_with_context()))); }
	};
}
