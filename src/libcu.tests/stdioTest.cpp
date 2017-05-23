#include "stdafx.h"

using namespace System;
using namespace System::Text;
using namespace System::Collections::Generic;
using namespace Microsoft::VisualStudio::TestTools::UnitTesting;

cudaError_t stdio_test1();
cudaError_t stdio_64bit();
cudaError_t stdio_ganging();
cudaError_t stdio_scanf();
namespace libcutests
{
	[TestClass]
	public ref class stdioTest
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
		[TestInitialize()]void TestInitialize() { allTestInitialize(); }
		[TestCleanup()] void TestCleanup() { allTestCleanup(); }
#pragma endregion 

		[TestMethod, TestCategory("core")] void stdio_test1() { Assert::AreEqual("no error", gcnew String(cudaGetErrorString(::stdio_test1()))); }
		[TestMethod, TestCategory("core")] void stdio_64bit() { Assert::AreEqual("no error", gcnew String(cudaGetErrorString(::stdio_64bit()))); }
		[TestMethod, TestCategory("core")] void stdio_ganging() { Assert::AreEqual("no error", gcnew String(cudaGetErrorString(::stdio_ganging()))); }
		[TestMethod, TestCategory("core")] void stdio_scanf() { Assert::AreEqual("no error", gcnew String(cudaGetErrorString(::stdio_scanf()))); }
	};
}
