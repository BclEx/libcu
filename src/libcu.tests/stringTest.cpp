#include "stdafx.h"

using namespace System;
using namespace System::Text;
using namespace System::Collections::Generic;
using namespace Microsoft::VisualStudio::TestTools::UnitTesting;

cudaError_t string_test1();
namespace libcutests
{
	[TestClass]
	public ref class stringTest
	{
	private:
		TestContext^ _testCtx;

	public: 
		///<summary>
		///Gets or sets the test context which provides information about and functionality for the current test run.
		///</summary>
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

		[TestMethod] void string_test1() { Assert::AreEqual("no error", gcnew String(cudaGetErrorString(::string_test1()))); }
	};
}
