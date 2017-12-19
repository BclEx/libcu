#include "stdafx.h"

using namespace System;
using namespace System::Text;
using namespace System::Collections::Generic;
using namespace Microsoft::VisualStudio::TestTools::UnitTesting;

cudaError_t crtdefs_test1();
cudaError_t ctype_test1();
namespace libcutests
{
	__BEGIN_TEST(crtdefs);
	[TestMethod, TestCategory("core")] void crtdefs_test1() { Assert::AreEqual("no error", gcnew String(cudaGetErrorString(::crtdefs_test1()))); }
	__END_TEST;
	__BEGIN_TEST(ctype);
	[TestMethod, TestCategory("core")] void ctype_test1() { Assert::AreEqual("no error", gcnew String(cudaGetErrorString(::ctype_test1()))); }
	__END_TEST;
}
