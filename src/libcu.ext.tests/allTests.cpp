#include "stdafx.h"

using namespace System;
using namespace System::Text;
using namespace System::Collections::Generic;
using namespace Microsoft::VisualStudio::TestTools::UnitTesting;

cudaError_t alloc_test1();
cudaError_t bitvec_test1();
cudaError_t convert_test1();
cudaError_t global_test1();
cudaError_t main_test1();
cudaError_t math_test1();
cudaError_t mutex_test1();
cudaError_t notify_test1();
cudaError_t pcache_test1();
cudaError_t pcache1_test1();
cudaError_t printf_test1();
cudaError_t random_test1();
cudaError_t status_test1();
cudaError_t utf_test1();
cudaError_t util_test1();
cudaError_t vsystem_test1();

namespace libcutests
{
	// alloc
	__BEGIN_TEST(alloc);
	[TestMethod, TestCategory("ext")] void alloc_test1() { Assert::AreEqual("no error", gcnew String(cudaGetErrorString(::alloc_test1()))); }
	__END_TEST;

	// bitvec
	__BEGIN_TEST(bitvec);
	[TestMethod, TestCategory("ext")] void bitvec_test1() { Assert::AreEqual("no error", gcnew String(cudaGetErrorString(::bitvec_test1()))); }
	__END_TEST;

	// convert
	__BEGIN_TEST(convert);
	[TestMethod, TestCategory("ext")] void convert_test1() { Assert::AreEqual("no error", gcnew String(cudaGetErrorString(::convert_test1()))); }
	__END_TEST;

	// global
	__BEGIN_TEST(global);
	[TestMethod, TestCategory("ext")] void global_test1() { Assert::AreEqual("no error", gcnew String(cudaGetErrorString(::global_test1()))); }
	__END_TEST;

	// main
	__BEGIN_TEST(main);
	[TestMethod, TestCategory("ext")] void main_test1() { Assert::AreEqual("no error", gcnew String(cudaGetErrorString(::main_test1()))); }
	__END_TEST;

	// math
	__BEGIN_TEST(math);
	[TestMethod, TestCategory("ext")] void math_test1() { Assert::AreEqual("no error", gcnew String(cudaGetErrorString(::math_test1()))); }
	__END_TEST;

	// mutex
	__BEGIN_TEST(mutex);
	[TestMethod, TestCategory("ext")] void mutex_test1() { Assert::AreEqual("no error", gcnew String(cudaGetErrorString(::mutex_test1()))); }
	__END_TEST;

	// notify
	__BEGIN_TEST(notify);
	[TestMethod, TestCategory("ext")] void notify_test1() { Assert::AreEqual("no error", gcnew String(cudaGetErrorString(::notify_test1()))); }
	__END_TEST;

	// pcache
	__BEGIN_TEST(pcache);
	[TestMethod, TestCategory("ext")] void pcache_test1() { Assert::AreEqual("no error", gcnew String(cudaGetErrorString(::pcache_test1()))); }
	__END_TEST;

	// pcache1
	__BEGIN_TEST(pcache1);
	[TestMethod, TestCategory("ext")] void pcache1_test1() { Assert::AreEqual("no error", gcnew String(cudaGetErrorString(::pcache1_test1()))); }
	__END_TEST;

	// printf
	__BEGIN_TEST(printf);
	[TestMethod, TestCategory("ext")] void printf_test1() { Assert::AreEqual("no error", gcnew String(cudaGetErrorString(::printf_test1()))); }
	__END_TEST;

	// random
	__BEGIN_TEST(random);
	[TestMethod, TestCategory("ext")] void random_test1() { Assert::AreEqual("no error", gcnew String(cudaGetErrorString(::random_test1()))); }
	__END_TEST;

	// status
	__BEGIN_TEST(status);
	[TestMethod, TestCategory("ext")] void status_test1() { Assert::AreEqual("no error", gcnew String(cudaGetErrorString(::status_test1()))); }
	__END_TEST;

	// utf
	__BEGIN_TEST(utf);
	[TestMethod, TestCategory("ext")] void utf_test1() { Assert::AreEqual("no error", gcnew String(cudaGetErrorString(::utf_test1()))); }
	__END_TEST;

	// util
	__BEGIN_TEST(util);
	[TestMethod, TestCategory("ext")] void util_test1() { Assert::AreEqual("no error", gcnew String(cudaGetErrorString(::util_test1()))); }
	__END_TEST;

	// vsystem
	__BEGIN_TEST(vsystem);
	[TestMethod, TestCategory("ext")] void vsystem_test1() { Assert::AreEqual("no error", gcnew String(cudaGetErrorString(::vsystem_test1()))); }
	__END_TEST;
}
