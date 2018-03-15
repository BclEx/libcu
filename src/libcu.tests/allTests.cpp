#include "stdafx.h"
#include <falloc.h>

using namespace System;
using namespace System::Text;
using namespace System::Collections::Generic;
using namespace Microsoft::VisualStudio::TestTools::UnitTesting;

cudaError_t crtdefs_test1();
cudaError_t ctype_test1();
cudaError_t dirent_test1();
cudaError_t errno_test1();
cudaError_t falloc_lauched_cuda_kernel();
cudaError_t falloc_alloc_with_getchunk();
cudaError_t falloc_alloc_with_getchunks();
cudaError_t falloc_alloc_with_context();
cudaError_t fcntl_test1();
cudaError_t fsystem_test1();
cudaError_t grp_test1();
cudaError_t pwd_test1();
cudaError_t regex_test1();
cudaError_t sentinel_test1();
cudaError_t setjmp_test1();
//cudaError_t stdarg_parse();
//cudaError_t stdarg_call();
cudaError_t stddef_test1();
cudaError_t stdio_test1();
cudaError_t stdio_64bit();
cudaError_t stdio_ganging();
cudaError_t stdio_scanf();
cudaError_t stdlib_test1();
cudaError_t stdlib_strtol();
cudaError_t stdlib_strtoq();
cudaError_t string_test1();
cudaError_t time_test1();
cudaError_t unistd_test1();

namespace libcutests
{
	// crtdefs
	__BEGIN_TEST(crtdefs);
	[TestMethod, TestCategory("core")] void crtdefs_test1() { Assert::AreEqual("no error", gcnew String(cudaGetErrorString(::crtdefs_test1()))); }
	__END_TEST;

	// ctype
	__BEGIN_TEST(ctype);
	[TestMethod, TestCategory("core")] void ctype_test1() { Assert::AreEqual("no error", gcnew String(cudaGetErrorString(::ctype_test1()))); }
	__END_TEST;

	// dirent
	__BEGIN_TEST(dirent);
	[TestMethod, TestCategory("fsystem")] void dirent_test1() { Assert::AreEqual("no error", gcnew String(cudaGetErrorString(::dirent_test1()))); }
	__END_TEST;

	// errno
	__BEGIN_TEST(errno);
	[TestMethod, TestCategory("core")] void errno_test1() { Assert::AreEqual("no error", gcnew String(cudaGetErrorString(::errno_test1()))); }
	__END_TEST;

	// falloc
	cudaDeviceFallocHeap _deviceFallocHeap;
	__BEGIN_TEST(falloc, \
		_deviceFallocHeap = cudaDeviceFallocHeapCreate(1024, 4098); cudaFallocSetDefaultHeap(_deviceFallocHeap); ,\
		cudaDeviceFallocHeapDestroy(_deviceFallocHeap););
	[TestMethod, TestCategory("falloc")] void falloc_lauched_cuda_kernel() { Assert::AreEqual("no error", gcnew String(cudaGetErrorString(::falloc_lauched_cuda_kernel()))); }
	[TestMethod, TestCategory("falloc")] void falloc_alloc_with_getchunk() { Assert::AreEqual("no error", gcnew String(cudaGetErrorString(::falloc_alloc_with_getchunk()))); }
	[TestMethod, TestCategory("falloc")] void falloc_alloc_with_getchunks() { Assert::AreEqual("no error", gcnew String(cudaGetErrorString(::falloc_alloc_with_getchunks()))); }
	[TestMethod, TestCategory("falloc")] void falloc_alloc_with_context() { Assert::AreEqual("no error", gcnew String(cudaGetErrorString(::falloc_alloc_with_context()))); }
	__END_TEST;

	// fcntl
	__BEGIN_TEST(fcntl);
	[TestMethod, TestCategory("fsystem")] void fcntl_test1() { Assert::AreEqual("no error", gcnew String(cudaGetErrorString(::fcntl_test1()))); }
	__END_TEST;

	// fsystem
	__BEGIN_TEST(fsystem);
	[TestMethod, TestCategory("fsystem")] void fsystem_test1() { Assert::AreEqual("no error", gcnew String(cudaGetErrorString(::fsystem_test1()))); }
	__END_TEST;

	// grp
	__BEGIN_TEST(grp);
	[TestMethod, TestCategory("core")] void grp_test1() { Assert::AreEqual("no error", gcnew String(cudaGetErrorString(::grp_test1()))); }
	__END_TEST;

	// pwd
	__BEGIN_TEST(pwd);
	[TestMethod, TestCategory("core")] void pwd_test1() { Assert::AreEqual("no error", gcnew String(cudaGetErrorString(::pwd_test1()))); }
	__END_TEST;

	// regex
	__BEGIN_TEST(regex);
	[TestMethod, TestCategory("core")] void regex_test1() { Assert::AreEqual("no error", gcnew String(cudaGetErrorString(::regex_test1()))); }
	__END_TEST;

	// sentinel
	__BEGIN_TEST(sentinel);
	[TestMethod, TestCategory("core")] void sentinel_test1() { Assert::AreEqual("no error", gcnew String(cudaGetErrorString(::sentinel_test1()))); }
	__END_TEST;

	// setjmp
	__BEGIN_TEST(setjmp);
	[TestMethod, TestCategory("core")] void setjmp_test1() { Assert::AreEqual("no error", gcnew String(cudaGetErrorString(::setjmp_test1()))); }
	__END_TEST;

	//// stdarg
	//__BEGIN_TEST(stdarg);
	//[TestMethod, TestCategory("core")] void stdarg_parse() { Assert::AreEqual("no error", gcnew String(cudaGetErrorString(::stdarg_parse()))); }
	//[TestMethod, TestCategory("core")] void stdarg_call() { Assert::AreEqual("no error", gcnew String(cudaGetErrorString(::stdarg_call()))); }
	//__END_TEST;

	// stddef
	__BEGIN_TEST(stddef);
	[TestMethod, TestCategory("core")] void stddef_test1() { Assert::AreEqual("no error", gcnew String(cudaGetErrorString(::stddef_test1()))); }
	__END_TEST;

	// stdio
	__BEGIN_TEST(stdio);
	[TestMethod, TestCategory("core")] void stdio_test1() { Assert::AreEqual("no error", gcnew String(cudaGetErrorString(::stdio_test1()))); }
	[TestMethod, TestCategory("core")] void stdio_64bit() { Assert::AreEqual("no error", gcnew String(cudaGetErrorString(::stdio_64bit()))); }
	[TestMethod, TestCategory("core")] void stdio_ganging() { Assert::AreEqual("no error", gcnew String(cudaGetErrorString(::stdio_ganging()))); }
	[TestMethod, TestCategory("core")] void stdio_scanf() { Assert::AreEqual("no error", gcnew String(cudaGetErrorString(::stdio_scanf()))); }
	__END_TEST;

	// stdlib
	__BEGIN_TEST(stdlib);
	[TestMethod, TestCategory("core")] void stdlib_test1() { Assert::AreEqual("no error", gcnew String(cudaGetErrorString(::stdlib_test1()))); }
	[TestMethod, TestCategory("core")] void stdlib_strtol() { Assert::AreEqual("no error", gcnew String(cudaGetErrorString(::stdlib_strtol()))); }
	[TestMethod, TestCategory("core")] void stdlib_strtoq() { Assert::AreEqual("no error", gcnew String(cudaGetErrorString(::stdlib_strtoq()))); }
	__END_TEST;

	// string
	__BEGIN_TEST(string);
	[TestMethod, TestCategory("core")] void string_test1() { Assert::AreEqual("no error", gcnew String(cudaGetErrorString(::string_test1()))); }
	__END_TEST;

	// time
	__BEGIN_TEST(time);
	[TestMethod, TestCategory("core")] void time_test1() { Assert::AreEqual("no error", gcnew String(cudaGetErrorString(::time_test1()))); }
	__END_TEST;

	// unistd
	__BEGIN_TEST(unistd);
	[TestMethod, TestCategory("fsystem")] void unistd_test1() { Assert::AreEqual("no error", gcnew String(cudaGetErrorString(::unistd_test1()))); }
	__END_TEST;
}
