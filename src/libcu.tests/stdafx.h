// stdafx.h : include file for standard system include files,
// or project specific include files that are used frequently,
// but are changed infrequently

#pragma once

#include <cuda_runtimecu.h>

void allClassInitialize(bool sentinel = true);
void allClassCleanup(bool sentinel = true);
void allTestInitialize();
void allTestCleanup();

#define __BEGIN_TEST(name) \
	[TestClass] \
	public ref class name##Test \
{ \
private: \
	TestContext^ _testCtx; \
public: \
	[ClassInitialize()] static void ClassInitialize(Microsoft::VisualStudio::TestTools::UnitTesting::TestContext^ testContext) { allClassInitialize(); } \
	[ClassCleanup()] static void ClassCleanup() { allClassCleanup(); } \
	[TestInitialize()]void TestInitialize() { allTestInitialize(); } \
	[TestCleanup()] void TestCleanup() { allTestCleanup(); } \
	property Microsoft::VisualStudio::TestTools::UnitTesting::TestContext^ TestContext \
{ \
	Microsoft::VisualStudio::TestTools::UnitTesting::TestContext^ get() { return _testCtx; } \
	System::Void set(Microsoft::VisualStudio::TestTools::UnitTesting::TestContext^ value) { _testCtx = value; } \
}

#define __END_TEST };