// stdafx.h : include file for standard system include files,
// or project specific include files that are used frequently,
// but are changed infrequently

#pragma once

#include <cuda_runtimecu.h>
#include <sentinel.h>

void allClassInitialize(bool sentinel = true);
void allClassCleanup(bool sentinel = true);
void allTestInitialize();
void allTestCleanup();
