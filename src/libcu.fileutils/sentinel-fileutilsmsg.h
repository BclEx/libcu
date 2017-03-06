/*
sentinel-fileutilsmsg.h - messages for sentinel
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

#ifndef _SENTINEL_FILEUTILSMSG_H
#define _SENTINEL_FILEUTILSMSG_H
#define HAS_GPU 0
#define HAS_HOSTSENTINEL 1
#include <sentinel.h>
#include <string.h>

enum {
	FILEUTILS_DCAT = 0,
	FILEUTILS_DMKDIR,
	FILEUTILS_DRMDIR,
};

struct fileutils_dcat
{
	static __forceinline __device__ char *Prepare(fileutils_dcat *t, char *data, char *dataEnd, long offset)
	{
		int strLength = (t->Str ? (int)strlen(t->Str) + 1 : 0);
		char *str = (char *)(data += _ROUND8(sizeof(*t)));
		char *end = (char *)(data += strLength);
		if (end > dataEnd) return nullptr;
		memcpy(str, t->Str, strLength);
		t->Str = str;
		return end;
	}
	sentinelMessage Base;
	char *Str;
	__device__ fileutils_dcat(char *str)
		: Base(true, FILEUTILS_DCAT, 1024, SENTINELPREPARE(Prepare)), Str(str) { sentinelClientSend(&Base, sizeof(fileutils_dcat)); }
	int RC;
};

struct fileutils_dmkdir
{
	static __forceinline __device__ char *Prepare(fileutils_dmkdir *t, char *data, char *dataEnd, long offset)
	{
		int strLength = (t->Str ? (int)strlen(t->Str) + 1 : 0);
		char *str = (char *)(data += _ROUND8(sizeof(*t)));
		char *end = (char *)(data += strLength);
		if (end > dataEnd) return nullptr;
		memcpy(str, t->Str, strLength);
		t->Str = str;
		return end;
	}
	sentinelMessage Base;
	char *Str; unsigned short Mode;
	__device__ fileutils_dmkdir(char *str, unsigned short mode)
		: Base(true, FILEUTILS_DMKDIR, 1024, SENTINELPREPARE(Prepare)), Str(str), Mode(mode) { sentinelClientSend(&Base, sizeof(fileutils_dmkdir)); }
	int RC;
};

struct fileutils_drmdir
{
	static __forceinline __device__ char *Prepare(fileutils_drmdir *t, char *data, char *dataEnd, long offset)
	{
		int strLength = (t->Str ? (int)strlen(t->Str) + 1 : 0);
		char *str = (char *)(data += _ROUND8(sizeof(*t)));
		char *end = (char *)(data += strLength);
		if (end > dataEnd) return nullptr;
		memcpy(str, t->Str, strLength);
		t->Str = str;
		return end;
	}
	sentinelMessage Base;
	char *Str;
	__device__ fileutils_drmdir(char *str)
		: Base(true, FILEUTILS_DRMDIR, 1024, SENTINELPREPARE(Prepare)), Str(str) { sentinelClientSend(&Base, sizeof(fileutils_drmdir)); }
	int RC;
};

#endif  /* _SENTINEL_FILEUTILSMSG_H */