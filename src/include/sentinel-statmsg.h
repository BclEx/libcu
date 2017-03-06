/*
sentinel-statmsg.h - messages for sentinel
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
#ifndef _SENTINEL_STATMSG_H
#define _SENTINEL_STATMSG_H

#include <sentinel.h>
#include <crtdefscu.h>
#include <stringcu.h>

enum {
	STAT_STAT = 45,
	STAT_FSTAT,
	//STAT_STAT64,
	//STAT_FSTAT64,
	STAT_MKDIR,
};

struct stat_stat
{
	static __forceinline __device__ char *Prepare(stat_stat *t, char *data, char *dataEnd, long offset)
	{
		int strLength = (t->Str ? (int)strlen(t->Str) + 1 : 0);
		char *str = (char *)(data += _ROUND8(sizeof(*t)));
		char *end = (char *)(data += strLength);
		if (end > dataEnd) return nullptr;
		memcpy(str, t->Str, strLength);
		t->Str = str + offset;
		t->Ptr = (struct stat *)(str + offset);
		return end;
	}
	sentinelMessage Base;
	const char *Str; struct stat *Ptr;
	__device__ stat_stat(const char *str, struct stat *ptr)
		: Base(true, STAT_STAT, 1024, SENTINELPREPARE(Prepare)), Str(str), Ptr(ptr) { sentinelDeviceSend(&Base, sizeof(stat_stat)); }
	int RC;
};

struct stat_fstat
{
	static __forceinline __device__ char *Prepare(stat_fstat *t, char *data, char *dataEnd, long offset)
	{
		char *ptr = (char *)(data += _ROUND8(sizeof(*t)));
		char *end = (char *)(data += 1024);
		if (end > dataEnd) return nullptr;
		t->Ptr = (struct stat *)ptr;
		return end;
	}
	sentinelMessage Base;
	int Handle; struct stat *Ptr;
	__device__ stat_fstat(int fd, struct stat *ptr)
		: Base(true, STAT_FSTAT), Handle(fd), Ptr(ptr) { sentinelDeviceSend(&Base, sizeof(stat_fstat)); }
	int RC;
};

struct stat_mkdir
{
	static __forceinline __device__ char *Prepare(stat_mkdir *t, char *data, char *dataEnd, long offset)
	{
		int strLength = (t->Str ? (int)strlen(t->Str) + 1 : 0);
		char *str = (char *)(data += _ROUND8(sizeof(*t)));
		char *end = (char *)(data += strLength);
		if (end > dataEnd) return nullptr;
		memcpy(str, t->Str, strLength);
		t->Str = str + offset;
		return end;
	}
	sentinelMessage Base;
	const char *Str; mode_t Mode;
	__device__ stat_mkdir(const char *str, mode_t mode)
		: Base(true, STAT_MKDIR, 1024, SENTINELPREPARE(Prepare)), Str(str), Mode(mode) { sentinelDeviceSend(&Base, sizeof(stat_mkdir)); }
	int RC;
};

#endif  /* _SENTINEL_STATMSG_H */