/*
sentinel-timemsg.h - messages for sentinel
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
#ifndef _SENTINEL_TIMEMSG_H
#define _SENTINEL_TIMEMSG_H

#include <sentinel.h>
#include <crtdefscu.h>
#include <stringcu.h>

enum {
	TIME_MKTIME = 60,
	TIME_STRFTIME,
};

struct time_mktime
{
	sentinelMessage Base;
	struct tm *Tp;
	__device__ time_mktime(struct tm *tp)
		: Base(true, TIME_MKTIME), Tp(tp) { sentinelDeviceSend(&Base, sizeof(time_mktime)); }
	time_t RC;
};

struct time_strftime
{
	static __forceinline __device__ char *Prepare(time_strftime *t, char *data, char *dataEnd, intptr_t offset)
	{
		int strLength = (t->Str ? (int)strlen(t->Str) + 1 : 0);
		int str2Length = (t->Str2 ? (int)strlen(t->Str2) + 1 : 0);
		char *str = (char *)(data += _ROUND8(sizeof(*t)));
		char *str2 = (char *)(data += strLength);
		char *end = (char *)(data += str2Length);
		if (end > dataEnd) return nullptr;
		memcpy(str, t->Str, strLength);
		memcpy(str2, t->Str2, str2Length);
		t->Str = str + offset;
		t->Str2 = str2 + offset;
		return end;
	}
	sentinelMessage Base;
	const char *Str; size_t Maxsize; const char *Str2; const struct tm *Tp;
	__device__ time_strftime(const char *str, size_t maxsize, const char *str2, const struct tm *tp)
		: Base(true, TIME_STRFTIME, 1024, SENTINELPREPARE(Prepare)), Str(str), Maxsize(maxsize), Str2(str2), Tp(tp) { sentinelDeviceSend(&Base, sizeof(time_strftime)); }
	size_t RC;
};

#endif  /* _SENTINEL_TIMEMSG_H */