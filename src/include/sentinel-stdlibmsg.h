/*
sentinel-stdlibmsg.h - messages for sentinel
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

#ifndef _SENTINEL_STDLIBMSG_H
#define _SENTINEL_STDLIBMSG_H
#include <sentinel.h>
#include <stringcu.h>

enum {
	STDLIB_EXIT = 30,
	STDLIB_SYSTEM,
};

struct stdlib_exit
{
	sentinelMessage Base;
	bool Std;
	int Status;
	__device__ stdlib_exit(bool std, int status)
		: Base(false, STDLIB_EXIT), Std(std), Status(status) { sentinelDeviceSend(this, sizeof(stdlib_exit)); }
};

struct stdlib_system
{
	static __forceinline __device__ char *Prepare(stdlib_system *t, char *data, char *dataEnd)
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
	const char *Str;
	__device__ stdlib_system(const char *str)
		: Base(false, STDLIB_SYSTEM, 1024, SENTINELPREPARE(Prepare)), Str(str) { sentinelDeviceSend(this, sizeof(stdlib_system)); }
	int RC;
};

#endif  /* _SENTINEL_STDLIBMSG_H */