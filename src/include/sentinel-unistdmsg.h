/*
sentinel-unistdmsg.h - messages for sentinel
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
#ifndef _SENTINEL_UNISTDMSG_H
#define _SENTINEL_UNISTDMSG_H

#include <sentinel.h>
#include <crtdefscu.h>
#include <stringcu.h>

enum {
	UNISTD_ACCESS = 35,
	UNISTD_LSEEK,
	//UNISTD_LSEEK64,
	UNISTD_CLOSE,
	UNISTD_READ,
	UNISTD_WRITE,
};

struct unistd_access
{
	static __forceinline __device__ char *Prepare(unistd_access *t, char *data, char *dataEnd, intptr_t offset)
	{
		int nameLength = (t->Name ? (int)strlen(t->Name) + 1 : 0);
		char *name = (char *)(data += _ROUND8(sizeof(*t)));
		char *end = (char *)(data += nameLength);
		if (end > dataEnd) return nullptr;
		memcpy(name, t->Name, nameLength);
		t->Name = name + offset;
		return end;
	}
	sentinelMessage Base;
	const char *Name; int Type;
	__device__ unistd_access(const char *name, int type)
		: Base(true, UNISTD_ACCESS, 1024, SENTINELPREPARE(Prepare)), Name(name), Type(type) { sentinelDeviceSend(&Base, sizeof(unistd_access)); }
	int RC;
};

struct unistd_lseek
{
	sentinelMessage Base;
	int Handle; long Offset; int Whence;
	__device__ unistd_lseek(int fd, long offset, int whence)
		: Base(true, UNISTD_LSEEK), Handle(fd), Offset(offset), Whence(whence) { sentinelDeviceSend(&Base, sizeof(unistd_lseek)); }
	long RC;
};

struct unistd_close
{
	sentinelMessage Base;
	int Handle;
	__device__ unistd_close(int fd)
		: Base(true, UNISTD_CLOSE), Handle(fd) { sentinelDeviceSend(&Base, sizeof(unistd_close)); }
	int RC;
};

struct unistd_read
{
	static __forceinline __device__ char *Prepare(unistd_read *t, char *data, char *dataEnd, intptr_t offset)
	{
		t->Ptr = (char *)(data += _ROUND8(sizeof(*t)));
		char *end = (char *)(data += 1024);
		if (end > dataEnd) return nullptr;
		return end;
	}
	sentinelMessage Base;
	int Handle; void *Ptr; size_t Size;
	__device__ unistd_read(bool wait, int fd, void *buf, size_t nbytes)
		: Base(wait, UNISTD_READ, 1024, SENTINELPREPARE(Prepare)), Handle(fd), Ptr(buf), Size(nbytes) { sentinelDeviceSend(&Base, sizeof(unistd_read)); }
	size_t RC;
};

struct unistd_write
{
	static __forceinline __device__ char *Prepare(unistd_write *t, char *data, char *dataEnd, intptr_t offset)
	{
		size_t size = t->Size;
		char *ptr = (char *)(data += _ROUND8(sizeof(*t)));
		char *end = (char *)(data += size);
		if (end > dataEnd) return nullptr;
		memcpy(ptr, t->Ptr, size);
		t->Ptr = (char *)ptr + offset;
		return end;
	}
	sentinelMessage Base;
	int Handle; const void *Ptr; size_t Size;
	__device__ unistd_write(bool wait, int fd, const void *buf, size_t n)
		: Base(wait, UNISTD_WRITE, 1024, SENTINELPREPARE(Prepare)), Handle(fd), Ptr(buf), Size(n) { sentinelDeviceSend(&Base, sizeof(unistd_write)); }
	size_t RC;
};

#endif  /* _SENTINEL_UNISTDMSG_H */