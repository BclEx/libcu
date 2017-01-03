/*
sentinel.h - lite message bus framework for device to host functions
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

#define HAS_GPU 1
#define HAS_HOSTSENTINEL 0

#if !defined(_INC_SENTINEL)
#define _INC_SENTINEL
#include <crtdefscu.h>

#define SENTINEL_MAGIC (unsigned short)0xC811
#define SENTINEL_MSGSIZE 4096
#define SENTINEL_MSGCOUNT 1
//#define SENTINEL_NAME "Global\\Sentinel"
#define SENTINEL_NAME "Sentinel"
#define SENTINEL_DEVICEMAPS 1

struct sentinelMessage
{
	bool Wait;
	char OP;
	int Size;
	char *(*Prepare)(void*,char*,char*);
	__device__ sentinelMessage(bool wait, char op, int size, char *(*prepare)(void*,char*,char*))
		: Wait(wait), OP(op), Size(size), Prepare(prepare) { }
public:
};
#define SENTINELPREPARE(P) ((char *(*)(void*,char*,char*))&P)

typedef struct
{
	unsigned short Magic;
	volatile long Status;
	int Length;
	char *Data;
	void Dump();
} sentinelCommand;

typedef struct
{
	long GetId;
	volatile long SetId;
	char Data[SENTINEL_MSGSIZE*SENTINEL_MSGCOUNT];
	void Dump();
} sentinelMap;

typedef struct sentinelExecutor
{
	sentinelExecutor *Next;
	const char *Name;
	bool (*Executor)(void*,sentinelMessage*,int);
	void *Tag;
} sentinelExecutor;

typedef struct sentinelContext
{
	sentinelMap *DeviceMap[SENTINEL_DEVICEMAPS];
	sentinelMap *HostMap;
	sentinelExecutor *List;
} sentinelContext;

#if HAS_HOSTSENTINEL
extern sentinelMap *_sentinelHostMap;
#endif
extern __constant__ sentinelMap *_sentinelDeviceMap[SENTINEL_DEVICEMAPS];

extern bool sentinelDefaultExecutor(void *tag, sentinelMessage *data, int length);
extern void sentinelServerInitialize(sentinelExecutor *executor = nullptr, char *mapHostName = SENTINEL_NAME); 
extern void sentinelServerShutdown();
#if HAS_HOSTSENTINEL
extern void sentinelClientInitialize(char *mapHostName = SENTINEL_NAME);
extern void sentinelClientShutdown();
#endif
//
extern sentinelExecutor *sentinelFindExecutor(const char *name);
extern void sentinelRegisterExecutor(sentinelExecutor *exec, bool makeDefault = false);
extern void sentinelUnregisterExecutor(sentinelExecutor *exec);
//
extern __device__ void sentinelSend(void *msg, int msgLength);

#endif  /* _INC_SENTINEL */