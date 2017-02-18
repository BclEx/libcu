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

#ifndef HAS_DEVICESENTINEL
#define HAS_DEVICESENTINEL 1
#endif
#ifndef HAS_HOSTSENTINEL
#define HAS_HOSTSENTINEL 1
#endif

#ifndef _SENTINEL_H
#define _SENTINEL_H
#include <crtdefscu.h>
#ifdef __cplusplus
extern "C" {
#endif

#define SENTINEL_MAGIC (unsigned short)0xC811
#define SENTINEL_MSGSIZE 4096
#define SENTINEL_MSGCOUNT 1
#define SENTINEL_NAME "Sentinel" //"Global\\Sentinel"
#define SENTINEL_DEVICEMAPS 1

	struct sentinelMessage
	{
		bool Wait;
		char OP;
		int Size;
		char *(*Prepare)(void*,char*,char*,long);
		__device__ sentinelMessage(bool wait, char op, int size = 0, char *(*prepare)(void*,char*,char*,long) = nullptr)
			: Wait(wait), OP(op), Size(size), Prepare(prepare) { }
	public:
	};
#define SENTINELPREPARE(P) ((char *(*)(void*,char*,char*,long))&P)
#ifndef _WIN64
#define SENTINELOFFSET(O) O
#else
#define SENTINELOFFSET(O) 0
#endif

	typedef struct //__align__(16)
	{
		unsigned short Magic;
		volatile long Status;
		int Length;
#ifndef _WIN64
		int Unknown;
#endif
		char *Data;
		void Dump(long offset);
	} sentinelCommand;

	typedef struct //__align__(16)
	{
		long GetId;
		volatile long SetId;
#ifndef _WIN64
		long Offset;
#endif
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
		sentinelExecutor *HostList;
		sentinelExecutor *DeviceList;
	} sentinelContext;

#if HAS_HOSTSENTINEL
	extern sentinelMap *_sentinelHostMap;
#endif
#if HAS_DEVICESENTINEL
	extern __constant__ sentinelMap *_sentinelDeviceMap[SENTINEL_DEVICEMAPS];
#endif

	extern bool sentinelDefaultExecutor(void *tag, sentinelMessage *data, int length);
	extern void sentinelServerInitialize(sentinelExecutor *executor = nullptr, char *mapHostName = SENTINEL_NAME, bool hostSentinel = true, bool deviceSentinel = true);
	extern void sentinelServerShutdown();
#if HAS_DEVICESENTINEL
	extern __device__ void sentinelDeviceSend(sentinelMessage *msg, int msgLength);
#endif
#if HAS_HOSTSENTINEL
	extern void sentinelClientInitialize(char *mapHostName = SENTINEL_NAME);
	extern void sentinelClientShutdown();
	extern void sentinelClientSend(sentinelMessage *msg, int msgLength);
#endif
	extern sentinelExecutor *sentinelFindExecutor(const char *name, bool forDevice = true);
	extern void sentinelRegisterExecutor(sentinelExecutor *exec, bool makeDefault = false, bool forDevice = true);
	extern void sentinelUnregisterExecutor(sentinelExecutor *exec, bool forDevice = true);

#ifdef  __cplusplus
}
#endif
#endif  /* _SENTINEL_H */