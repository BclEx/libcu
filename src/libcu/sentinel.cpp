#include <windows.h>
#include <process.h>
#include <stdio.h>
#include <sentinel.h>
#include <assert.h>
#include <cuda_runtimecu.h>

void sentinelCommand::Dump()
{
	register char *b = Data;
	register int l = Length;
	printf("Cmd: %d[%d]'", ((sentinelMessage*)Data)->OP, l); for (int i = 0; i < l; i++) printf("%02x", b[i] & 0xff); printf("'\n");
}

void sentinelMap::Dump()
{
	register char *b = (char *)this;
	register int l = sizeof(sentinelMap);
	printf("Map: 0x%x[%d]'", b, l); for (int i = 0; i < l; i++) printf("%02x", b[i] & 0xff); printf("'\n");
}

static sentinelContext _ctx;
static sentinelExecutor _baseExecutor;

// HOSTSENTINEL
#if HAS_HOSTSENTINEL

static sentinelMap *_sentinelHostMap = nullptr;
static HANDLE _hostMapHandle = NULL;
static int *_hostMap = nullptr;
static HANDLE _threadHostHandle = NULL;
static unsigned int __stdcall sentinelHostThread(void *data) 
{
	sentinelContext *ctx = &_ctx;
	sentinelMap *map = ctx->HostMap;
	while (map) {
		long id = map->GetId;
		sentinelCommand *cmd = (sentinelCommand *)&map->Data[id%sizeof(map->Data)];
		volatile long *status = (volatile long *)&cmd->Status;
		unsigned int s_;
		while (_threadHostHandle && (s_ = InterlockedCompareExchange((long *)status, 3, 2)) != 2) { /*printf("(%d)", s_);*/ Sleep(50); } //
		if (!_threadHostHandle) return 0;
		if (cmd->Magic != SENTINEL_MAGIC) {
			printf("Bad Sentinel Magic");
			exit(1);
		}
		//map->Dump();
		//cmd->Dump();
		sentinelMessage *msg = (sentinelMessage *)cmd->Data;
		for (sentinelExecutor *exec = _ctx.HostList; exec && exec->Executor && !exec->Executor(exec->Tag, msg, cmd->Length); exec = exec->Next) { }
		//printf(".");
		*status = (msg->Wait ? 4 : 0);
		map->GetId += SENTINEL_MSGSIZE;
	}
	return 0;
}

#endif

// DEVICESENTINEL
#if HAS_DEVICESENTINEL

#include <sentinel-stdiomsg.h>

static bool _sentinelDevice = false;
static int *_deviceMap[SENTINEL_DEVICEMAPS];
static HANDLE _threadDeviceHandle[SENTINEL_DEVICEMAPS];
static unsigned int __stdcall sentinelDeviceThread(void *data) 
{
	int threadId = (int)data;
	sentinelContext *ctx = &_ctx;
	sentinelMap *map = ctx->DeviceMap[threadId];
	while (map) {
		long id = map->GetId;
		sentinelCommand *cmd = (sentinelCommand *)&map->Data[id%sizeof(map->Data)];
		volatile long *status = (volatile long *)&cmd->Status;
		unsigned int s_;
		while (_threadDeviceHandle[threadId] && (s_ = InterlockedCompareExchange((long *)status, 3, 2)) != 2) { /*printf("(%d)", s_);*/ Sleep(50); }
		if (!_threadDeviceHandle[threadId]) return 0;
		if (cmd->Magic != SENTINEL_MAGIC) {
			printf("Bad Sentinel Magic");
			exit(1);
		}
#ifndef _WIN64
		// x86: must reset Data member after device transfer
		int offset = ((char *)&cmd->Data + 4) - cmd->Data;
		cmd->Data += offset;
#endif
		printf("hst: %x %x %x\n", (char *)&cmd->Data + 4, cmd->Data, ((char *)&cmd->Data + 4) - cmd->Data);
		//map->Dump();
		cmd->Dump();
		sentinelMessage *msg = (sentinelMessage *)cmd->Data;
#ifndef _WIN64
		// x86: must reset Data member after device transfer
		if (msg->Offset) {
			printf("off: %x %x\n", msg->Offset, &stdio_freopen::Offset);
			//msg->Offset(msg, offset);
		}
#endif
		for (sentinelExecutor *exec = _ctx.DeviceList; exec && exec->Executor && !exec->Executor(exec->Tag, msg, cmd->Length); exec = exec->Next) { }
		/*printf(".");*/
		*status = (msg->Wait ? 4 : 0);
		map->GetId += SENTINEL_MSGSIZE;
	}
	return 0;
}

#endif

// https://msdn.microsoft.com/en-us/library/windows/desktop/aa366551(v=vs.85).aspx
// https://github.com/pathscale/nvidia_sdk_samples/blob/master/simpleStreams/0_Simple/simpleStreams/simpleStreams.cu
void sentinelServerInitialize(sentinelExecutor *executor, char *mapHostName, bool hostSentinel, bool deviceSentinel)
{
	// create host map
#if HAS_HOSTSENTINEL
	if (hostSentinel) {
		_hostMapHandle = CreateFileMapping(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, sizeof(sentinelMap) + MEMORY_ALIGNMENT, mapHostName);
		if (!_hostMapHandle) {
			printf("Could not create file mapping object (%d).\n", GetLastError());
			exit(1);
		}
		_hostMap = (int *)MapViewOfFile(_hostMapHandle, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(sentinelMap) + MEMORY_ALIGNMENT);
		if (!_hostMap) {
			printf("Could not map view of file (%d).\n", GetLastError());
			CloseHandle(_hostMapHandle);
			exit(1);
		}
		_sentinelHostMap = _ctx.HostMap = (sentinelMap *)_ROUNDN(_hostMap, MEMORY_ALIGNMENT);
	}
#endif

	// create device maps
#if HAS_DEVICESENTINEL
	if (deviceSentinel) {
		_sentinelDevice = true;
		sentinelMap *d_deviceMap[SENTINEL_DEVICEMAPS];
		for (int i = 0; i < SENTINEL_DEVICEMAPS; i++) {
			cudaErrorCheckF(cudaHostAlloc(&_deviceMap[i], sizeof(sentinelMap), cudaHostAllocPortable | cudaHostAllocMapped), goto initialize_error);
			d_deviceMap[i] = _ctx.DeviceMap[i] = (sentinelMap *)_deviceMap[i];
			cudaErrorCheckF(cudaHostGetDevicePointer(&d_deviceMap[i], _ctx.DeviceMap[i], 0), goto initialize_error);
		}
		cudaErrorCheckF(cudaMemcpyToSymbol(_sentinelDeviceMap, &d_deviceMap, sizeof(d_deviceMap)), goto initialize_error);
	}
#endif

	// register executor
	_baseExecutor.Name = "base";
	_baseExecutor.Executor = sentinelDefaultExecutor;
	_baseExecutor.Tag = nullptr;
	sentinelRegisterExecutor(&_baseExecutor, true);
	if (executor)
		sentinelRegisterExecutor(executor, true);

	// launch threads
#if HAS_HOSTSENTINEL
	if (hostSentinel) {
		_threadHostHandle = (HANDLE)_beginthreadex(0, 0, sentinelHostThread, nullptr, 0, 0);
	}
#endif
#if HAS_DEVICESENTINEL
	if (deviceSentinel) {
		memset(_threadDeviceHandle, 0, sizeof(_threadDeviceHandle));
		for (int i = 0; i < SENTINEL_DEVICEMAPS; i++)
			_threadDeviceHandle[i] = (HANDLE)_beginthreadex(0, 0, sentinelDeviceThread, (void *)i, 0, 0);
	}
#endif
	return;
initialize_error:
	sentinelServerShutdown();
	exit(1);
}

void sentinelServerShutdown()
{
	// close host map
#if HAS_HOSTSENTINEL
	if (_hostMapHandle) { 
		if (_threadHostHandle) { CloseHandle(_threadHostHandle); _threadHostHandle = NULL; }
		if (_hostMap) { UnmapViewOfFile(_hostMap); _hostMap = nullptr; }
		CloseHandle(_hostMapHandle); _hostMapHandle = NULL;
	}
#endif
	// close device maps
#if HAS_DEVICESENTINEL
	if (_sentinelDevice) {
		for (int i = 0; i < SENTINEL_DEVICEMAPS; i++) {
			if (_threadDeviceHandle[i]) { CloseHandle(_threadDeviceHandle[i]); _threadDeviceHandle[i] = NULL; }
			if (_deviceMap[i]) { cudaErrorCheckA(cudaFreeHost(_deviceMap[i])); _deviceMap[i] = nullptr; }
		}
	}
#endif
}

#if HAS_HOSTSENTINEL
void sentinelClientInitialize(char *mapHostName)
{
	_hostMapHandle = OpenFileMapping(FILE_MAP_ALL_ACCESS, FALSE, mapHostName);
	if (!_hostMapHandle) {
		printf("Could not open file mapping object (%d).\n", GetLastError());
		exit(1);
	}
	_hostMap = (int *)MapViewOfFile(_hostMapHandle, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(sentinelMap) + MEMORY_ALIGNMENT);
	if (!_hostMap) {
		printf("Could not map view of file (%d).\n", GetLastError());
		CloseHandle(_hostMapHandle);
		exit(1);
	}
	_sentinelHostMap = _ctx.HostMap = (sentinelMap *)_ROUNDN(_hostMap, MEMORY_ALIGNMENT);
}

void sentinelClientShutdown()
{
	if (_hostMap) { UnmapViewOfFile(_hostMap); _hostMap = nullptr; }
	if (_hostMapHandle) { CloseHandle(_hostMapHandle); _hostMapHandle = NULL; }
}
#endif

sentinelExecutor *sentinelFindExecutor(const char *name, bool forDevice)
{
	sentinelExecutor *list = (forDevice ? _ctx.DeviceList : _ctx.HostList);
	sentinelExecutor *exec = nullptr;
	for (exec = list; exec && name && strcmp(name, exec->Name); exec = exec->Next) { }
	return exec;
}

static void sentinelUnlinkExecutor(sentinelExecutor *exec, bool forDevice)
{
	sentinelExecutor *list = (forDevice ? _ctx.DeviceList : _ctx.HostList);
	if (!exec) { }
	else if (list == exec)
		if (forDevice) _ctx.DeviceList = exec->Next;
		else _ctx.HostList = exec->Next;
	else if (list) {
		sentinelExecutor *p = list;
		while (p->Next && p->Next != exec)
			p = p->Next;
		if (p->Next == exec)
			p->Next = exec->Next;
	}
}

void sentinelRegisterExecutor(sentinelExecutor *exec, bool makeDefault, bool forDevice)
{
	sentinelUnlinkExecutor(exec, forDevice);
	sentinelExecutor *list = (forDevice ? _ctx.DeviceList : _ctx.HostList);
	if (makeDefault || !list) {
		exec->Next = list;
		if (forDevice) _ctx.DeviceList = exec;
		else _ctx.HostList = exec;
	}
	else {
		exec->Next = list->Next;
		list->Next = exec;
	}
}

void sentinelUnregisterExecutor(sentinelExecutor *exec, bool forDevice)
{
	sentinelUnlinkExecutor(exec, forDevice);
}