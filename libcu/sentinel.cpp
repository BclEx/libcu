#include <windows.h>
#include <process.h>
#include <assert.h>
#include <stdio.h>
#include <sentinel.h>
#include <cuda_runtimecu.h>

void sentinelCommand::Dump()
{
	register char *b = Data;
	register int l = Length;
	printf("Command: 0x%x[%d] '", b, l); for (int i = 0; i < l; i++) printf("%02x", b[i] & 0xff); printf("'\n");
}

void sentinelMap::Dump()
{
	register char *b = (char *)this;
	register int l = sizeof(sentinelMap);
	printf("Map: 0x%x[%d] '", b, l); for (int i = 0; i < l; i++) printf("%02x", b[i] & 0xff); printf("'\n");
}

static sentinelContext _ctx;

// HOSTSENTINEL
#if HAS_HOSTSENTINEL

static HANDLE _threadHostHandle = NULL;
static unsigned int __stdcall sentinelHostThread(void *data) 
{
	sentinelContext *ctx = &_ctx;
	sentinelMap *map = ctx->HostMap;
	while (map)
	{
		long id = map->GetId;
		sentinelCommand *cmd = (sentinelCommand *)&map->Data[id%sizeof(map->Data)];
		volatile long *status = (volatile long *)&cmd->Status;
		unsigned int s_;
		while (_threadHostHandle && (s_ = InterlockedCompareExchange((long *)status, 3, 2)) != 2) { /*printf("[%d ]", s_);*/ Sleep(50); } //
		if (!_threadHostHandle) return 0;
		if (cmd->Magic != SENTINEL_MAGIC)
		{
			printf("Bad Sentinel Magic");
			exit(1);
		}
		//map->Dump();
		//cmd->Dump();
		sentinelMessage *msg = (sentinelMessage *)cmd->Data;
		for (sentinelExecutor *exec = _ctx.List; exec && exec->Executor && !exec->Executor(exec->Tag, msg, cmd->Length); exec = exec->Next) { }
		//printf(".");
		*status = (!msg->Wait ? 4 : 0);
		map->GetId += SENTINEL_MSGSIZE;
	}
	return 0;
}

#endif

// DEVICESENTINEL
static HANDLE _threadDeviceHandle[SENTINEL_DEVICEMAPS];
static unsigned int __stdcall sentinelDeviceThread(void *data) 
{
	int threadId = (int)data;
	sentinelContext *ctx = &_ctx; 
	sentinelMap *map = ctx->DeviceMap[threadId];
	while (map)
	{
		long id = map->GetId;
		sentinelCommand *cmd = (sentinelCommand *)&map->Data[id%sizeof(map->Data)];
		volatile long *status = (volatile long *)&cmd->Status;
		unsigned int s_;
		while (_threadDeviceHandle[threadId] && (s_ = InterlockedCompareExchange((long *)status, 3, 2)) != 2) { /*printf("[%d ]", s_);*/ Sleep(50); } //
		if (!_threadDeviceHandle[threadId]) return 0;
		if (cmd->Magic != SENTINEL_MAGIC)
		{
			printf("Bad Sentinel Magic");
			exit(1);
		}
		map->Dump();
		cmd->Dump();
		sentinelMessage *msg = (sentinelMessage *)cmd->Data;
		for (sentinelExecutor *exec = _ctx.List; exec && exec->Executor && !exec->Executor(exec->Tag, msg, cmd->Length); exec = exec->Next) { }
		printf(".");
		*status = (!msg->Wait ? 4 : 0);
		map->GetId += SENTINEL_MSGSIZE;
	}
	return 0;
}

static sentinelExecutor _baseExecutor;
#if HAS_HOSTSENTINEL
static sentinelMap *_sentinelHostMap = nullptr;
static HANDLE _hostMapHandle = NULL;
static int *_hostMap = nullptr;
#endif
static int *_deviceMap[SENTINEL_DEVICEMAPS];

// https://msdn.microsoft.com/en-us/library/windows/desktop/aa366551(v=vs.85).aspx
// https://github.com/pathscale/nvidia_sdk_samples/blob/master/simpleStreams/0_Simple/simpleStreams/simpleStreams.cu
void sentinelServerInitialize(sentinelExecutor *executor, char *mapHostName)
{
	// create host map
#if HAS_HOSTSENTINEL
	_hostMapHandle = CreateFileMapping(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, sizeof(sentinelMap) + MEMORY_ALIGNMENT, mapHostName);
	if (!_hostMapHandle)
	{
		printf("Could not create file mapping object (%d).\n", GetLastError());
		exit(1);
	}
	_hostMap = (int *)MapViewOfFile(_hostMapHandle, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(sentinelMap) + MEMORY_ALIGNMENT);
	if (!_hostMap)
	{
		printf("Could not map view of file (%d).\n", GetLastError());
		CloseHandle(_hostMapHandle);
		exit(1);
	}
	_sentinelHostMap = _ctx.HostMap = (sentinelMap *)ROUNDN(_hostMap, MEMORY_ALIGNMENT);
#endif

	// create device maps
#if HAS_GPU
	sentinelMap *d_deviceMap[SENTINEL_DEVICEMAPS];
	for (int i = 0; i < SENTINEL_DEVICEMAPS; i++)
	{
		cudaErrorCheckF(cudaHostAlloc(&_deviceMap[i], sizeof(sentinelMap), cudaHostAllocPortable | cudaHostAllocMapped), goto initialize_error);
		d_deviceMap[i] = _ctx.DeviceMap[i] = (sentinelMap *)_deviceMap[i];
		cudaErrorCheckF(cudaHostGetDevicePointer(&d_deviceMap[i], _ctx.DeviceMap[i], 0), goto initialize_error);
	}
	cudaErrorCheckF(cudaMemcpyToSymbol(_sentinelDeviceMap, &d_deviceMap, sizeof(d_deviceMap)), goto initialize_error);
#else
	for (int i = 0; i < SENTINEL_DEVICEMAPS; i++)
	{
		//_deviceMap[i] = (int *)VirtualAlloc(NULL, (sizeof(SentinelMap) + MEMORY_ALIGNMENT), MEM_RESERVE|MEM_COMMIT, PAGE_READWRITE);
		_deviceMap[i] = (int *)malloc(sizeof(sentinelMap) + MEMORY_ALIGNMENT);
		_sentinelDeviceMap[i] = _ctx.DeviceMap[i] = (sentinelMap *)ROUNDN(_deviceMap[i], MEMORY_ALIGNMENT);
		if (!_sentinelDeviceMap[i])
		{
			printf("Could not create map.\n");
			goto initialize_error;
		}
		memset(_sentinelDeviceMap[i], 0, sizeof(sentinelMap));
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
	_threadHostHandle = (HANDLE)_beginthreadex(0, 0, sentinelHostThread, nullptr, 0, 0);
#endif
	memset(_threadDeviceHandle, 0, sizeof(_threadDeviceHandle));
	for (int i = 0; i < SENTINEL_DEVICEMAPS; i++)
		_threadDeviceHandle[i] = (HANDLE)_beginthreadex(0, 0, sentinelDeviceThread, (void *)i, 0, 0);
	return;
initialize_error:
	sentinelServerShutdown();
	exit(1);
}

void sentinelServerShutdown()
{
	// close host map
#if HAS_HOSTSENTINEL
	if (_threadHostHandle) { CloseHandle(_threadHostHandle); _threadHostHandle = NULL; }
	if (_hostMap) { UnmapViewOfFile(_hostMap); _hostMap = nullptr; }
	if (_hostMapHandle) { CloseHandle(_hostMapHandle); _hostMapHandle = NULL; }
#endif
	// close device maps
	for (int i = 0; i < SENTINEL_DEVICEMAPS; i++)
	{
		if (_threadDeviceHandle[i]) { CloseHandle(_threadDeviceHandle[i]); _threadDeviceHandle[i] = NULL; }
#if HAS_GPU
		if (_deviceMap[i]) { cudaErrorCheckA(cudaFreeHost(_deviceMap[i])); _deviceMap[i] = nullptr; }
#else
		if (_deviceMap[i]) { free(_deviceMap[i]); /*VirtualFree(_deviceMap[i], 0, MEM_RELEASE);*/ _deviceMap[i] = nullptr; }
#endif
	}
}

#if HAS_HOSTSENTINEL
void sentinelClientInitialize(char *mapHostName)
{
	_hostMapHandle = OpenFileMapping(FILE_MAP_ALL_ACCESS, FALSE, mapHostName);
	if (!_hostMapHandle)
	{
		printf("Could not open file mapping object (%d).\n", GetLastError());
		exit(1);
	}
	_hostMap = (int *)MapViewOfFile(_hostMapHandle, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(sentinelMap) + MEMORY_ALIGNMENT);
	if (!_hostMap)
	{
		printf("Could not map view of file (%d).\n", GetLastError());
		CloseHandle(_hostMapHandle);
		exit(1);
	}
	_sentinelHostMap = _ctx.HostMap = (sentinelMap *)ROUNDN(_hostMap, MEMORY_ALIGNMENT);
}
#endif

#if HAS_HOSTSENTINEL
void sentinelClientShutdown()
{
	if (_hostMap) { UnmapViewOfFile(_hostMap); _hostMap = nullptr; }
	if (_hostMapHandle) { CloseHandle(_hostMapHandle); _hostMapHandle = NULL; }
}
#endif

sentinelExecutor *sentinelFindExecutor(const char *name)
{
	sentinelExecutor *exec = nullptr;
	for (exec = _ctx.List; exec && name && strcmp(name, exec->Name); exec = exec->Next) { }
	return exec;
}

static void sentinelUnlinkExecutor(sentinelExecutor *exec)
{
	if (!exec) { }
	else if (_ctx.List == exec)
		_ctx.List = exec->Next;
	else if (_ctx.List)
	{
		sentinelExecutor *p = _ctx.List;
		while (p->Next && p->Next != exec)
			p = p->Next;
		if (p->Next == exec)
			p->Next = exec->Next;
	}
}

void sentinelRegisterExecutor(sentinelExecutor *exec, bool makeDefault)
{
	sentinelUnlinkExecutor(exec);
	if (makeDefault || !_ctx.List)
	{
		exec->Next = _ctx.List;
		_ctx.List = exec;
	}
	else
	{
		exec->Next = _ctx.List->Next;
		_ctx.List->Next = exec;
	}
	assert(_ctx.List != nullptr);
}

void sentinelUnregisterExecutor(sentinelExecutor *exec)
{
	sentinelUnlinkExecutor(exec);
}