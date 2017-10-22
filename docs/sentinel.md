# sentinel

SENTINEL_MSGSIZE 4096
SENTINEL_MSGCOUNT 1
SENTINEL_DEVICEMAPS 1

sentinelMessage:
- Wait - flag to asyc or wait
- OP - operation
- Size - size of message
- Prepare() - method to prepare message for transport

sentinelCommand:
- Magic - magic
- Status:volatile - control flag
- Length - length of data
- Data[...] - data

sentinelMap:
- GetId - current reading location
- SetId:volatile - current writing location, atomicaly incremeted by SENTINEL_MSGSIZE 
- Offset - used for map alignment
- Data[SENTINEL_MSGSIZE*SENTINEL_MSGCOUNT]


sentinelExecutor:
- Next - linked list pointer
- Name - name of executor
- Executor() - attempts to process messages
- Tag - optional data for executor


sentinelContext:
- DeviceMap[SENTINEL_DEVICEMAPS] - sentinelMap(s) used for device
- HostMap - sentinelMap used for host IPC
- HostList - linked list of sentinelExecutor(s) for host processing
- DeviceList - linked list of sentinelExecutor(s) for device processing



# EXAMPLE

enum {
	MODULE_SIMPLE = 100,
	MODULE_STRING,
};

struct module_simple {
	sentinelMessage Base;
	int Value;
	__device__ module_simple(int value)
		: Base(true, MODULE_SIMPLE), Value(value) { sentinelDeviceSend(&Base, sizeof(module_simple)); }
	int RC;
};

struct module_string {
	static __forceinline __device__ char *Prepare(module_string *t, char *data, char *dataEnd, intptr_t offset)
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
	const char *Str;
	__device__ module_string(const char *str)
		: Base(true, MODULE_STRING, 1024, SENTINELPREPARE(Prepare)), Str(str) { sentinelDeviceSend(&Base, sizeof(module_string)); }
	int RC;
};

bool sentinelExecutor(void *tag, sentinelMessage *data, int length, char *(**hostPrepare)(void*,char*,char*,intptr_t))
{
	switch (data->OP) {
	case MODULE_SIMPLE: { module_simple *msg = (module_simple *)data; msg->RC = msg->Value; return true; }
	case MODULE_STRING: { module_string *msg = (module_string *)data; msg->RC = strlen(msg->Str); return true; }
	}
	return false;
}

to call:

module_simple msg(123);
int rc = msg.RC;

module_string msg("123");
int rc = msg.RC;