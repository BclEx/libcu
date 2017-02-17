#include <cuda_runtimecu.h>
#include <sentinel.h>

#if HAS_DEVICESENTINEL
__BEGIN_DECLS;

__device__ volatile unsigned int _sentinelMapId;
__constant__ sentinelMap *_sentinelDeviceMap[SENTINEL_DEVICEMAPS];
__device__ void sentinelDeviceSend(void *msg, int msgLength)
{
	//#ifndef _WIN64
	//	printf("Sentinel currently only works in x64.\n");
	//	abort();
	//#endif
	sentinelMap *map = _sentinelDeviceMap[_sentinelMapId++ % SENTINEL_DEVICEMAPS];
	sentinelMessage *msg2 = (sentinelMessage *)msg;
	int length = msgLength + msg2->Size;
	long id = atomicAdd((int *)&map->SetId, SENTINEL_MSGSIZE);
	sentinelCommand *cmd = (sentinelCommand *)&map->Data[id%sizeof(map->Data)];
	volatile long *status = (volatile long *)&cmd->Status;
	cmd->Magic = SENTINEL_MAGIC;
	cmd->Length = msgLength;
	cmd->Data = (char *)cmd + _ROUND8(sizeof(sentinelCommand));
	if (msg2->Prepare && !msg2->Prepare(msg, cmd->Data, cmd->Data+length, map->Offset)) {
		printf("msg too long");
		abort();
	}
	memcpy(cmd->Data, msg2, msgLength);
	printf("Msg: %d[%d]'", msg2->OP, msgLength); for (int i = 0; i < msgLength; i++) printf("%02x", ((char *)msg2)[i] & 0xff); printf("'\n");

	*status = 2;
	if (msg2->Wait) {
		unsigned int s_; do { s_ = *status; /*printf("%d ", s_);*/ __syncthreads(); } while (s_ != 4); __syncthreads();
		memcpy(msg2, cmd->Data, msgLength);
		*status = 0;
	}
}

__END_DECLS;
#endif