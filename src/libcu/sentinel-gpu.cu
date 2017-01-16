#include <cuda_runtimecu.h>
#include <sentinel.h>
#ifdef __CUDA_ARCH
#if HAS_GPU

__device__ volatile unsigned int _sentinelMapId;
__constant__ sentinelMap *_sentinelDeviceMap[SENTINEL_DEVICEMAPS];
__device__ void sentinelSend(void *msg, int msgLength)
{
#ifndef _WIN64
	printf("Sentinel currently only works in x64.\n");
	abort();
#endif
	sentinelMap *map = _sentinelDeviceMap[_sentinelMapId++ % SENTINEL_DEVICEMAPS];
	sentinelMessage *msg2 = (sentinelMessage *)msg;
	int length = msgLength + msg2->Size;
	long id = atomicAdd((int *)&map->SetId, SENTINEL_MSGSIZE);
	sentinelCommand *cmd = (sentinelCommand *)&map->Data[id%sizeof(map->Data)];
	volatile long *status = (volatile long *)&cmd->Status;
	//while (atomicCAS((unsigned int *)status, 1, 0) != 0) { __syncthreads(); }
	cmd->Data = (char *)cmd + _ROUND8(sizeof(sentinelCommand));
	cmd->Magic = SENTINEL_MAGIC;
	cmd->Length = msgLength;
	if (msg2->Prepare && !msg2->Prepare(msg, cmd->Data, cmd->Data+length)) {
		printf("msg too long");
		abort();
	}
	memcpy(cmd->Data, msg, msgLength);
	//printf("Msg: %d[%d]'", msg2->OP, msgLength); for (int i = 0; i < msgLength; i++) printf("%02x", ((char *)cmd->Data)[i] & 0xff); printf("'\n");

	*status = 2;
	if (msg2->Wait) {
		unsigned int s_; do { s_ = *status; /*printf("%d ", s_);*/ __syncthreads(); } while (s_ != 4); __syncthreads();
		memcpy(msg, cmd->Data, msgLength);
		*status = 0;
	}
}

#endif
#endif