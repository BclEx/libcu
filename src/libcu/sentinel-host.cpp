#include <windows.h>
#include <stdio.h>
#include <sentinel.h>

#if !HAS_GPU

volatile unsigned int _sentinelMapId;
sentinelMap *_sentinelDeviceMap[SENTINEL_DEVICEMAPS];
void sentinelSend(void *msg, int msgLength)
{
#ifndef _WIN64
	printf("Sentinel currently only works in x64.\n");
	abort();
#else
	sentinelMap *map = _sentinelDeviceMap[_sentinelMapId++ % SENTINEL_DEVICEMAPS];
	sentinelMessage *msg2 = (sentinelMessage *)msg;
	int length = msgLength + msg2->Size;
	long id = (InterlockedAdd((long *)&map->SetId, SENTINEL_MSGSIZE) - SENTINEL_MSGSIZE);
	sentinelCommand *cmd = (sentinelCommand *)&map->Data[id%sizeof(map->Data)];
	volatile long *status = (volatile long *)&cmd->Status;
	while (InterlockedCompareExchange((long *)status, 1, 0) != 0) { }
	cmd->Data = (char *)cmd + _ROUND8(sizeof(sentinelCommand));
	cmd->Magic = SENTINEL_MAGIC;
	cmd->Length = msgLength;
	if (msg2->Prepare && !msg2->Prepare(msg, cmd->Data, cmd->Data+length)) {
		printf("msg too long");
		exit(0);
	}
	memcpy(cmd->Data, msg, msgLength);
	*status = 2;
	if (msg2->Wait) {
		while (InterlockedCompareExchange((long *)status, 5, 4) != 4) { }
		memcpy(msg, cmd->Data, msgLength);
		*status = 0;
	}
#endif
}

#endif