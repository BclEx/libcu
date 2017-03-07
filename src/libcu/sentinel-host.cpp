#include <windows.h>
#include <stdio.h>
#include <sentinel.h>

#if HAS_HOSTSENTINEL

sentinelMap *_sentinelHostMap = nullptr;
void sentinelClientSend(sentinelMessage *msg, int msgLength)
{
#ifndef _WIN64
	printf("Sentinel currently only works in x64.\n");
	abort();
#else
	sentinelMap *map = _sentinelHostMap;
	if (!map) {
		printf("sentinel: device map not defined. did you start sentinel?\n");
		exit(0);
	}
	int length = msgLength + msg->Size;
	long id = (InterlockedAdd((long *)&map->SetId, SENTINEL_MSGSIZE) - SENTINEL_MSGSIZE);
	sentinelCommand *cmd = (sentinelCommand *)&map->Data[id%sizeof(map->Data)];
	volatile long *status = (volatile long *)&cmd->Status;
	//while (InterlockedCompareExchange((long *)status, 1, 0) != 0) { }
	cmd->Data = (char *)cmd + _ROUND8(sizeof(sentinelCommand));
	cmd->Magic = SENTINEL_MAGIC;
	cmd->Length = msgLength;
	if (msg->Prepare && !msg->Prepare(msg, cmd->Data, cmd->Data+length, map->Offset)) {
		printf("msg too long");
		exit(0);
	}
	memcpy(cmd->Data, msg, msgLength);
	printf("Msg: %d[%d]'", msg->OP, msgLength); for (int i = 0; i < msgLength; i++) printf("%02x", ((char *)msg)[i] & 0xff); printf("'\n");

	*status = 2;
	if (msg->Wait) {
		while (InterlockedCompareExchange((long *)status, 5, 4) != 4) { }
		memcpy(msg, cmd->Data, msgLength);
		*status = 0;
	}
#endif
}

#endif