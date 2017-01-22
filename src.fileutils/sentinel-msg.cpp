#define _CRT_SECURE_NO_WARNINGS
#include "sentinel-fileutilsmsg.h"

#define panic(fmt, ...) { printf(fmt, __VA_ARGS__); exit(1); }

int dcat(char *str);

bool sentinelDefaultExecutor(void *tag, sentinelMessage *data, int length)
{
	switch (data->OP) {
	case FILEUTILS_DCAT: { fileutils_dcat *msg = (fileutils_dcat *)data; msg->RC = dcat(msg->Str); return true; }
	}
	return false;
}