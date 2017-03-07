#define _CRT_SECURE_NO_WARNINGS
#include "sentinel-fileutilsmsg.h"

#define panic(fmt, ...) { printf(fmt, __VA_ARGS__); exit(1); }

int dcat(char *str);
int dmkdir(char *str, unsigned short mode);
int drmdir(char *str);

bool sentinelFileUtilsExecutor(void *tag, sentinelMessage *data, int length)
{
	switch (data->OP) {
	case FILEUTILS_DCAT: { fileutils_dcat *msg = (fileutils_dcat *)data; msg->RC = dcat(msg->Str); return true; }
	case FILEUTILS_DMKDIR: { fileutils_dmkdir *msg = (fileutils_dmkdir *)data; msg->RC = dmkdir(msg->Str, msg->Mode); return true; }
	case FILEUTILS_DRMDIR: { fileutils_drmdir *msg = (fileutils_drmdir *)data; msg->RC = drmdir(msg->Str); return true; }
	}
	return false;
}