#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include "sentinel-fileutilsmsg.h"

#define CAT_BUF_SIZE 4096

char colon[2] = { ':', ' ' };

void dumpfile(FILE *f)
{
	int nred;
	static char readbuf[CAT_BUF_SIZE];
	while ((nred = fread(readbuf, 1, CAT_BUF_SIZE, f)) > 0)
		fwrite(readbuf, nred, 1, stdout);
}

int main(int argc, char **argv)
{
	atexit(sentinelClientShutdown);
	sentinelClientInitialize();
	if (argc <= 1)
		dumpfile(stdin);
	else
	{
		for (int i = 1; i < argc; i++)
		{
			fileutils_dcat msg(argv[i]);
			int errno = msg.RC;
			if (!errno)
			{
				fwrite(argv[0], 1, strlen(argv[0]), stderr);
				fwrite(colon, 2, 1, stderr);
				fwrite(argv[i], 1, strlen(argv[i]), stderr);
				fwrite(colon, 1, 2, stderr);
				fwrite(strerror(errno), 1, strlen(strerror(errno)), stderr);
				fwrite("\n", 1, 1, stderr);
			}
		}
	}
	exit(0);
}
