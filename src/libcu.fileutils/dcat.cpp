#define _CRT_SECURE_NO_WARNINGS
#include <stdlib.h>
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
	else for (int i = 1; i < argc; i++) {
		fileutils_dcat msg(argv[i]);
		int r = msg.RC;
		if (!r)
		{
			fwrite(argv[0], strlen(argv[0]), 1, stderr);
			fwrite(colon, 2, 1, stderr);
			fwrite(argv[i], strlen(argv[i]), 1, stderr);
			fwrite(colon, 2, 1, stderr);
			fwrite(strerror(r), strlen(strerror(r)), 1, stderr);
			fwrite("\n", 1, 1, stderr);
		}
	}
	exit(0);
}
