#define _CRT_SECURE_NO_WARNINGS
#include <stdlib.h>
#include <stdio.h>
#include "sentinel-fileutilsmsg.h"

unsigned short _newMode = 0666; // & ~umask(0);

int makeDir(char *name, int f)
{
	char iname[256];
	strcpy(iname, name);

	char *line;
	if ((line = strchr(iname, '/')) && f) {
		while (line > iname && *line == '/')
			--line;
		line[1] = 0;
		makeDir(iname, 1);
	}
	fileutils_dmkdir msg(name, _newMode);
	return (msg.RC && !f ? 1 : 0);
}

int main(int argc, char **argv)
{
	atexit(sentinelClientShutdown);
	sentinelClientInitialize();
	int parent = (argv[1][0] == '-' && argv[1][1] == 'p' ? 1 : 0);

	int r = 0;
	for (int i = parent + 1; i < argc; i++) {
		if (argv[i][0] != '-') {
			if (argv[i][strlen(argv[i])-1] == '/')
				argv[i][strlen(argv[i])-1] = '\0';
			if (makeDir(argv[i], parent)) {
				fwrite("mkdir: cannot create directory ", 31, 1, stderr);
				fwrite(argv[i], strlen(argv[i]), 1, stderr);
				fwrite("\n", 1, 1, stderr);
				r = 1;
			}
		} else {
			fwrite("mkdir: usage error.\n", 20, 1, stderr);
			exit(1);
		}
	}
	exit(r);
}
