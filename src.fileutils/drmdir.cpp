#define _CRT_SECURE_NO_WARNINGS
#include <stdlib.h>
#include <stdio.h>
#include "..\src\libcu.fileutils\sentinel-fileutilsmsg.h"

unsigned short _newMode = 0666; // & ~umask(0);

__forceinline int rmdir(char *name) { fileutils_drmdir msg(name); return msg.RC; }

int removeDir(char *name, int f)
{
	int r, r2 = 2;
	char *line;
	while (!(r = rmdir(name)) && (line = strchr(name,'/')) && f) {
		while (line > name && *line == '/')
			--line;
		line[1] = 0;
		r2 = 0;
	}
	return (r && r2);
}

int main(int argc, char **argv)
{
	atexit(sentinelClientShutdown);
	sentinelClientInitialize();
	int parent = (argv[1][0] == '-' && argv[1][1] == 'p' ? 1 : 0);

	int r = 0;
	for (int i = parent + 1; i < argc; i++) {
		if (argv[i][0] != '-') {
			while (argv[i][strlen(argv[i])-1] == '/')
				argv[i][strlen(argv[i])-1] = '\0';
			if (removeDir(argv[i], parent)) {
				fwrite("rmdir: cannot remove directory ", 31, 1, stderr);
				fwrite(argv[i], strlen(argv[i]), 1, stderr);
				fwrite("\n", 1, 1, stderr);
				r = 1;
			}
		}
		else {
			fwrite("rmdir: usage error.\n", 20, 1, stderr);
			exit(1);
		}
	}
	exit(r);
}
