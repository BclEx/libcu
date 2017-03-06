#ifndef _FSYSTEM_H
#define _FSYSTEM_H
#include <featurescu.h>
#include <fcntl.h>
#include <ext/memfile.h>

__BEGIN_DECLS;

struct dirEnt_t
{
	dirent dir;		// Entry information
	dirEnt_t *next;	// Next entity in the directory
	union {
		dirEnt_t *list;	// List of entities in the directory
		memfile_t *file; // Memory file associated with this element
	} u;
};

__device__ int fsystemRename(const char *old, const char *new_);
__device__ int fsystemUnlink(const char *path);
__device__ dirEnt_t *fsystemMkdir(const char *__restrict path, int mode);
__device__ dirEnt_t *fsystemOpen(const char *__restrict path, int mode);
__device__ void fsystemShutdown(dirEnt_t *dir);

__END_DECLS;
#endif  /* _FSYSTEM_H */