#ifndef _FSYSTEM_H
#define _FSYSTEM_H
#include <featurescu.h>
#include <fcntl.h>
#include <ext/memfile.h>
#include <_dirent.h>

__BEGIN_DECLS;

struct dirEnt_t {
	dirent dir;		// Entry information
	dirEnt_t *next;	// Next entity in the directory.
	char *path;		// Path/Key
	union {
		dirEnt_t *list;	// List of entities in the directory
		memfile_t *file; // Memory file associated with this element
	} u;
};

struct file_t {
	char *base;
};

__device__ void expandPath(const char *path, char *newPath);
__device__ dirEnt_t *fsystemOpendir(const char *path);
__device__ int fsystemRename(const char *old, const char *new_);
__device__ int fsystemUnlink(const char *path);
__device__ dirEnt_t *fsystemMkdir(const char *__restrict path, int mode, int *r);
__device__ dirEnt_t *fsystemOpen(const char *__restrict path, int mode, int *fd);
__device__ void fsystemReset();

extern __device__ dirEnt_t __iob_root;
extern __constant__ file_t __iob_files[CORE_MAXFILESTREAM];
#define GETFD(fd) (INT_MAX-(fd))
#define GETFILE(fd) (&__iob_files[GETFD(fd)])

__END_DECLS;
#endif  /* _FSYSTEM_H */