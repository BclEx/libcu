#include <stdiocu.h>
#include <stdlibcu.h>
#include <stddefcu.h>
#include <assert.h>
#include <ext/hash.h>
#include <ext/memfile.h>
#include <_dirent.h>
#include <errnocu.h>
#include "fsystem.h"

__BEGIN_DECLS;

__device__ dirEnt_t __iob_root = { { 0, 0, 1, 0, ":\\" }, nullptr };
__device__ hash_t __iob_dir = HASHINIT;

__device__ dirEnt_t *findDir(const char *path, const char **file)
{
	char *file2 = strrchr((char *)path, '\\');
	if (!file2) {
		_set_errno(EINVAL);
		return nullptr;
	}
	*file2 = 0;
	dirEnt_t *ent = !strcmp(path, ":")
		? &__iob_root 
		: (dirEnt_t *)hashFind(&__iob_dir, path);
	*file2 = '\\';
	*file = file2 + 1;
	return ent;
}

__device__ int fsystemRename(const char *old, const char *new_)
{
	dirEnt_t *ent = (dirEnt_t *)hashFind(&__iob_dir, old);
	if (!ent) {
		_set_errno(ENOENT);
		return -1;
	}
	return 0;
}

__device__ int fsystemUnlink(const char *path)
{
	dirEnt_t *ent = (dirEnt_t *)hashFind(&__iob_dir, path);
	if (!ent) {
		_set_errno(ENOENT);
		return -1;
	}
	const char *name;
	dirEnt_t *parentEnt = findDir(path, &name);
	if (!parentEnt) {
		_set_errno(ENOENT);
		return -1;
	}

	//// directory not empty
	//if (ent->dir.d_type == 1 && ent->list) {
	//	_set_errno(ENOENT);
	//	return -1;
	//}

	// remove from directory
	dirEnt_t *list = parentEnt->u.list;
	if (list == ent)
		parentEnt->u.list = ent->next;
	else if (list) {
		dirEnt_t *p = list;
		while (p->next && p->next != ent)
			p = p->next;
		if (p->next == ent)
			p->next = ent->next;
	}

	// free entity
	return 0;
}

__device__ dirEnt_t *fsystemMkdir(const char *__restrict path, int mode)
{
	dirEnt_t *dirEnt = (dirEnt_t *)hashFind(&__iob_dir, path);
	if (!dirEnt) {
		const char *name;
		dirEnt_t *parentEnt = findDir(path, &name);
		if (!parentEnt) {
			_set_errno(ENOENT);
			return nullptr;
		}
		// create directory
		dirEnt = (dirEnt_t *)malloc(sizeof(dirEnt_t));
		if (hashInsert(&__iob_dir, path, dirEnt))
			panic("removed directory");
		dirEnt->dir.d_type = 1;
		strcpy(dirEnt->dir.d_name, name);
		// add to directory
		dirEnt->next = parentEnt->u.list; parentEnt->u.list = dirEnt;
	}
	return dirEnt;
}

__device__ dirEnt_t *fsystemOpen(const char *__restrict path, int mode)
{
	dirEnt_t *fileEnt = (dirEnt_t *)hashFind(&__iob_dir, path);
	if (!fileEnt) {
		if ((mode & O_RDONLY)) {
			_set_errno(EINVAL); // So illegal mode.
			return nullptr;
		}
		const char *name;
		dirEnt_t *parentEnt = findDir(path, &name);
		if (!parentEnt) {
			_set_errno(ENOENT);
			return nullptr;
		}
		// create file
		fileEnt = (dirEnt_t *)malloc(_ROUND64(sizeof(dirEnt_t)) + __sizeofMemfile_t);
		if (hashInsert(&__iob_dir, path, fileEnt))
			panic("removed file");
		fileEnt->dir.d_type = 2;
		strcpy(fileEnt->dir.d_name, name);
		fileEnt->u.file = (memfile_t *)((char *)fileEnt + _ROUND64(sizeof(dirEnt_t)));
		memfileOpen(fileEnt->u.file);
		// add to directory
		fileEnt->next = parentEnt->u.list; parentEnt->u.list = fileEnt;
	}
	return fileEnt;
}

__device__ void fsystemShutdown(dirEnt_t *dir)
{
}

__END_DECLS;