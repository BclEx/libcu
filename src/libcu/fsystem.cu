#include <stdiocu.h>
#include <stdlibcu.h>
#include <stddefcu.h>
#include <assert.h>
#include <ext/hash.h>
#include <errnocu.h>
#include "fsystem.h"

__BEGIN_DECLS;

// FILES
#pragma region FILES

typedef struct __align__(8)
{
	file_t *file;			// reference
	unsigned short id;		// ID of author
	unsigned short threadid;// thread ID of author
} fileRef;

__device__ fileRef __iob_fileRefs[CORE_MAXFILESTREAM]; // Start of circular buffer (set up by host)
volatile __device__ fileRef *__iob_freeFilePtr = __iob_fileRefs; // Current atomically-incremented non-wrapped offset
volatile __device__ fileRef *__iob_retnFilePtr = __iob_fileRefs; // Current atomically-incremented non-wrapped offset
__constant__ file_t __iob_files[CORE_MAXFILESTREAM];

static __device__ __forceinline void writeFileRef(fileRef *ref, file_t *f)
{
	ref->file = f;
	ref->id = gridDim.x*blockIdx.y + blockIdx.x;
	ref->threadid = blockDim.x*blockDim.y*threadIdx.z + blockDim.x*threadIdx.y + threadIdx.x;
}

static __device__ int fileGet(file_t **file)
{
	// advance circular buffer
	size_t offset = (atomicAdd((uintptr_t *)&__iob_freeFilePtr, sizeof(fileRef)) - (size_t)&__iob_fileRefs);
	offset %= (sizeof(fileRef)*CORE_MAXFILESTREAM);
	int offsetId = offset / sizeof(fileRef);
	fileRef *ref = (fileRef *)((char *)&__iob_fileRefs + offset);
	file_t *f = ref->file;
	if (!f) {
		f = &__iob_files[offsetId];
		writeFileRef(ref, f);
	}
	*file = f;
	return GETFD(offsetId);
}

static __device__ void fileFree(int fd)
{
	//if (!f) return;
	file_t *f = GETFILE(fd);
	// advance circular buffer
	size_t offset = atomicAdd((uintptr_t *)&__iob_retnFilePtr, sizeof(fileRef)) - (size_t)&__iob_fileRefs;
	offset %= (sizeof(fileRef)*CORE_MAXFILESTREAM);
	fileRef *ref = (fileRef *)((char *)&__iob_fileRefs + offset);
	writeFileRef(ref, f);
}

#pragma endregion

__device__ char __cwd[MAX_PATH] = ":\\";
__device__ dirEnt_t __iob_root = { { 0, 0, 0, 1, ":\\" }, nullptr, nullptr };
__device__ hash_t __iob_dir = HASHINIT;

__device__ void expandPath(const char *path, char *newPath)
{
	register unsigned char *d = (unsigned char *)newPath;
	register unsigned char *s;
	// add cwd
	if (path[0] != ':') {
		s = (unsigned char *)__cwd;
		while (*s) { *d++ = *s++; }
	}
	// add path
	s = (unsigned char *)path;
	int i = 0;
	while (*s) {
		int c = *s;
		if (c == '/') c = '\\'; // switch from unix path
		if (c == '\\') {
			// directory reached
			if (i == 1 && s[-1] == '.') d -= 2; // self directory
			else if (i == 2 && s[-1] == '.' && s[-2] == '.') while (*d >= *newPath && *d != '\\') *d--; // parent directory
			i = 0;
		}
		// advance
		*d++ = c; s++; i++;
	}
	// remove trailing '\'
	d[*d == '\\' ? 0 : 1] = 0;
}

static __device__ void freeEnt(dirEnt_t *ent)
{
	if (ent->dir.d_type == 1) {
		dirEnt_t *p = ent->u.list;
		while (p) {
			dirEnt_t *next = p->next;
			freeEnt(p);
			p = next;
		}
	} else if (ent->dir.d_type == 2)
		memfileClose(ent->u.file);
	if (ent != &__iob_root) free(ent);
	else __iob_root.u.list = nullptr;
}

static __device__ dirEnt_t *findDir(const char *path, const char **file)
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
	char newPath[MAX_PATH]; expandPath(old, newPath);
	dirEnt_t *ent = (dirEnt_t *)hashFind(&__iob_dir, old);
	if (!ent) {
		_set_errno(ENOENT);
		return -1;
	}
	return 0;
}

__device__ int fsystemUnlink(const char *path)
{
	char newPath[MAX_PATH]; expandPath(path, newPath);
	dirEnt_t *ent = (dirEnt_t *)hashFind(&__iob_dir, newPath);
	if (!ent) {
		_set_errno(ENOENT);
		return -1;
	}
	const char *name;
	dirEnt_t *parentEnt = findDir(newPath, &name);
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
	freeEnt(ent);
	return 0;
}

__device__ dirEnt_t *fsystemMkdir(const char *__restrict path, int mode, int *r)
{
	char newPath[MAX_PATH]; expandPath(path, newPath);
	dirEnt_t *dirEnt = (dirEnt_t *)hashFind(&__iob_dir, newPath);
	if (dirEnt) {
		*r = 1;
		return dirEnt;
	}
	const char *name;
	dirEnt_t *parentEnt = findDir(newPath, &name);
	if (!parentEnt) {
		_set_errno(ENOENT);
		*r = -1;
		return nullptr;
	}
	// create directory
	dirEnt = (dirEnt_t *)malloc(sizeof(dirEnt_t));
	if (hashInsert(&__iob_dir, newPath, dirEnt))
		panic("removed directory");
	dirEnt->dir.d_type = 1;
	strcpy(dirEnt->dir.d_name, name);
	// add to directory
	dirEnt->next = parentEnt->u.list; parentEnt->u.list = dirEnt;
	*r = 0;
	return dirEnt;
}

__device__ dirEnt_t *fsystemOpen(const char *__restrict path, int mode, int *fd)
{
	char newPath[MAX_PATH]; expandPath(path, newPath);
	dirEnt_t *fileEnt = (dirEnt_t *)hashFind(&__iob_dir, newPath);
	if (fileEnt) {
		file_t *f; *fd = fileGet(&f);
		f->base = (char *)fileEnt;
		return fileEnt;
	}
	if ((mode & 0xF) == O_RDONLY) {
		_set_errno(EINVAL); // So illegal mode.
		*fd = -1;
		return nullptr;
	}
	const char *name;
	dirEnt_t *parentEnt = findDir(newPath, &name);
	if (!parentEnt) {
		_set_errno(ENOENT);
		*fd = -1;
		return nullptr;
	}
	// create file
	fileEnt = (dirEnt_t *)malloc(_ROUND64(sizeof(dirEnt_t)) + __sizeofMemfile_t);
	if (hashInsert(&__iob_dir, newPath, fileEnt))
		panic("removed file");
	fileEnt->dir.d_type = 2;
	strcpy(fileEnt->dir.d_name, name);
	fileEnt->u.file = (memfile_t *)((char *)fileEnt + _ROUND64(sizeof(dirEnt_t)));
	memfileOpen(fileEnt->u.file);
	// add to directory
	fileEnt->next = parentEnt->u.list; parentEnt->u.list = fileEnt;
	// set to file
	file_t *f; *fd = fileGet(&f);
	f->base = (char *)fileEnt;
	return fileEnt;
}

__device__ void fsystemReset()
{
	freeEnt(&__iob_root);
}

__END_DECLS;