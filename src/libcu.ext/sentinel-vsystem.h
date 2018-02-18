#ifndef _SENTINEL_VSYSTEM_H
#define _SENTINEL_VSYSTEM_H
#include <sentinel.h>
#include <crtdefscu.h>
#include <stringcu.h>
#include <ext\vsystem.h>
#undef RC

enum {
	VSYSFILE_CLOSE = 500,
	VSYSFILE_READ,
	VSYSFILE_WRITE,
	VSYSFILE_TRUNCATE,
	VSYSFILE_SYNC,
	VSYSFILE_FILESIZE,
	VSYSFILE_LOCK,
	VSYSFILE_UNLOCK,
	VSYSFILE_CHECKRESERVEDLOCK,
	VSYSFILE_FILECONTROL,
	VSYSFILE_SECTORSIZE,
	VSYSFILE_DEVICECHARACTERISTICS,
	VSYSFILE_SHMMAP,
	VSYSFILE_SHMLOCK,
	VSYSFILE_SHMBARRIER,
	VSYSFILE_SHMUNMAP,
	VSYSFILE_FETCH,
	VSYSFILE_UNFETCH,
	VSYSTEM_OPEN,
	VSYSTEM_DELETE,
	VSYSTEM_ACCESS,
	VSYSTEM_FULLPATHNAME,
	VSYSTEM_DLOPEN,
	VSYSTEM_DLERROR,
	VSYSTEM_DLSYM,
	VSYSTEM_DLCLOSE,
	VSYSTEM_RANDOMNESS,
	VSYSTEM_SLEEP,
	VSYSTEM_CURRENTTIME,
	VSYSTEM_GETLASTERROR,
	VSYSTEM_CURRENTTIMEINT64,
	VSYSTEM_SETSYSTEMCALL,
	VSYSTEM_GETSYSTEMCALL,
	VSYSTEM_NEXTSYSTEMCALL,
};

struct vsysfile_close {
	sentinelMessage Base;
	vsysfile *F;
	__device__ vsysfile_close(vsysfile *f)
		: Base(true, VSYSFILE_CLOSE), F(f) { sentinelDeviceSend(&Base, sizeof(vsysfile_close)); }
	int RC;
};

struct vsysfile_read {
	static __forceinline__ __device__ char *Prepare(vsysfile_read *t, char *data, char *dataEnd, intptr_t offset) {
		char *ptr = (char *)(data += ROUND8_(sizeof(*t)));
		char *end = (char *)(data += 1024);
		if (end > dataEnd) return nullptr;
		t->Ptr = ptr + offset;
		return end;
	}
	static __forceinline__ __device__ bool Postfix(vsysfile_read *t, intptr_t offset)
	{
		char *ptr = (char *)t->Ptr - offset;
		memcpy(t->Buf, ptr, t->Amount);
		return true;
	}
	sentinelMessage Base;
	vsysfile *F; void *Buf; int Amount; int64_t Offset;
	__device__ vsysfile_read(vsysfile *f, void *buf, int amount, int64_t offset)
		: Base(true, VSYSFILE_READ, 1024, SENTINELPREPARE(Prepare), SENTINELPOSTFIX(Postfix)), F(f), Buf(buf), Amount(amount), Offset(offset) { sentinelDeviceSend(&Base, sizeof(vsysfile_read)); }
	int RC;
	void *Ptr;
};

struct vsysfile_write {
	static __forceinline__ __device__ char *Prepare(vsysfile_write *t, char *data, char *dataEnd, intptr_t offset) {
		char *buf = (char *)(data += ROUND8_(sizeof(*t)));
		char *end = (char *)(data += 1024);
		if (end > dataEnd) return nullptr;
		memcpy(buf, t->Buf, t->Amount);
		t->Buf = buf + offset;
		return end;
	}
	sentinelMessage Base;
	vsysfile *F; const void *Buf; int Amount; int64_t Offset;
	__device__ vsysfile_write(vsysfile *f, const void *buf, int amount, int64_t offset)
		: Base(true, VSYSFILE_WRITE, 1024, SENTINELPREPARE(Prepare)), F(f), Buf(buf), Amount(amount), Offset(offset) { sentinelDeviceSend(&Base, sizeof(vsysfile_write)); }
	int RC;
};

struct vsysfile_truncate {
	sentinelMessage Base;
	vsysfile *F; int64_t Size;
	__device__ vsysfile_truncate(vsysfile *f, int64_t size)
		: Base(true, VSYSFILE_TRUNCATE), F(f), Size(size) { sentinelDeviceSend(&Base, sizeof(vsysfile_truncate)); }
	int RC;
};

struct vsysfile_sync {
	sentinelMessage Base;
	vsysfile *F; int Flags;
	__device__ vsysfile_sync(vsysfile *f, int flags)
		: Base(true, VSYSFILE_SYNC), F(f), Flags(flags) { sentinelDeviceSend(&Base, sizeof(vsysfile_sync)); }
	int RC;
};

struct vsysfile_fileSize {
	sentinelMessage Base;
	vsysfile *F;
	__device__ vsysfile_fileSize(vsysfile *f)
		: Base(true, VSYSFILE_FILESIZE), F(f) { sentinelDeviceSend(&Base, sizeof(vsysfile_fileSize)); }
	int64_t Size;
	int RC;
};

struct vsysfile_lock {
	sentinelMessage Base;
	vsysfile *F; int Lock;
	__device__ vsysfile_lock(vsysfile *f, int lock)
		: Base(true, VSYSFILE_LOCK), F(f), Lock(lock) { sentinelDeviceSend(&Base, sizeof(vsysfile_lock)); }
	int RC;
};

struct vsysfile_unlock {
	sentinelMessage Base;
	vsysfile *F; int Lock;
	__device__ vsysfile_unlock(vsysfile *f, int lock)
		: Base(true, VSYSFILE_UNLOCK), F(f), Lock(lock) { sentinelDeviceSend(&Base, sizeof(vsysfile_unlock)); }
	int RC;
};

struct vsysfile_checkReservedLock {
	sentinelMessage Base;
	vsysfile *F;
	__device__ vsysfile_checkReservedLock(vsysfile *f)
		: Base(true, VSYSFILE_CHECKRESERVEDLOCK), F(f) { sentinelDeviceSend(&Base, sizeof(vsysfile_checkReservedLock)); }
	int Lock;
	int RC;
};

// fileControl
// sectorSize
// deviceCharacteristics
// shmMap
// shmLock
// shmBarrier
// shmUnmap
// fetch
// unfetch

struct vsystem_open {
	static __forceinline__ __device__ char *Prepare(vsystem_open *t, char *data, char *dataEnd, intptr_t offset) {
		// filenames are double-zero terminated if they are not URIs with parameters.
		int nameLength;
		if (t->Name) { nameLength = (int)strlen(t->Name) + 1; nameLength += (int)strlen((const char *)t->Name[nameLength]) + 1; }
		else nameLength = 0;
		char *name = (char *)(data += ROUND8_(sizeof(*t)));
		char *end = (char *)(data += nameLength);
		if (end > dataEnd) return nullptr;
		memcpy(name, t->Name, nameLength);
		t->Name = name + offset;
		return end;
	}
	sentinelMessage Base;
	const char *Name; int Flags;
	__device__ vsystem_open(const char *name, int flags)
		: Base(true, VSYSTEM_OPEN, 1024, SENTINELPREPARE(Prepare)), Name(name), Flags(flags) { sentinelDeviceSend(&Base, sizeof(vsystem_open)); }
	vsysfile *F;
	int OutFlags;
	int RC;
};

struct vsystem_delete {
	static __forceinline__ __device__ char *Prepare(vsystem_delete *t, char *data, char *dataEnd, intptr_t offset) {
		int filenameLength = t->Filename ? (int)strlen(t->Filename) + 1 : 0;
		char *filename = (char *)(data += ROUND8_(sizeof(*t)));
		char *end = (char *)(data += filenameLength);
		if (end > dataEnd) return nullptr;
		memcpy(filename, t->Filename, filenameLength);
		t->Filename = filename + offset;
		return end;
	}
	sentinelMessage Base;
	const char *Filename; bool SyncDir;
	__device__ vsystem_delete(const char *filename, bool syncDir)
		: Base(true, VSYSTEM_DELETE, 1024, SENTINELPREPARE(Prepare)), Filename(filename), SyncDir(syncDir) { sentinelDeviceSend(&Base, sizeof(vsystem_delete)); }
	int RC;
};

struct vsystem_access {
	static __forceinline__ __device__ char *Prepare(vsystem_access *t, char *data, char *dataEnd, intptr_t offset) {
		int filenameLength = t->Filename ? (int)strlen(t->Filename) + 1 : 0;
		char *filename = (char *)(data += ROUND8_(sizeof(*t)));
		char *end = (char *)(data += filenameLength);
		if (end > dataEnd) return nullptr;
		memcpy(filename, t->Filename, filenameLength);
		t->Filename = filename + offset;
		return end;
	}
	sentinelMessage Base;
	const char *Filename; int Flags;
	__device__ vsystem_access(const char *filename, int flags)
		: Base(true, VSYSTEM_ACCESS, 1024, SENTINELPREPARE(Prepare)), Filename(filename), Flags(flags) { sentinelDeviceSend(&Base, sizeof(vsystem_access)); }
	int ResOut;
	int RC;
};

struct vsystem_fullPathname {
	static __forceinline__ __device__ char *Prepare(vsystem_fullPathname *t, char *data, char *dataEnd, intptr_t offset) {
		int relativeLength = t->Relative ? (int)strlen(t->Relative) + 1 : 0;
		char *relative = (char *)(data += ROUND8_(sizeof(*t)));
		char *ptr = (char *)(data += relativeLength);
		char *end = (char *)(data += 1024);
		if (end > dataEnd) return nullptr;
		memcpy(relative, t->Relative, relativeLength);
		t->Relative = relative + offset;
		t->Ptr = ptr + offset;
		return end;
	}
	static __forceinline__ __device__ bool Postfix(vsystem_fullPathname *t, intptr_t offset)
	{
		char *ptr = (char *)t->Ptr - offset;
		memcpy(t->Full, ptr, strlen(t->Full));
		return true;
	}
	sentinelMessage Base;
	const char *Relative; int FullLength; char *Full;
	__device__ vsystem_fullPathname(const char *relative, int fullLength, char *full)
		: Base(true, VSYSTEM_FULLPATHNAME, 2024, SENTINELPREPARE(Prepare), SENTINELPOSTFIX(Postfix)), Relative(relative), FullLength(fullLength), Full(full) { sentinelDeviceSend(&Base, sizeof(vsystem_fullPathname)); }
	int RC;
	char *Ptr;
};

// dlOpen
// dlError
// dlSym
// dlClose
// randomness
// sleep
// currentTime

struct vsystem_getLastError {
	static __forceinline__ __device__ char *Prepare(vsystem_getLastError *t, char *data, char *dataEnd, intptr_t offset) {
		t->Buf = (char *)(data += ROUND8_(sizeof(*t)));
		char *end = (char *)(data += 1024);
		if (end > dataEnd) return nullptr;
		return end;
	}
	sentinelMessage Base;
	int BufLength;
	__device__ vsystem_getLastError(int bufLength)
		: Base(true, VSYSTEM_GETLASTERROR, 2024, SENTINELPREPARE(Prepare)), BufLength(bufLength) { sentinelDeviceSend(&Base, sizeof(vsystem_getLastError)); }
	char *Buf;
	int RC;
};

// currentTimeInt64
// setSystemCall
// getSystemCall
// nextSystemCall

#endif  /* _SENTINEL_VSYSTEM_H */