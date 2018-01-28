#include "sentinel-vsystem.h"

static bool Executor(void *tag, sentinelMessage *data, int length, char *(**hostPrepare)(void*,char*,char*,intptr_t))
{
	vsystem *vfs = nullptr;
	if (data->OP < VSYSFILE_CLOSE || data->OP > VSYSTEM_NEXTSYSTEMCALL) return false;
	switch (data->OP) {
	case VSYSFILE_CLOSE: { vsysfile_close *msg = (vsysfile_close *)data; vsys_close(msg->F); return true; }
	case VSYSFILE_READ: { vsysfile_read *msg = (vsysfile_read *)data; msg->RC = vsys_read(msg->F, msg->Buf, msg->Amount, msg->Offset); return true; }
	case VSYSFILE_WRITE: { vsysfile_write *msg = (vsysfile_write *)data; msg->RC = vsys_write(msg->F, msg->Buf, msg->Amount, msg->Offset); return true; }
	case VSYSFILE_TRUNCATE: { vsysfile_truncate *msg = (vsysfile_truncate *)data; msg->RC = vsys_truncate(msg->F, msg->Size); return true; }
	case VSYSFILE_SYNC: { vsysfile_sync *msg = (vsysfile_sync *)data; msg->RC = vsys_sync(msg->F, msg->Flags); return true; }
	case VSYSFILE_FILESIZE: { vsysfile_fileSize *msg = (vsysfile_fileSize *)data; msg->RC = vsys_fileSize(msg->F, &msg->Size); return true; }
	case VSYSFILE_LOCK: { vsysfile_lock *msg = (vsysfile_lock *)data; msg->RC = vsys_lock(msg->F, msg->Lock); return true; }
	case VSYSFILE_UNLOCK: { vsysfile_unlock *msg = (vsysfile_unlock *)data; msg->RC = vsys_unlock(msg->F, msg->Lock); return true; }
	case VSYSFILE_CHECKRESERVEDLOCK: { vsysfile_checkReservedLock *msg = (vsysfile_checkReservedLock *)data; msg->RC = vsys_checkReservedLock(msg->F, &msg->Lock); return true; }
									 // fileControl
									 // sectorSize
									 // deviceCharacteristics
									 // shmMap
									 // shmLock
									 // shmBarrier
									 // shmUnmap
									 // fetch
									 // unfetch
	case VSYSTEM_OPEN: {
		vsystem_open *msg = (vsystem_open *)data;
		vsysfile *f = nullptr; //(vsysfile *)allocZero(vfs->SizeOsFile);
		msg->RC = vsys_open(vfs, msg->Name, f, msg->Flags, &msg->OutFlags);
		msg->F = f;
		return true; }
	case VSYSTEM_DELETE: { vsystem_delete *msg = (vsystem_delete *)data; msg->RC = vsys_delete(vfs, msg->Filename, msg->SyncDir); return true; }
	case VSYSTEM_ACCESS: { vsystem_access *msg = (vsystem_access *)data; msg->RC = vsys_access(vfs, msg->Filename, msg->Flags, &msg->ResOut); return true; }
	case VSYSTEM_FULLPATHNAME: { vsystem_fullPathname *msg = (vsystem_fullPathname *)data; msg->RC = vsys_fullPathname(vfs, msg->Relative, msg->FullLength, msg->Full); return true; }
							   // dlOpen
							   // dlError
							   // dlSym
							   // dlClose
							   // randomness
							   // sleep
							   // currentTime
	case VSYSTEM_GETLASTERROR: { vsystem_getLastError *msg = (vsystem_getLastError *)data; msg->RC = vsys_getLastError(vfs); return true; }
							   // currentTimeInt64
							   // setSystemCall
							   // getSystemCall
							   // nextSystemCall
	}
	return false;
}

//static RuntimeSentinelExecutor _sysExecutor;
//void VSystemSentinel::Initialize()
//{
//	MutexEx masterMutex;
//	SysEx::Initialize(masterMutex);
//	VSystem *vfs = VSystem::FindVfs(nullptr);
//	_sysExecutor.Name = "sys";
//	_sysExecutor.Executor = Executor;
//	_sysExecutor.Tag = vfs;
//	RuntimeSentinel::ServerInitialize(&_sysExecutor);
//}
//
//void VSystemSentinel::Shutdown()
//{
//	RuntimeSentinel::ServerShutdown();
//	SysEx::Shutdown();
//}

