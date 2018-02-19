#include <stdlibcu.h> //: notify.c
#include <stringcu.h> //: notify.c
#include <ext/global.h>
#include <assert.h>

/* Omit this entire file if SQLITE_ENABLE_UNLOCK_NOTIFY is not defined. */
#ifdef LIBCU_ENABLE_UNLOCK_NOTIFY

#define assertMutexHeld() assert(mutex_held(mutexAlloc(MUTEX_STATIC_MASTER)))

/* Head of a linked list of all sqlite3 objects created by this process for which either sqlite3.pBlockingConnection or sqlite3.pUnlockConnection
** is not NULL. This variable may only accessed while the STATIC_MASTER mutex is held.
*/
static __hostb_device__ tagbase_t *WSD_ _blockedList = 0;

#ifndef NDEBUG
/* This function is a complex assert() that verifies the following properties of the blocked connections list:
**
**   1) Each entry in the list has a non-NULL value for either pUnlockConnection or pBlockingConnection, or both.
**
**   2) All entries in the list that share a common value for xUnlockNotify are grouped together.
**
**   3) If the argument db is not NULL, then none of the entries in the blocked connections list have pUnlockConnection or pBlockingConnection
**      set to db. This is used when closing connection db.
*/
static __host_device__ void checkListProperties(tagbase_t *tag)
{
	for (tagbase_t *p = _blockedList; p; p = p->nextBlocked) {
		// Verify property (1)
		assert(p->unlockConnection || p->blockingConnection);
		// Verify property (2)
		bool seen = false;
		for (tagbase_t *p2 = _blockedList; p2 != p; p2 = p2->nextBlocked) {
			if(p2->unlockNotify == p->unlockNotify) seen = true;
			assert(p2->unlockNotify == p->unlockNotify || !seen);
			assert(!tag || p->unlockConnection != tag);
			assert(!tag || p->blockingConnection != tag);
		}
	}
}
#else
#define checkListProperties(x)
#endif

/* Remove connection db from the blocked connections list. If connection db is not currently a part of the list, this function is a no-op. */
static __host_device__ void removeFromBlockedList(tagbase_t *tag)
{
	assertMutexHeld();
	for (tagbase_t **pp = &_blockedList; *pp; pp = &(*pp)->nextBlocked)
		if (*pp == tag) {
			*pp = (*pp)->nextBlocked;
			break;
		}
}

/* Add connection db to the blocked connections list. It is assumed that it is not already a part of the list. */
static __host_device__ void addToBlockedList(tagbase_t *tag)
{
	assertMutexHeld();
	tagbase_t **pp; for (pp = &_blockedList; *pp && (*pp)->unlockNotify != tag->unlockNotify; pp = &(*pp)->nextBlocked);
	tag->nextBlocked = *pp;
	*pp = tag;
}

/* Obtain the STATIC_MASTER mutex. */
static __host_device__ void enterMutex()
{
	mutex_enter(mutexAlloc(MUTEX_STATIC_MASTER));
	checkListProperties(0);
}

/* Release the STATIC_MASTER mutex. */
static __host_device__ void leaveMutex()
{
	assertMutexHeld();
	checkListProperties(0);
	mutex_leave(mutexAlloc(MUTEX_STATIC_MASTER));
}

/* Register an unlock-notify callback.
**
** This is called after connection "db" has attempted some operation but has received an SQLITE_LOCKED error because another connection
** (call it pOther) in the same process was busy using the same shared cache.  pOther is found by looking at db->pBlockingConnection.
**
** If there is no blocking connection, the callback is invoked immediately, before this routine returns.
**
** If pOther is already blocked on db, then report SQLITE_LOCKED, to indicate a deadlock.
**
** Otherwise, make arrangements to invoke xNotify when pOther drops its locks.
**
** Each call to this routine overrides any prior callbacks registered on the same "db".  If xNotify==0 then any prior callbacks are immediately cancelled.
*/
__host_device__ RC notify_unlock(tagbase_t *tag, void (*notify)(void**,int), void *arg) //: sqlite3_unlock_notify
{
	RC rc = RC_OK;
	mutex_enter(tag->mutex);
	enterMutex();
	if (!notify) {
		removeFromBlockedList(tag);
		tag->blockingConnection = nullptr;
		tag->unlockConnection = nullptr;
		tag->unlockNotify = nullptr;
		tag->unlockArg = nullptr;
	}
	// The blocking transaction has been concluded. Or there never was a  blocking transaction. In either case, invoke the notify callback immediately. 
	else if (!tag->blockingConnection) 
		notify(&arg, 1);
	else {
		tagbase_t *p; for (p = tag->blockingConnection; p && p != tag; p = p->unlockConnection);
		if (p) rc = RC_LOCKED; // Deadlock detected.
		else {
			tag->unlockConnection = tag->blockingConnection;
			tag->unlockNotify = notify;
			tag->unlockArg = arg;
			removeFromBlockedList(tag);
			addToBlockedList(tag);
		}
	}
	leaveMutex();
	assert(!tag->mallocFailed );
	tagErrorWithMsg(tag, rc, rc ? "database is deadlocked" : nullptr);
	mutex_leave(tag->mutex);
	return rc;
}

/* This function is called while stepping or preparing a statement associated with connection db. The operation will return SQLITE_LOCKED
** to the user because it requires a lock that will not be available until connection pBlocker concludes its current transaction.
*/
__host_device__ void notifyConnectionBlocked(tagbase_t *tag, tagbase_t *blocker) //: sqlite3ConnectionBlocked
{ 
	enterMutex();
	if (!tag->blockingConnection && !tag->unlockConnection)
		addToBlockedList(tag);
	tag->blockingConnection = blocker;
	leaveMutex();
}

/* This function is called when the transaction opened by database db has just finished. Locks held 
** by database connection db have been released.
**
** This function loops through each entry in the blocked connections list and does the following:
**
**   1) If the sqlite3.pBlockingConnection member of a list entry is set to db, then set pBlockingConnection=0.
**
**   2) If the sqlite3.pUnlockConnection member of a list entry is set to db, then invoke the configured unlock-notify callback and set pUnlockConnection=0.
**
**   3) If the two steps above mean that pBlockingConnection==0 and pUnlockConnection==0, remove the entry from the blocked connections list.
*/
__host_device__ void notifyConnectionUnlocked(tagbase_t *tag) //: sqlite3ConnectionUnlocked
{
	void (*unlockNotify)(void**,int) = 0; // Unlock-notify cb to invoke
	void *statics[16]; // Starter space for aArg[].  No malloc required
	void **args = statics; // Arguments to the unlock callback
	void **dyns = nullptr; // Dynamically allocated space for aArg[]
	int argCount = 0; // Number of entries in aArg[]
	enterMutex(); // Enter STATIC_MASTER mutex
	/* This loop runs once for each entry in the blocked-connections list. */
	tagbase_t **pp; for (pp = &_blockedList; *pp; /* no-op */) {
		tagbase_t *p = *pp;
		// Step 1.
		if (p->blockingConnection == tag)
			p->blockingConnection = nullptr;
		// Step 2.
		if (p->unlockConnection == tag) {
			assert(p->unlockNotify);
			if (p->unlockNotify != unlockNotify && argCount != 0) {
				unlockNotify(args, argCount);
				argCount = 0;
			}

			allocBenignBegin();
			assert(args == dyns || (!dyns && args == statics));
			assert(argCount <= (int)ARRAYSIZE_(statics) || args == dyns);
			if ((!dyns && argCount == (int)ARRAYSIZE_(statics)) || (dyns && argCount == (int)(allocSize(dyns)/sizeof(void *)))) {
				// The aArg[] array needs to grow.
				void **newArgs = (void **)alloc(argCount * sizeof(void *) * 2);
				if (newArgs) {
					memcpy(newArgs, args, argCount * sizeof(void *));
					mfree(dyns);
					dyns = args = newArgs;
				}
				else {
					// This occurs when the array of context pointers that need to be passed to the unlock-notify callback is larger than the
					// aStatic[] array allocated on the stack and the attempt to allocate a larger array from the heap has failed.
					//
					// This is a difficult situation to handle. Returning an error code to the caller is insufficient, as even if an error code
					// is returned the transaction on connection db will still be closed and the unlock-notify callbacks on blocked connections
					// will go unissued. This might cause the application to wait indefinitely for an unlock-notify callback that will never 
					// arrive.
					//
					// Instead, invoke the unlock-notify callback with the context array already accumulated. We can then clear the array and
					// begin accumulating any further context pointers without requiring any dynamic allocation. This is sub-optimal because
					// it means that instead of one callback with a large array of context pointers the application will receive two or more
					// callbacks with smaller arrays of context pointers, which will reduce the applications ability to prioritize multiple 
					// connections. But it is the best that can be done under the circumstances.
					unlockNotify(args, argCount);
					argCount = 0;
				}
			}
			allocBenignEnd();

			args[argCount++] = p->unlockArg;
			unlockNotify = p->unlockNotify;
			p->unlockConnection = nullptr;
			p->unlockNotify = nullptr;
			p->unlockArg = nullptr;
		}
		// Step 3.
		// Remove connection p from the blocked connections list.
		if (!p->blockingConnection && !p->unlockConnection) { *pp = p->nextBlocked; p->nextBlocked = nullptr; }
		else pp = &p->nextBlocked;
	}
	if (argCount != 0)
		unlockNotify(args, argCount);
	mfree(dyns);
	leaveMutex(); // Leave STATIC_MASTER mutex
}

/* This is called when the database connection passed as an argument is being closed. The connection is removed from the blocked list. */
__host_device__ void notifyConnectionClosed(tagbase_t *tag) //: sqlite3ConnectionClosed
{
	notifyConnectionUnlocked(tag);
	enterMutex();
	removeFromBlockedList(tag);
	checkListProperties(tag);
	leaveMutex();
}
#endif
