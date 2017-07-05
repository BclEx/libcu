#include <ext/status.h>
#include <assert.h>

/* Variables in which to record status information. */
#if _PTRSIZE > 4
typedef uint64_t statusValue_t;
#else
typedef uint32_t statusValue_t;
#endif
static __hostb_device__ _WSD struct Status {
	statusValue_t nowValue[10]; // Current value
	statusValue_t maxValue[10]; // Maximum value
} _status = { {0,}, {0,} };

/* Elements of _status[] are protected by either the memory allocator mutex, or by the pcache1 mutex.  The following array determines which. */
static __host_constant__ const char MutexStatics[] = {
	0,  // STATUS_MEMORY_USED
	1,  // STATUS_PAGECACHE_USED
	1,  // STATUS_PAGECACHE_OVERFLOW
	0,  // STATUS_SCRATCH_USED
	0,  // STATUS_SCRATCH_OVERFLOW
	0,  // STATUS_MALLOC_SIZE
	0,  // STATUS_PARSER_STACK
	1,  // STATUS_PAGECACHE_SIZE
	0,  // STATUS_SCRATCH_SIZE
	0,  // STATUS_MALLOC_COUNT
};

/* The "_status" macro will resolve to the status information state vector.  If writable static data is unsupported on the target,
** we have to locate the state vector at run-time.  In the more common case where writable static data is supported, _status can refer directly
** to the "g_status" state vector declared above.
*/
#ifdef OMIT_WSD
#define _statusInit statusValue_t *x = &GLOBAL(statusValue_t, h_status)
#define _status x[0]
#else
#define _statusInit
#define _status _status
#endif

/* Return the current value of a status parameter.  The caller must be holding the appropriate mutex. */
__host_device__ int64_t status_now(STATUS op) //: sqlite3StatusValue
{
	_statusInit;
	assert(op >= 0 && op < _LENGTHOF(_status.nowValue));
	assert(op >= 0 && op < _LENGTHOF(MutexStatics));
	assert(mutex_held(MutexStatics[op] ? allocCacheMutex() : allocMutex()));
	return _status.nowValue[op];
}

/*
** Add N to the value of a status record.  The caller must hold the appropriate mutex.  (Locking is checked by assert()).
**
** The status_inc() routine can accept positive or negative values for N. The value of N is added to the current status value and the high-water
** mark is adjusted if necessary.
**
** The status_dec() routine lowers the current value by N.  The highwater mark is unchanged.  N must be non-negative for status_dec().
*/
__host_device__ void status_inc(STATUS op, int n) //: sqlite3StatusUp
{
	_statusInit;
	assert(op >= 0 && op < _LENGTHOF(_status.nowValue));
	assert(op >= 0 && op < _LENGTHOF(MutexStatics));
	assert(mutex_held(MutexStatics[op] ? allocCacheMutex() : allocMutex()));
	_status.nowValue[op] += n;
	if (_status.nowValue[op] > _status.maxValue[op])
		_status.maxValue[op] = _status.nowValue[op];
}

__host_device__ void status_dec(STATUS op, int n) //: sqlite3StatusDown
{
	_statusInit;
	assert(op >= 0 && op < _LENGTHOF(_status.nowValue));
	assert(op >= 0 && op < _LENGTHOF(MutexStatics));
	assert(mutex_held(MutexStatics[op] ? allocCacheMutex() : allocMutex()));
	_status.nowValue[op] -= n;
}

/* Adjust the highwater mark if necessary. The caller must hold the appropriate mutex. */
__host_device__ void status_max(STATUS op, int x)
{
	_statusInit;
	assert(op >= 0 && op < _LENGTHOF(_status.nowValue));
	assert(op >= 0 && op < _LENGTHOF(MutexStatics));
	assert(mutex_held(MutexStatics[op] ? allocCacheMutex() : allocMutex()));
	assert(op == STATUS_MALLOC_SIZE || op == STATUS_PAGECACHE_SIZE || op == STATUS_SCRATCH_SIZE || op == STATUS_PARSER_STACK);
	statusValue_t newValue = (statusValue_t)x;
	if (newValue > _status.maxValue[op])
		_status.maxValue[op] = newValue;
}

/* Query status information. */
__host_device__ RC status64(STATUS op, int64_t *current, int64_t *highwater, bool resetFlag)
{
	_statusInit;
	if (op < 0 || op >= _LENGTHOF(_status.nowValue))
		return RC_MISUSE_BKPT;
#ifdef ENABLE_API_ARMOR
	if (!current || !highwater) return RC_MISUSE_BKPT;
#endif
	mutex *mutex = MutexStatics[op] ? allocCacheMutex() : allocMutex();
	mutex_enter(mutex);
	*current = _status.nowValue[op];
	*highwater = _status.maxValue[op];
	if (resetFlag)
		_status.maxValue[op] = _status.nowValue[op];
	mutex_leave(mutex);
	(void)mutex; // Prevent warning when LIBCU_THREADSAFE = 0
	return RC_OK;
}
__host_device__ RC status(STATUS op, int *current, int *highwater, bool resetFlag)
{
#ifdef ENABLE_API_ARMOR
	if (!current || !highwater) return RC_MISUSE_BKPT;
#endif
	int64_t current2 = 0, highwater2 = 0;
	RC rc = status64(op, &current2, &highwater2, resetFlag);
	if (!rc) { *current = (int)current2; *highwater = (int)highwater2; }
	return rc;
}

/* Query status information for a single tag object */
__host_device__ RC tagstatus(tagbase_t *tag, STATUS op, int *current, int *highwater, bool resetFlag) //: sqlite3_db_status
{
	RC rc = 0;
#ifdef ENABLE_API_ARMOR
	if (!tagSafetyCheckOk(tag) || !current || !highwater)
		return RC_MISUSE_BKPT;
#endif
	mutex_enter(tag->mutex);
	switch (op) {
	case TAGSTATUS_LOOKASIDE_USED: {
		*current = tag->lookaside.outs;
		*highwater = tag->lookaside.maxOuts;
		if (resetFlag)
			tag->lookaside.maxOuts = tag->lookaside.outs;
		break; }
	case TAGSTATUS_LOOKASIDE_HIT:
	case TAGSTATUS_LOOKASIDE_MISS_SIZE:
	case TAGSTATUS_LOOKASIDE_MISS_FULL: {
		ASSERTCOVERAGE(op == TAGSTATUS_LOOKASIDE_HIT);
		ASSERTCOVERAGE(op == TAGSTATUS_LOOKASIDE_MISS_SIZE);
		ASSERTCOVERAGE(op == TAGSTATUS_LOOKASIDE_MISS_FULL);
		assert((op - TAGSTATUS_LOOKASIDE_HIT) >= 0);
		assert((op - TAGSTATUS_LOOKASIDE_HIT) < 3);
		*current = 0;
		*highwater = tag->lookaside.stats[op - TAGSTATUS_LOOKASIDE_HIT];
		if (resetFlag)
			tag->lookaside.stats[op - TAGSTATUS_LOOKASIDE_HIT] = 0;
		break; }
	default: { rc = 1; }
	}
	mutex_leave(tag->mutex);
	return rc;
}

//case TAGSTATUS_CACHE_USED_SHARED:
//case TAGSTATUS_CACHE_USED: {
//	/* 
//	** Return an approximation for the amount of memory currently used by all pagers associated with the given tag object.  The
//	** highwater mark is meaningless and is returned as zero.
//	*/
//	int totalUsed = 0;
//	int i;
//	sqlite3BtreeEnterAll(db);
//	for(i=0; i<db->nDb; i++){
//		Btree *pBt = db->aDb[i].pBt;
//		if( pBt ){
//			Pager *pPager = sqlite3BtreePager(pBt);
//			int nByte = sqlite3PagerMemUsed(pPager);
//			if( op==SQLITE_DBSTATUS_CACHE_USED_SHARED ){
//				nByte = nByte / sqlite3BtreeConnectionCount(pBt);
//			}
//			totalUsed += nByte;
//		}
//	}
//	sqlite3BtreeLeaveAll(db);
//	*current = totalUsed;
//	*highwater = 0;
//	break; }
//case TAGSTATUS_SCHEMA_USED: {
//	/*
//	** *current gets an accurate estimate of the amount of memory used to store the schema for all databases (main, temp, and any ATTACHed
//	** databases.  *highwater is set to zero.
//	*/
//	int i; // Used to iterate through schemas
//	int nByte = 0; // Used to accumulate return value
//	sqlite3BtreeEnterAll(db);
//	db->pnBytesFreed = &nByte;
//	for(i=0; i<db->nDb; i++){
//		Schema *pSchema = db->aDb[i].pSchema;
//		if( ALWAYS(pSchema!=0) ){
//			HashElem *p;

//			nByte += sqlite3GlobalConfig.m.xRoundup(sizeof(HashElem)) * (
//				pSchema->tblHash.count 
//				+ pSchema->trigHash.count
//				+ pSchema->idxHash.count
//				+ pSchema->fkeyHash.count
//				);
//			nByte += sqlite3_msize(pSchema->tblHash.ht);
//			nByte += sqlite3_msize(pSchema->trigHash.ht);
//			nByte += sqlite3_msize(pSchema->idxHash.ht);
//			nByte += sqlite3_msize(pSchema->fkeyHash.ht);

//			for(p=sqliteHashFirst(&pSchema->trigHash); p; p=sqliteHashNext(p)){
//				sqlite3DeleteTrigger(db, (Trigger*)sqliteHashData(p));
//			}
//			for(p=sqliteHashFirst(&pSchema->tblHash); p; p=sqliteHashNext(p)){
//				sqlite3DeleteTable(db, (Table *)sqliteHashData(p));
//			}
//		}
//	}
//	db->pnBytesFreed = 0;
//	sqlite3BtreeLeaveAll(db);
//	*highwater = 0;
//	*current = bytes;
//	break; }
//case TAGSTATUS_STMT_USED: {
//	/*
//	** *pCurrent gets an accurate estimate of the amount of memory used to store all prepared statements.
//	** *pHighwater is set to zero.
//	*/
//	struct Vdbe *pVdbe;         /* Used to iterate through VMs */
//	int nByte = 0;              /* Used to accumulate return value */

//	db->pnBytesFreed = &nByte;
//	for(pVdbe=db->pVdbe; pVdbe; pVdbe=pVdbe->pNext){
//		sqlite3VdbeClearObject(db, pVdbe);
//		sqlite3DbFree(db, pVdbe);
//	}
//	db->pnBytesFreed = 0;

//	*pHighwater = 0;  /* IMP: R-64479-57858 */
//	*pCurrent = nByte;
//	break; }
//case SQLITE_DBSTATUS_CACHE_HIT:
//case SQLITE_DBSTATUS_CACHE_MISS:
//case SQLITE_DBSTATUS_CACHE_WRITE: {
//	/*
//	** Set *pCurrent to the total cache hits or misses encountered by all
//	** pagers the database handle is connected to. *pHighwater is always set 
//	** to zero.
//	*/
//	int i;
//	int nRet = 0;
//	assert( SQLITE_DBSTATUS_CACHE_MISS==SQLITE_DBSTATUS_CACHE_HIT+1 );
//	assert( SQLITE_DBSTATUS_CACHE_WRITE==SQLITE_DBSTATUS_CACHE_HIT+2 );

//	for(i=0; i<db->nDb; i++){
//		if( db->aDb[i].pBt ){
//			Pager *pPager = sqlite3BtreePager(db->aDb[i].pBt);
//			sqlite3PagerCacheStat(pPager, op, resetFlag, &nRet);
//		}
//	}
//	*pHighwater = 0; /* IMP: R-42420-56072 */
//	/* IMP: R-54100-20147 */
//	/* IMP: R-29431-39229 */
//	*pCurrent = nRet;
//	break; }
//case TAGSTATUS_DEFERRED_FKS: {
//	/* Set *pCurrent to non-zero if there are unresolved deferred foreign key constraints.  Set *pCurrent to zero if all foreign key constraints
//	** have been satisfied.  The *pHighwater is always set to zero.
//	*/
//	*pHighwater = 0;  /* IMP: R-11967-56545 */
//	*pCurrent = db->nDeferredImmCons>0 || db->nDeferredCons>0;
//	break; }
