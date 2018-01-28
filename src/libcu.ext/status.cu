#include <ext/status.h> //: status.c
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
static __host_constant__ const char statusMutexStatics[] = {
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
** to the "_status" state vector declared above.
*/
#ifdef OMIT_WSD
#define _statusInit statusValue_t *x = &_GLOBAL(statusValue_t, _status)
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
	assert(op >= 0 && op < _LENGTHOF(statusMutexStatics));
	assert(mutex_held(statusMutexStatics[op] ? pcacheMutex() : allocMutex()));
	return _status.nowValue[op];
}

/* Add N to the value of a status record.  The caller must hold the appropriate mutex.  (Locking is checked by assert()).
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
	assert(op >= 0 && op < _LENGTHOF(statusMutexStatics));
	assert(mutex_held(statusMutexStatics[op] ? pcacheMutex() : allocMutex()));
	_status.nowValue[op] += n;
	if (_status.nowValue[op] > _status.maxValue[op])
		_status.maxValue[op] = _status.nowValue[op];
}

__host_device__ void status_dec(STATUS op, int n) //: sqlite3StatusDown
{
	_statusInit;
	assert(n >= 0);
	assert(op >= 0 && op < _LENGTHOF(statusMutexStatics));
	assert(mutex_held(statusMutexStatics[op] ? pcacheMutex() : allocMutex()));
	assert(op >= 0 && op < _LENGTHOF(_status.nowValue));
	_status.nowValue[op] -= n;
}

/* Adjust the highwater mark if necessary. The caller must hold the appropriate mutex. */
__host_device__ void status_max(STATUS op, int x) //: sqlite3StatusHighwater
{
	_statusInit;
	assert(x >= 0);
	statusValue_t newValue = (statusValue_t)x;
	assert(op >= 0 && op < _LENGTHOF(_status.nowValue));
	assert(op >= 0 && op < _LENGTHOF(statusMutexStatics));
	assert(mutex_held(statusMutexStatics[op] ? pcacheMutex() : allocMutex()));
	assert(op == STATUS_MALLOC_SIZE || op == STATUS_PAGECACHE_SIZE || op == STATUS_PARSER_STACK);
	if (newValue > _status.maxValue[op])
		_status.maxValue[op] = newValue;
}

/* Query status information. */
__host_device__ RC status64(STATUS op, int64_t *current, int64_t *highwater, bool resetFlag) //: sqlite3_status64
{
	_statusInit;
	if (op < 0 || op >= _LENGTHOF(_status.nowValue))
		return RC_MISUSE_BKPT;
#ifdef ENABLE_API_ARMOR
	if (!current || !highwater) return RC_MISUSE_BKPT;
#endif
	mutex *mutex = statusMutexStatics[op] ? pcacheMutex() : allocMutex();
	mutex_enter(mutex);
	*current = _status.nowValue[op];
	*highwater = _status.maxValue[op];
	if (resetFlag)
		_status.maxValue[op] = _status.nowValue[op];
	mutex_leave(mutex);
	(void)mutex; // Prevent warning when LIBCU_THREADSAFE = 0
	return RC_OK;
}
__host_device__ RC status(STATUS op, int *current, int *highwater, bool resetFlag) //: sqlite3_status
{
#ifdef ENABLE_API_ARMOR
	if (!current || !highwater) return RC_MISUSE_BKPT;
#endif
	int64_t current2 = 0, highwater2 = 0;
	RC rc = status64(op, &current2, &highwater2, resetFlag);
	if (!rc) { *current = (int)current2; *highwater = (int)highwater2; }
	return rc;
}

/* Return the number of LookasideSlot elements on the linked list */
__host_device__ static uint32_t countLookasideSlots(LookasideSlot *p)
{
	uint32_t cnt = 0;
	while (p) { p = p->next; cnt++; }
	return cnt;
}

/* Count the number of slots of lookaside memory that are outstanding */
__host_device__ int taglookasideUsed(tagbase_t *tag, int *highwater) //: sqlite3LookasideUsed
{
	uint32_t inits = countLookasideSlots(tag->lookaside.init);
	uint32_t frees = countLookasideSlots(tag->lookaside.free);
	if (highwater) *highwater = tag->lookaside.slots - inits;
	return tag->lookaside.slots - inits + frees;
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
		*current = taglookasideUsed(tag, highwater);
		if (resetFlag) {
			LookasideSlot *p = tag->lookaside.free;
			if (p) {
				while (p->next) p = p->next;
				p->next = tag->lookaside.init;
				tag->lookaside.init = tag->lookaside.free;
				tag->lookaside.free = nullptr;
			}
		}
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
	default: { rc = RC_ERROR; }
	}
	mutex_leave(tag->mutex);
	return rc;
}
