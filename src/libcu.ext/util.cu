#include <ext/util.h> //: util.c
#include <assert.h>

/* Give a callback to the test harness that can be used to simulate faults in places where it is difficult or expensive to do so purely by means
** of inputs.
**
** The intent of the integer argument is to let the fault simulator know which of multiple sqlite3FaultSim() calls has been hit.
**
** Return whatever integer value the test callback returns, or return RC_OK if no test callback is installed.
*/
#ifndef LIBCU_UNTESTABLE
__host_device__ RC sqlite3FaultSim(int test) //: sqlite3FaultSim
{
	int (*callback)(int) = _runtimeConfig.testCallback;
	return callback ? callback(test) : RC_OK;
}
#endif

/* Helper function for tagError() - called rarely.  Broken out into a separate routine to avoid unnecessary register saves on entry to tagError(). */
//static __host_device__ void tagErrorFinish(tagbase_t *tag, int errCode)
//{
//	if (tag->pErr) sqlite3ValueSetNull(tag->pErr);
//	tagSystemError(tag, errCode);
//}

/* Set the current error code to errCode and clear any prior error message. Also set tag.sysErrno (by calling tagSystem) if the errCode indicates
** that would be appropriate.
*/
__host_device__ void tagError(tagbase_t *tag, int errCode) //: sqlite3Error
{
	assert(tag);
	tag->errCode = errCode;
	//if (errCode || tag->pErr) tagErrorFinish(tag, errCode);
}

/* Load the tag.sysErrno field if that is an appropriate thing to do based on the Libcu error code in rc. */
__host_device__ void tagSystemError(tagbase_t *tag, RC rc) //: sqlite3SystemError
{
	if (rc == RC_IOERR_NOMEM) return;
	rc &= 0xff;
	//if (rc == RC_CANTOPEN || rc == RC_IOERR)
	//	tag->SysErrno = sqlite3OsGetLastError(tag->vsystem);
}

/* Set the most recent error code and error string for the sqlite handle "db". The error code is set to "err_code".
**
** If it is not NULL, string zFormat specifies the format of the error string in the style of the printf functions: The following
** format characters are allowed:
**
**      %s      Insert a string
**      %z      A string that should be freed after use
**      %d      Insert an integer
**      %T      Insert a token
**      %S      Insert the first element of a SrcList
**
** zFormat and any string tokens that follow it are assumed to be encoded in UTF-8.
**
** To clear the most recent error for sqlite handle "db", sqlite3Error should be called with err_code set to SQLITE_OK and zFormat set to NULL.
*/
//__host_device__ void tagErrorWithMsg(tagbase_t *tag, int errCode, const char *format, ...)
//{
//	assert(tag);
//	tag->errCode = errCode;
//	tagSystemError(tag, errCode);
//	if (!format)
//		tagError(tag, errCode);
//	//else if (tag->pErr || (tag->pErr = sqlite3ValueNew(db)) != 0) {
//	//	va_list va;
//	//	va_start(va, format);
//	//	char *z = sqlite3VMPrintf(tag, format, va);
//	//	va_end(va);
//	//	sqlite3ValueSetStr(tag->pErr, -1, z, TEXTENCODE_UTF8, SQLITE_DYNAMIC);
//	//}
//}

/* Log an error that is an API call on a connection pointer that should not have been used.  The "type" of connection pointer is given as the
** argument.  The zType is a word like "NULL" or "closed" or "invalid".
*/
static __host_device__ void logBadConnection(const char *type)
{
	//sqlite3_log(RC_MISUSE, "API call with %s database connection pointer", type);
}

/* Check to make sure we have a valid db pointer.  This test is not foolproof but it does provide some measure of protection against
** misuse of the interface such as passing in db pointers that are NULL or which have been previously closed.  If this routine returns
** 1 it means that the db pointer is valid and 0 if it should not be dereferenced for any reason.  The calling function should invoke
** SQLITE_MISUSE immediately.
**
** sqlite3SafetyCheckOk() requires that the db pointer be valid for use.  sqlite3SafetyCheckSickOrOk() allows a db pointer that failed to
** open properly and is not fit for general use but which can be used as an argument to sqlite3_errmsg() or sqlite3_close().
*/
__host_device__ bool tagSafetyCheckOk(tagbase_t *tag) //: sqlite3SafetyCheckOk
{
	if (!tag) {
		logBadConnection("NULL");
		return false;
	}
	//uint32_t magic = tag->magic;
	//if (magic != LIBCU_MAGIC_OPEN) {
	//	if (tagSafetyCheckSickOrOk(tag)) {
	//		ASSERTCOVERAGE(_runtimeConfig.log);
	//		logBadConnection("unopened");
	//	}
	//	return false;
	//}
	return true;
}
__host_device__ bool tagSafetyCheckSickOrOk(tagbase_t *tag) //: sqlite3SafetyCheckSickOrOk
{
	//uint32_t magic = tag->magic;
	//if (magic != LIBCU_MAGIC_SICK && magic != LIBCU_MAGIC_OPEN && magic != LIBCU_MAGIC_BUSY) {
	//	ASSERTCOVERAGE(_runtimeConfig.log);
	//	logBadConnection("invalid");
	//	return false;
	//}
	return true;
}
