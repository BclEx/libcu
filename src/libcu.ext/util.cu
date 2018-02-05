#include <ext/util.h> //: util.c
#include <assert.h>

/* Routine needed to support the testcase() macro. */
// __coverage(int x) //: sqlite3Coverage

/* Give a callback to the test harness that can be used to simulate faults in places where it is difficult or expensive to do so purely by means
** of inputs.
**
** The intent of the integer argument is to let the fault simulator know which of multiple sqlite3FaultSim() calls has been hit.
**
** Return whatever integer value the test callback returns, or return RC_OK if no test callback is installed.
*/
#ifndef LIBCU_UNTESTABLE
__host_device__ RC _faultSim(int test) //: sqlite3FaultSim
{
	int (*callback)(int) = _runtimeConfig.testCallback;
	return callback ? callback(test) : RC_OK;
}
#endif

/* Helper function for tagError() - called rarely.  Broken out into a separate routine to avoid unnecessary register saves on entry to tagError(). */
static __host_device__ void tagErrorFinish(tagbase_t *tag, int errCode) //: sqlite3ErrorFinish
{
	//if (tag->err) sqlite3ValueSetNull(tag->err);
	tagSystemError(tag, errCode);
}

/* Set the current error code to errCode and clear any prior error message. Also set tag.sysErrno (by calling tagSystem) if the errCode indicates
** that would be appropriate.
*/
__host_device__ void tagError(tagbase_t *tag, int errCode) //: sqlite3Error
{
	assert(tag);
	tag->errCode = errCode;
	if (errCode || tag->err) tagErrorFinish(tag, errCode);
}

/* Load the tag.sysErrno field if that is an appropriate thing to do based on the Libcu error code in rc. */
__host_device__ void tagSystemError(tagbase_t *tag, RC rc) //: sqlite3SystemError
{
	if (rc == RC_IOERR_NOMEM) return;
	rc &= 0xff;
	if (rc == RC_CANTOPEN || rc == RC_IOERR)
		tag->sysErrno = vsys_getLastError(tag->vsys);
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
__host_device__ void tagErrorWithMsgv(tagbase_t *tag, int errCode, const char *format, va_list va) //: sqlite3ErrorWithMsg
{
	assert(tag);
	tag->errCode = errCode;
	tagSystemError(tag, errCode);
	if (!format)
		tagError(tag, errCode);
	//else if (tag->err || (tag->err = sqlite3ValueNew(tag))) {
	//	char *z = vmtagprintf(tag, format, va);
	//	sqlite3ValueSetStr(tag->err, -1, z, TEXTENCODE_UTF8, SQLITE_DYNAMIC);
	//}
}

/* Add an error message to pParse->zErrMsg and increment pParse->nErr.
** The following formatting characters are allowed:
**
**      %s      Insert a string
**      %z      A string that should be freed after use
**      %d      Insert an integer
**      %T      Insert a token
**      %S      Insert the first element of a SrcList
**
** This function should be used to report any error that occurs while compiling an SQL statement (i.e. within sqlite3_prepare()). The
** last thing the sqlite3_prepare() function does is copy the error stored by this function into the database handle using sqlite3Error().
** Functions sqlite3Error() or sqlite3ErrorWithMsg() should be used during statement execution (sqlite3_step() etc.).
*/
__host_device__ void sqlite3ErrorMsgv(parsebase_t *parse, const char *format, va_list va) //: sqlite3ErrorMsg
{
	tagbase_t *tag = parse->tag;
	char *msg = vmtagprintf(tag, format, va);
	if (tag->suppressErr)
		tagfree(tag, msg);
	else {
		parse->errs++;
		tagfree(tag, parse->errMsg);
		parse->errMsg = msg;
		parse->rc = RC_ERROR;
	}
}
#ifndef __CUDA_ARCH__
__host_device__ void sqlite3ErrorMsg(parsebase_t *parse, const char *format, ...) { va_list va; va_start(va, format); sqlite3ErrorMsgv(parse, format, va); va_end(va); }
#else
STDARGvoid(sqlite3ErrorMsg, sqlite3ErrorMsgv(parse, format, va), parsebase_t *parse, const char *format);
#endif

/* Convert an SQL-style quoted string into a normal string by removing the quote characters.  The conversion is done in-place.  If the
** input does not begin with a quote character, then this routine is a no-op.
**
** The input string must be zero-terminated.  A new zero-terminator is added to the dequoted string.
**
** The return value is -1 if no dequoting occurs or the length of the dequoted string, exclusive of the zero terminator, if dequoting does occur.
**
** 2002-Feb-14: This routine is extended to remove MS-Access style brackets from around identifiers.  For example:  "[a-b-c]" becomes "a-b-c".
*/
__host_device__ void dequote(char *z)
{
	if (!z) return;
	char quote = z[0];
	if (!isquote(quote)) return;
	if (quote == '[') quote = ']';
	int i, j; for (i = 1, j = 0; ; i++) {
		assert(z[i]);
		if (z[i] == quote){
			if (z[i+1] == quote) { z[j++] = quote; i++; }
			else break;
		}
		else z[j++] = z[i];
	}
	z[j] = 0;
}

/* Log an error that is an API call on a connection pointer that should not have been used.  The "type" of connection pointer is given as the
** argument.  The zType is a word like "NULL" or "closed" or "invalid".
*/
static __host_device__ void logBadConnection(const char *type)
{
	_log(RC_MISUSE, "API call with %s database connection pointer", type);
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
	uint32_t magic = tag->magic;
	if (magic != TAG_MAGIC_OPEN) {
		if (tagSafetyCheckSickOrOk(tag)) {
			TESTCASE(_runtimeConfig.log);
			logBadConnection("unopened");
		}
		return false;
	}
	return true;
}
__host_device__ bool tagSafetyCheckSickOrOk(tagbase_t *tag) //: sqlite3SafetyCheckSickOrOk
{
	uint32_t magic = tag->magic;
	if (magic != TAG_MAGIC_SICK && magic != TAG_MAGIC_OPEN && magic != TAG_MAGIC_BUSY) {
		TESTCASE(_runtimeConfig.log);
		logBadConnection("invalid");
		return false;
	}
	return true;
}

#ifdef LIBCU_ENABLE_8_3_NAMES
/*
** If LIBCU_ENABLE_8_3_NAMES is set at compile-time and if the database filename in zBaseFilename is a URI with the "8_3_names=1" parameter and
** if filename in z[] has a suffix (a.k.a. "extension") that is longer than three characters, then shorten the suffix on z[] to be the last three
** characters of the original suffix.
**
** If LIBCU_ENABLE_8_3_NAMES is set to 2 at compile-time, then always do the suffix shortening regardless of URI parameter.
**
** Examples:
**     test.db-journal    =>   test.nal
**     test.db-wal        =>   test.wal
**     test.db-shm        =>   test.shm
**     test.db-mj7f3319fa =>   test.9fa
*/
__host_device__ void util_fileSuffix3(const char *baseFilename, char *z) //: sqlite3FileSuffix3
{
#if LIBCU_ENABLE_8_3_NAMES < 2
	if (uriBoolean(baseFilename, "8_3_names", 0) )
#endif
	{
		int i, sz;
		int size = strlen(z);
		for (i = size - 1; i > 0 && z[i] != '/' && z[i] != '.'; i--) { }
		if (z[i] == '.' && _ALWAYS(size > i + 4)) memmove(&z[i+1], &z[size-3], 4);
	}
}
#endif

/*
** Add a new name/number pair to a vlist_t.  This might require that the vlist_t object be reallocated, so return the new vlist_t.  If an OOM
** error occurs, the original vlist_t returned and the db->mallocFailed flag is set.
**
** A vlist_t is really just an array of integers.  To destroy a vlist_t, simply pass it to sqlite3DbFree().
**
** The first integer is the number of integers allocated for the whole vlist_t.  The second integer is the number of integers actually used.
** Each name/number pair is encoded by subsequent groups of 3 or more integers.
**
** Each name/number pair starts with two integers which are the numeric value for the pair and the size of the name/number pair, respectively.
** The text name overlays one or more following integers.  The text name is always zero-terminated.
**
** Conceptually:
**
**    struct vlist_t {
**      int nAlloc;   // Number of allocated slots 
**      int nUsed;    // Number of used slots 
**      struct VListEntry {
**        int iValue;    // Value for this entry
**        int nSlot;     // Slots used by this entry
**        // ... variable name goes here
**      } a[0];
**    }
**
** During code generation, pointers to the variable names within the vlist_t are taken.  When that happens, nAlloc is set to zero as an 
** indication that the vlist_t may never again be enlarged, since the accompanying realloc() would invalidate the pointers.
*/
__host_device__ vlist_t *util_vlistadd(tagbase_t *tag, vlist_t *list, const char *name, int nameLength, int id) //: sqlite3VListAdd
{
	int ints = nameLength / 4 + 3; // number of sizeof(int) objects needed for zName
	assert(!list || list[0] >= 3); // Verify ok to add new elements
	if (!list || list[1] + ints > list[0]) {
		// Enlarge the allocation
		int allocSize = (list ? list[0] * 2 : 10) + ints;
		vlist_t *newItem = (vlist_t *)tagrealloc(tag, list, allocSize * sizeof(int));
		if (!newItem) return list;
		if (!list) newItem[1] = 2;
		list = newItem;
		list[0] = allocSize;
	}
	int i = list[1]; // Index in list[] where zName is stored
	list[i] = id;
	list[i + 1] = ints;
	char *z = (char *)&list[i + 2]; // Pointer to where zName will be stored
	list[1] = i + ints;
	assert(list[1] <= list[0]);
	memcpy(z, name, nameLength);
	z[nameLength] = 0;
	return list;
}

/* Return a pointer to the name of a variable in the given vlist_t that has the value iVal.  Or return a NULL if there is no such variable in the list */
__host_device__ const char *util_vlistIdToName(vlist_t *list, int id) //: sqlite3VListNumToName
{
	if (!list) return 0;
	int max = list[1];
	int i = 2;
	do {
		if (list[i] == id) return (char *)&list[i + 2];
		i += list[i + 1];
	} while (i < max);
	return 0;
}

/* Return the number of the variable named zName, if it is in vlist_t. or return 0 if there is no such variable. */
__host_device__ int util_vlistNameToId(vlist_t *list, const char *name, int nameLength) //: sqlite3VListNameToNum
{
	if (!list) return 0;
	int max = list[1];
	int i = 2;
	do {
		const char *z = (const char *)&list[i + 2];
		if (!strncmp(z, name, nameLength) && !z[nameLength]) return list[i];
		i += list[i + 1];
	} while (i < max);
	return 0;
}
