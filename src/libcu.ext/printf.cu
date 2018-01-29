#include <stringcu.h>
#include <stdiocu.h>
#include <ext/global.h> //: printf.c
#include <assert.h>

#pragma region From: printf.c

#define LIBCU_MAXLENGTH 1000000000

__host_device__ char *vmtagprintf(tagbase_t *tag, const char *format, va_list va) //: sqlite3VMPrintf, sqlite3MPrintf
{
	assert(tag != nullptr);
	char base[PRINT_BUF_SIZE];
	strbld_t b;
	strbldInit(&b, tag, base, sizeof(base), tag->limits[TAG_LIMIT_LENGTH]);
	b.flags = PRINTF_INTERNAL;
	strbldAppendFormatv(&b, format, va);
	char *str = strbldToString(&b);
	if (b.error == STRACCUM_NOMEM)
		tagOomFault(tag);
	return str;
}

__host_device__ char *vmprintf(const char *format, va_list va) //: sqlite3_vmprintf, sqlite3_mprintf
{
#ifdef ENABLE_API_ARMOR
	if (!format) { (void)RC_MISUSE_BKPT; return nullptr; }
#endif
#ifndef OMIT_AUTOINIT
	if (runtimeInitialize()) return nullptr;
#endif
	char base[PRINT_BUF_SIZE];
	strbld_t b;
	strbldInit(&b, nullptr, base, sizeof(base), LIBCU_MAXLENGTH);
	strbldAppendFormatv(&b, format, va);
	return strbldToString(&b);
}

__host_device__ char *vmsnprintf(char *__restrict s, size_t maxlen, const char *format, va_list va) //: sqlite3_vsnprintf, sqlite3_snprintf
{
	if (maxlen <= 0) return (char *)s;
#ifdef ENABLE_API_ARMOR
	if (!s || !format) { (void)RC_MISUSE_BKPT; if (s) s[0] = 0; return s; }
#endif
	strbld_t b;
	strbldInit(&b, nullptr, (char *)s, (int)maxlen, 0);
	strbldAppendFormatv(&b, format, va);
	return strbldToString(&b);
}

/* This is the routine that actually formats the sqlite3_log() message. We house it in a separate routine from sqlite3_log() to avoid using
** stack space on small-stack systems when logging is disabled.
**
** sqlite3_log() must render into a static buffer.  It cannot dynamically allocate memory because it might be called while the memory allocator
** mutex is held.
**
** sqlite3VXPrintf() might ask for *temporary* memory allocations for certain format characters (%q) or for very large precisions or widths.
** Care must be taken that any sqlite3_log() calls that occur while the memory mutex is held do not use these mechanisms.
*/
static __host_device__ void renderLogMsg(int errCode, const char *format, va_list va)
{
	char base[PRINT_BUF_SIZE * 3];
	strbld_t b;
	strbldInit(&b, nullptr, base, sizeof(base), 0);
	strbldAppendFormatv(&b, format, va);
	_runtimeConfig.log(_runtimeConfig.logArg, errCode, strbldToString(&b));
}

/* Format and write a message to the log if logging is enabled. */
__host_device__ void _logv(int errCode, const char *format, va_list va) //: sqlite3_log
{
	if (_runtimeConfig.log)
		renderLogMsg(errCode, format, va);
}

#if defined(_DEBUG) || defined(LIBCU_HAVE_OS_TRACE)
/* A version of printf() that understands %lld.  Used for debugging. The printf() built into some versions of windows does not understand %lld
** and segfaults if you give it a long long int.
*/
__host_device__ void _debugv(const char *format, va_list va) //: sqlite3DebugPrintf
{
	char base[500];
	strbld_t b;
	strbldInit(&b, nullptr, base, sizeof(base), 0);
	strbldAppendFormatv(&b, format, va);
	strbldToString(&b);
	printf("%s", base);
	//fprintf(stdout, "%s", base);
	//fflush(stdout);
}
#endif

#pragma endregion