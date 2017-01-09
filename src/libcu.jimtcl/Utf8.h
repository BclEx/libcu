#ifndef UTF8_UTIL_H
#define UTF8_UTIL_H

#ifdef __cplusplus
extern "C" {
#endif

	// UTF-8 utility functions
	// 
	// (c) 2010 Steve Bennett <steveb@workware.net.au>
	// 
	// See LICENCE for licence details.
#include "jim-config.h"

#define MAX_UTF8_LEN 4 // Currently we support unicode points up to 2^22-1

	// Converts the given unicode codepoint (0 - 0x1fffff) to utf-8 and stores the result at 'p'.
	// Returns the number of utf-8 characters (up to MAX_UTF8_LEN).
	__device__ int utf8_fromunicode(char *p, unsigned uc);

#ifndef JIM_UTF8
#include <ctypecu.h>

	// No utf-8 support. 1 byte = 1 char
#define utf8_strlen(S, B) ((B) < 0 ? (int)strlen(S) : (B))
#define utf8_tounicode(S, CP) (*(CP) = (unsigned char)*(S), 1)
#define utf8_getchars(CP, C) (*(CP) = (C), 1)
#define utf8_upper(C) _toupper(C)
#define utf8_title(C) _toupper(C)
#define utf8_lower(C) _tolower(C)
#define utf8_index(C, I) (I)
#define utf8_charlen(C) 1
#define utf8_prev_len(S, L) 1

#else
#if !defined(JIM_BOOTSTRAP)
#define utf8_getchars utf8_fromunicode

	// Returns the length of the utf-8 sequence starting with 'c'.
	// Returns 1-4, or -1 if this is not a valid start byte.
	// Note that charlen=4 is not supported by the rest of the API.
	__device__ int utf8_charlen(int c);

	// Returns the number of characters in the utf-8 string of the given byte length.
	// Any bytes which are not part of an valid utf-8 sequence are treated as individual characters.
	// The string *must* be null terminated.
	// Does not support unicode code points > \u1fffff
	__device__ int utf8_strlen(const char *str, int bytelen);

	// Returns the byte index of the given character in the utf-8 string.
	// The string *must* be null terminated.
	// This will return the byte length of a utf-8 string if given the char length.
	__device__ int utf8_index(const char *str, int charindex);

	// Returns the unicode codepoint corresponding to the utf-8 sequence 'str'.
	// Stores the result in *uc and returns the number of bytes consumed.
	// If 'str' is null terminated, then an invalid utf-8 sequence at the end of the string will be returned as individual bytes.
	// If it is not null terminated, the length *must* be checked first.
	// Does not support unicode code points > \u1fffff
	__device__ int utf8_tounicode(const char *str, int *uc);

	// Returns the number of bytes before 'str' that the previous utf-8 character sequence starts (which may be the middle of a sequence).
	// Looks back at most 'len' bytes backwards, which must be > 0. If no start char is found, returns -len
	__device__ int utf8_prev_len(const char *str, int len);

	// Returns the upper-case variant of the given unicode codepoint.
	// Unicode code points > \uffff are returned unchanged.
	__device__ int utf8_upper(int uc);

	// Returns the title-case variant of the given unicode codepoint.
	// If none, returns utf8_upper().
	// Unicode code points > \uffff are returned unchanged.
	__device__ int utf8_title(int uc);

	// Returns the lower-case variant of the given unicode codepoint.
	// NOTE: Use utf8_upper() in preference for case-insensitive matching.
	// Unicode code points > \uffff are returned unchanged.
	__device__ int utf8_lower(int uc);
#endif

#endif

#ifdef __cplusplus
}
#endif

#endif
