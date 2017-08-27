#include <stringcu.h>
#include <stdlibcu.h>
#include <stdargcu.h>
#include <stddefcu.h>
#include <ctypecu.h>
#include <limits.h>
#include <assert.h>

__BEGIN_DECLS;

/* Copy N bytes of SRC to DEST.  */
//builtin: extern __device__ void *memcpy(void *__restrict dest, const void *__restrict src, size_t n);

/* Copy N bytes of SRC to DEST, guaranteeing correct behavior for overlapping strings.  */
__device__ void *memmove_(void *dest, const void *src, size_t n)
{
	if (!n) return dest;
	register unsigned char *a = (unsigned char *)dest;
	register unsigned char *b = (unsigned char *)src;
	if (a == b) return a; // No need to do that thing.
	if (a < b && b < a + n) { // Check for destructive overlap.
		a += n; b += n; // Destructive overlap ...
		while (n-- > 0) { *--a= *--b; } // have to copy backwards.
		return a;
	}
	while (n-- > 0) { *a++ = *b++; } // Do an ascending copy.
	return a;
}

/* Set N bytes of S to C.  */
//builtin: extern __device__ void *memset(void *s, int c, size_t n);

/* Compare N bytes of S1 and S2.  */
__device__ int memcmp_(const void *s1, const void *s2, size_t n)
{
	if (!n) return 0;
	register unsigned char *a = (unsigned char *)s1;
	register unsigned char *b = (unsigned char *)s2;
	while (--n > 0 && *a == *b) { a++; b++; }
	return *a - *b;
}

/* Search N bytes of S for C.  */
__device__ void *memchr_(const void *s, int c, size_t n)
{
	if (!n) return nullptr;
	register const char *p = (const char *)s;
	do {
		if (*p++ == c)
			return (void *)(p - 1);
	} while (--n > 0);
	return nullptr;
}

/* Copy SRC to DEST.  */
__device__ char *strcpy_(char *__restrict dest, const char *__restrict src)
{
	register unsigned char *d = (unsigned char *)dest;
	register unsigned char *s = (unsigned char *)src;
	while (*s) { *d++ = *s++; } *d = *s;
	return (char *)d;
}

/* Copy no more than N characters of SRC to DEST.  */
__device__ char *strncpy_(char *__restrict dest, const char *__restrict src, size_t n)
{
	register unsigned char *d = (unsigned char *)dest;
	register unsigned char *s = (unsigned char *)src;
	size_t i = 0;
	for (; i < n && *s; ++i, ++d, ++s) *d = *s;
	for (; i < n; ++i, ++d, ++s) *d = 0;
	return (char *)d;
}

/* Append SRC onto DEST.  */
__device__ char *strcat_(char *__restrict dest, const char *__restrict src)
{
	register unsigned char *d = (unsigned char *)dest;
	register unsigned char *s = (unsigned char *)src;
	while (*d) d++;
	while (*s) { *d++ = *s++; } *d = *s;
	return (char *)d;
}

/* Append no more than N characters from SRC onto DEST.  */
__device__ char *strncat_(char *__restrict dest, const char *__restrict src, size_t n)
{
	register unsigned char *d = (unsigned char *)dest;
	register unsigned char *s = (unsigned char *)src;
	while (*d) d++;
	while (*s && !--n) { *d++ = *s++; } *d = *s;
	return (char *)d;
}

/* Compare S1 and S2.  */
__device__ int strcmp_(const char *s1, const char *s2)
{
	register unsigned char *a = (unsigned char *)s1;
	register unsigned char *b = (unsigned char *)s2;
	while (*a && *a == *b) { a++; b++; }
	return *a - *b;
}
/* Compare S1 and S2. Case insensitive.  */
__device__ int stricmp_(const char *s1, const char *s2)
{
	register unsigned char *a = (unsigned char *)s1;
	register unsigned char *b = (unsigned char *)s2;
	while (*a && __curtUpperToLower[*a] == __curtUpperToLower[*b]) { a++; b++; }
	return __curtUpperToLower[*a] - __curtUpperToLower[*b];
}

/* Compare N characters of S1 and S2.  */
__device__ int strncmp_(const char *s1, const char *s2, size_t n)
{
	register unsigned char *a = (unsigned char *)s1;
	register unsigned char *b = (unsigned char *)s2;
	while (n-- > 0 && *a && *a == *b) { a++; b++; }
	return !n ? 0 : *a - *b;
}
/* Compare N characters of S1 and S2. Case insensitive.  */
__device__ int strnicmp_(const char *s1, const char *s2, size_t n)
{
	register unsigned char *a = (unsigned char *)s1;
	register unsigned char *b = (unsigned char *)s2;
	while (n-- > 0 && *a && __curtUpperToLower[*a] == __curtUpperToLower[*b]) { a++; b++; }
	return !n ? 0 : __curtUpperToLower[*a] - __curtUpperToLower[*b];
}

/* Compare the collated forms of S1 and S2.  */
__device__ int strcoll_(const char *s1, const char *s2)
{
	panic("Not Implemented");
	return -1;
}

/* Put a transformation of SRC into no more than N bytes of DEST.  */
__device__ size_t strxfrm_(char *__restrict dest, const char *__restrict src, size_t n)
{
	panic("Not Implemented");
	return 0;
}

/* Duplicate S, returning an identical malloc'd string.  */
__device__ char *strdup_(const char *s)
{
	const char *old = s;
	size_t len = strlen(old) + 1;
	char *new_ = (char *)malloc(len);
	(char *)memcpy(new_, old, len);	
	return new_;
}

/* Return a malloc'd copy of at most N bytes of STRING.  The resultant string is terminated even if no null terminator appears before STRING[N].  */
__device__ char *strndup_(const char *s, size_t n)
{
	const char *old = s;
	size_t len = strnlen(old, n);
	char *new_ = (char *)malloc(len + 1);
	new_[len] = '\0';
	(char *)memcpy(new_, old, len);	
	return new_;
}

/* Find the first occurrence of C in S.  */
__device__ char *strchr_(const char *s, int c)
{
	register unsigned char *s1 = (unsigned char *)s;
	register unsigned char l = (unsigned char)__curtUpperToLower[c];
	while (*s1 && __curtUpperToLower[*s1] != l) s1++;
	return (char *)(*s1 ? s1 : nullptr);
}

/* Find the last occurrence of C in S.  */
__device__ char *strrchr_(const char *s, int c)
{
	char *save; char c1;
	for (save = (char *)0; c1 = *s; s++)
		if (c1 == c)
			save = (char *)s;
	return save;
}

/* Return the length of the initial segment of S which consists entirely of characters not in REJECT.  */
__device__ size_t strcspn_(const char *s, const char *reject)
{
	panic("Not Implemented");
	return 0;
}

/* Return the length of the initial segment of S which consists entirely of characters in ACCEPT.  */
__device__ size_t strspn_(const char *s, const char *accept)
{
	panic("Not Implemented");
	return 0;
}

/* Find the first occurrence in S of any character in ACCEPT.  */
__device__ char *strpbrk_(const char *s, const char *accept)
{
	register const char *scanp;
	register int c, sc;
	while (c = *s++) {
		for (scanp = accept; sc = *scanp++;)
			if (sc == c)
				return (char *)(s - 1);
	}
	return nullptr;
}

/* Find the first occurrence of NEEDLE in HAYSTACK.  */
__device__ char *strstr_(const char *haystack, const char *needle)
{
	if (!*needle) return (char *)haystack;
	char *p1 = (char *)haystack, *p2 = (char *)needle;
	char *p1Adv = (char *)haystack;
	while (*++p2)
		p1Adv++;
	while (*p1Adv) {
		char *p1Begin = p1;
		p2 = (char *)needle;
		while (*p1 && *p2 && *p1 == *p2) {
			p1++;
			p2++;
		}
		if (!*p2)
			return p1Begin;
		p1 = p1Begin + 1;
		p1Adv++;
	}
	return nullptr;
}

/* Divide S into tokens separated by characters in DELIM.  */
__device__ char *strtok_(char *__restrict s, const char *__restrict delim)
{
	panic("Not Implemented");
	return nullptr;
}

/* inline: Return the length of S.  */
__device__ size_t strlen_(const char *s)
{
	if (!s) return 0;
	register const char *s2 = s;
	while (*s2) { s2++; }
	return 0x3fffffff & (int)(s2 - s);
}

/* inline: Return the length of S.  */
//__device__ size_t strlen16(const void *s)
//{
//	if (!s) return 0;
//	register const char *s2 = (const char *)s;
//	int n; for (n = 0; s2[n] || s2[n+1]; n += 2) { }
//	return n;
//}

/* Find the length of STRING, but scan at most MAXLEN characters. If no '\0' terminator is found in that many characters, return MAXLEN.  */
__device__ size_t strnlen_(const char *s, size_t maxlen)
{
	if (!s) return 0;
	register const char *s2 = s;
	register const char *s2m = s + maxlen;
	while (*s2 && s2 <= s2m) { s2++; }
	return 0x3fffffff & (int)(s2 - s);
}

__device__ void *mempcpy_(void *__restrict dest, const void *__restrict src, size_t n)
{
	panic("Not Implemented");
	return nullptr;
}

/* Return a string describing the meaning of the `errno' code in ERRNUM.  */
__device__ char *strerror_(int errnum)
{
	return "ERROR";
}

#pragma region strbld

#define BUFSIZE PRINT_BUF_SIZE  // Size of the output buffer

/* An "etByte" is an 8-bit unsigned value. */
typedef unsigned char etByte;
#define TYPE_RADIX		0 // non-decimal integer types.  %x %o
#define TYPE_FLOAT		1 // Floating point.  %f
#define TYPE_EXP		2 // Exponentional notation. %e and %E
#define TYPE_GENERIC	3 // Floating or exponential, depending on exponent. %g
#define TYPE_SIZE		4 // Return number of characters processed so far. %n
#define TYPE_STRING	5 // Strings. %s
#define TYPE_DYNSTRING	6 // Dynamically allocated strings. %z
#define TYPE_PERCENT	7 // Percent symbol. %%
#define TYPE_CHARX		8 // Characters. %c
// The rest are extensions, not normally found in printf()
#define TYPE_SQLESCAPE	9 // Strings with '\'' doubled.  %q
#define TYPE_SQLESCAPE2 10 // Strings with '\'' doubled and enclosed in '', NULL pointers replaced by SQL NULL.  %Q
#define TYPE_TOKEN		11 // a pointer to a Token structure
#define TYPE_SRCLIST	12 // a pointer to a SrcList
#define TYPE_POINTER	13 // The %p conversion
#define TYPE_SQLESCAPE3 14 // %w -> Strings with '\"' doubled
#define TYPE_ORDINAL	15 // %r -> 1st, 2nd, 3rd, 4th, etc.  English only
#define TYPE_DECIMAL	16 // %d or %u, but not %x, %o
//
#define TYPE_INVALID	17 // Any unrecognized conversion type

/* Allowed values for et_info.flags */
#define FLAG_SIGNED	1 // True if the value to convert is signed
#define FLAG_STRING	4 // Allow infinite precision

// Each builtin conversion character (ex: the 'd' in "%d") is described by an instance of the following structure
typedef struct info_t {
	// Information about each format field
	char fmtType;	// The format field code letter
	etByte base;	// The base for radix conversion
	etByte flags;	// One or more of FLAG_ constants below
	etByte type;	// Conversion paradigm
	etByte charset;	// Offset into aDigits[] of the digits string
	etByte prefix;	// Offset into aPrefix[] of the prefix string
} info_t;

// The following table is searched linearly, so it is good to put the most frequently used conversion types first.
__device__ static const char _digits[] = "0123456789ABCDEF0123456789abcdef";
__device__ static const char _prefix[] = "-x0\000X0";
__device__ static const info_t _info[] = {
	{ 'd', 10, 1, TYPE_DECIMAL,    0,  0 },
	{ 's',  0, 4, TYPE_STRING,     0,  0 },
	{ 'g',  0, 1, TYPE_GENERIC,    30, 0 },
	{ 'z',  0, 4, TYPE_DYNSTRING,  0,  0 },
	{ 'q',  0, 4, TYPE_SQLESCAPE,  0,  0 },
	{ 'Q',  0, 4, TYPE_SQLESCAPE2, 0,  0 },
	{ 'w',  0, 4, TYPE_SQLESCAPE3, 0,  0 },
	{ 'c',  0, 0, TYPE_CHARX,      0,  0 },
	{ 'o',  8, 0, TYPE_RADIX,      0,  2 },
	{ 'u', 10, 0, TYPE_DECIMAL,    0,  0 },
	{ 'x', 16, 0, TYPE_RADIX,      16, 1 },
	{ 'X', 16, 0, TYPE_RADIX,      0,  4 },
#ifndef OMIT_FLOATING_POINT
	{ 'f',  0, 1, TYPE_FLOAT,      0,  0 },
	{ 'e',  0, 1, TYPE_EXP,        30, 0 },
	{ 'E',  0, 1, TYPE_EXP,        14, 0 },
	{ 'G',  0, 1, TYPE_GENERIC,    14, 0 },
#endif
	{ 'i', 10, 1, TYPE_DECIMAL,    0,  0 },
	{ 'n',  0, 0, TYPE_SIZE,       0,  0 },
	{ '%',  0, 0, TYPE_PERCENT,    0,  0 },
	{ 'p', 16, 0, TYPE_POINTER,    0,  1 },
	// All the rest are undocumented and are for internal use only
	{ 'T',  0, 0, TYPE_TOKEN,      0,  0 },
	{ 'S',  0, 0, TYPE_SRCLIST,    0,  0 },
	{ 'r', 10, 1, TYPE_ORDINAL,    0,  0 },
};

#ifndef OMIT_FLOATING_POINT
static __device__ char getDigit(long_double *val, int *cnt)
{
	if (*cnt <= 0) return '0';
	(*cnt)--;
	int digit = (int)*val;
	long_double d = digit;
	digit += '0';
	*val = (*val - d)*10.0;
	return (char)digit;
}
#endif

/* Set the StrAccum object to an error mode. */
static __device__ void strbldSetError(strbld_t *b, unsigned char error)
{
	assert(error == STRACCUM_NOMEM || error == STRACCUM_TOOBIG);
	b->error = error;
	b->size = 0;
}

/* Render a string given by "fmt" into the strbld_t object. */
static __constant__ const char _ord[] = "thstndrd";
__device__ void strbldAppendFormat(strbld_t *b, const char *fmt, va_list va) //: sqlite3VXPrintf
{
	char buf[BUFSIZE]; // Conversion buffer
	char *bufpt = nullptr; // Pointer to the conversion buffer

	bool noArgs; void *args = nullptr; // Arguments for SQLITE_PRINTF_SQLFUNC
	if (b->flags & PRINTF_SQLFUNC) { noArgs = false; args = va_arg(va, void *); }
	else noArgs = true;

	int c; // Next character in the format string
	int width = 0; // Width of the current field
	int length = 0; // Length of the field
	etByte flag_leftjustify;	// True if "-" flag is present
	etByte flag_prefix;			// '+' or ' ' or 0 for prefix
	etByte flag_alternateform;	// True if "#" flag is present
	etByte flag_altform2;		// True if "!" flag is present
	etByte flag_zeropad;		// True if field width constant starts with zero
	etByte flag_long;			// 1 for the "l" flag, 2 for "ll", 0 by default
	etByte done;				// Loop termination flag
	etByte thousand;			// Thousands separator for %d and %u
	etByte type = TYPE_INVALID;// Conversion paradigm
	for (; (c = *fmt); ++fmt) {
		if (c != '%') {
			bufpt = (char *)fmt;
			do { fmt++; } while (*fmt && *fmt != '%');
			strbldAppend(b, bufpt, (int)(fmt - bufpt));
			if (!*fmt) break;
		}
		if (!(c = *++fmt)) {
			strbldAppend(b, "%", 1);
			break;
		}
		// Find out what flags are present
		flag_leftjustify = flag_prefix = thousand = flag_alternateform = flag_altform2 = flag_zeropad = 0;
		done = false; // Loop termination flag
		do {
			switch (c) {
			case '-': flag_leftjustify = true; break;
			case '+': flag_prefix = '+'; break;
			case ' ': flag_prefix = ' '; break;
			case '#': flag_alternateform = true; break;
			case '!': flag_altform2 = true; break;
			case '0': flag_zeropad = true; break;
			case ',': thousand = ','; break;
			default: done = true; break;
			}
		} while (!done && (c = *++fmt));
		// Get the field width
		if (c == '*') {
			width = noArgs ? va_arg(va, int) : (int)__extsystem.getIntegerArg(args);
			if (width < 0) {
				flag_leftjustify = true;
				width = width >= -2147483647 ? -width : 0;
			}
			c = *++fmt;
		}
		else {
			unsigned wx = 0;
			while (c >= '0' && c <= '9') {
				wx = wx*10 + c - '0';
				c = *++fmt;
			}
			ASSERTCOVERAGE(wx > 0x7fffffff);
			width = wx & 0x7fffffff;
		}
		assert(width >= 0);
#ifdef LIBCU_PRINTF_PRECISION_LIMIT
		if (width > LIBCU_PRINTF_PRECISION_LIMIT)
			width = LIBCU_PRINTF_PRECISION_LIMIT;
#endif

		// Get the precision
		int precision; // Precision of the current field
		if (c == '.') {
			c = *++fmt;
			if (c == '*') {
				precision = noArgs ? va_arg(va, int) : (int)__extsystem.getIntegerArg(args);
				c = *++fmt;
				if (precision < 0)
					precision = precision >= -2147483647 ? -precision : -1;
			}
			else {
				unsigned px = 0;
				while (c >= '0' && c <= '9') {
					px = px*10 + c - '0';
					c = *++fmt;
				}
				ASSERTCOVERAGE(px > 0x7fffffff);
				precision = px & 0x7fffffff;
			}
		}
		else precision = -1;
		assert(precision >= -1);
#ifdef LIBCU_PRINTF_PRECISION_LIMIT
		if (precision > LIBCU_PRINTF_PRECISION_LIMIT)
			precision = LIBCU_PRINTF_PRECISION_LIMIT;
#endif

		// Get the conversion type modifier
		if (c == 'l') {
			flag_long = 1;
			c = *++fmt;
			if (c == 'l') {
				flag_long = 2;
				c = *++fmt;
			}
		}
		else flag_long = 0;
		// Fetch the info entry for the field
		const info_t *info = &_info[0]; // Pointer to the appropriate info structure
		type = TYPE_INVALID; // Conversion paradigm
		int idx;
		for (idx = 0; idx < _LENGTHOF(_info); idx++) {
			if (c == _info[idx].fmtType) {
				info = &_info[idx];
				type = info->type;
				break;
			}
		}

		// At this point, variables are initialized as follows:
		//   flag_alternateform          TRUE if a '#' is present.
		//   flag_altform2               TRUE if a '!' is present.
		//   flag_prefix                 '+' or ' ' or zero
		//   flag_leftjustify            TRUE if a '-' is present or if the field width was negative.
		//   flag_zeropad                TRUE if the width began with 0.
		//   flag_long                   1 for "l", 2 for "ll"
		//   width                       The specified field width.  This is always non-negative.  Zero is the default.
		//   precision                   The specified precision.  The default is -1.
		//   type                        The class of the conversion.
		//   info                        Pointer to the appropriate info struct.
		char prefix; // Prefix character.  "+" or "-" or " " or '\0'.
		unsigned long long longvalue; // Value for integer types
		long_double realvalue; // Value for real types
		char *extra = nullptr; // Malloced memory used by some conversion
		char *out; // Rendering buffer
		int outLength; // Size of the rendering buffer
#ifndef OMIT_FLOATING_POINT
		int exp, e2; // exponent of real numbers
		int nsd; // Number of significant digits returned
		double rounder; // Used for rounding floating point values
		etByte flag_dp; // True if decimal point should be shown
		etByte flag_rtz; // True if trailing zeros should be removed
#endif
		switch (type) {
		case TYPE_POINTER:
			flag_long = sizeof(char *) == sizeof(int64_t) ? 2 :
				sizeof(char *) == sizeof(long int) ? 1 : 0;
			// Fall through into the next case
		case TYPE_ORDINAL:
		case TYPE_RADIX:
			thousand = 0;
			// Fall through into the next case
		case TYPE_DECIMAL:
			if (info->flags & FLAG_SIGNED) {
				int64_t v = noArgs ? 
					flag_long ? (flag_long == 2 ? va_arg(va, int64_t) : va_arg(va, long int)) : va_arg(va, int) :
					__extsystem.getIntegerArg(args);
				if (v < 0) { longvalue = (v == LLONG_MIN ? ((uint64_t)1)<<63 : -v); prefix = '-'; }
				else { longvalue = v; prefix = flag_prefix; }
			}
			else {
				longvalue = noArgs ? 
					flag_long ? (flag_long == 2 ? va_arg(va, uint64_t) : va_arg(va, unsigned long int)) : va_arg(va, unsigned int) :
					(uint64_t)__extsystem.getIntegerArg(args);
				prefix = 0;
			}
			if (longvalue == 0) flag_alternateform = false;
			if (flag_zeropad && precision < width-(prefix!=0))
				precision = width-(prefix!=0);
			if (precision < BUFSIZE-10-BUFSIZE/3) {
				outLength = BUFSIZE;
				out = buf;
			}
			else {
				uint64_t n = (uint64_t)precision + 10 + precision/3;
				out = extra = (char *)malloc(outLength);
				if (!out) {
					strbldSetError(b, STRACCUM_NOMEM);
					return;
				}
				outLength = (int)n;
			}
			bufpt = &out[outLength-1];
			if (type == TYPE_ORDINAL) {
				int x = (int)(longvalue % 10);
				if (x >= 4 || (longvalue/10)%10 == 1) x = 0;
				*(--bufpt) = _ord[x*2+1];
				*(--bufpt) = _ord[x*2];
			}
			{
				const char *cset = &_digits[info->charset]; // Use registers for speed
				etByte base = info->base;
				do { *(--bufpt) = cset[longvalue%base]; longvalue = longvalue/base; } while (longvalue > 0); // Convert to ascii
			}
			length = (int)(&out[outLength-1]-bufpt);
			while (precision > length) { *(--bufpt) = '0'; length++; } // Zero pad
			if (thousand) {
				int nn = (length-1)/3; // Number of "," to insert
				int ix = (length-1)%3 + 1;
				bufpt -= nn;
				for (idx = 0; nn > 0; idx++) {
					bufpt[idx] = bufpt[idx+nn];
					ix--;
					if (!ix) { bufpt[++idx] = thousand; nn--; ix = 3; }
				}
			}
			if (prefix) *(--bufpt) = prefix; // Add sign
			if (flag_alternateform && info->prefix) { // Add "0" or "0x"
				const char *pre = &_prefix[info->prefix];
				char x; for (; (x = *pre); pre++) *(--bufpt) = x;
			}
			length = (int)(&out[outLength-1]-bufpt);
			break;
		case TYPE_FLOAT:
		case TYPE_EXP:
		case TYPE_GENERIC:
			realvalue = noArgs ? va_arg(va, double) : __extsystem.getDoubleArg(args);
#ifdef OMIT_FLOATING_POINT
			length = 0;
#else
			if (precision < 0) precision = 6; // Set default precision
			if (realvalue < 0.0) { realvalue = -realvalue; prefix = '-'; }
			else prefix = flag_prefix;
			if (type == TYPE_GENERIC && precision > 0) precision--;
			ASSERTCOVERAGE(precision > 0xfff);
			for (idx = precision&0xfff, rounder = 0.5; idx > 0; idx--, rounder *= 0.1) { }
			if (type == TYPE_FLOAT) realvalue += rounder;
			// Normalize realvalue to within 10.0 > realvalue >= 1.0
			exp = 0;
			if (isnan((double)realvalue)) {
				bufpt = "NaN";
				length = 3;
				break;
			}
			if (realvalue > 0.0) {
				long_double scale = 1.0;
				while (realvalue >= 1e100*scale && exp <= 350) { scale *= 1e100; exp += 100; }
				while (realvalue >= 1e10*scale && exp<=350) { scale *= 1e10; exp += 10; }
				while (realvalue >= 10.0*scale && exp<=350) { scale *= 10.0; exp++; }
				realvalue /= scale;
				while (realvalue < 1e-8) { realvalue *= 1e8; exp -= 8; }
				while (realvalue < 1.0) { realvalue *= 10.0; exp--; }
				if (exp > 350) {
					bufpt = buf;
					buf[0] = prefix;
					memcpy(buf+(prefix!=0), "Inf", 4);
					length = 3+(prefix!=0);
					break;
				}
			}
			bufpt = buf;
			// If the field type is etGENERIC, then convert to either etEXP or etFLOAT, as appropriate.
			if (type != TYPE_FLOAT) {
				realvalue += rounder;
				if (realvalue >= 10.0) { realvalue *= 0.1; exp++; }
			}
			if (type == TYPE_GENERIC) {
				flag_rtz = !flag_alternateform;
				if (exp < -4 || exp > precision) type = TYPE_EXP;
				else { precision = precision - exp; type = TYPE_FLOAT; }
			}
			else flag_rtz = flag_altform2;
			e2 = (type == TYPE_EXP ? 0 : exp);
			if (_MAX(e2,0)+(int64_t)precision+(int64_t)width > BUFSIZE - 15) {
				bufpt = extra = (char *)malloc(_MAX(e2,0)+(int64_t)precision+(int64_t)width+15);
				if (!bufpt) {
					strbldSetError(b, STRACCUM_NOMEM);
					return;
				}
			}
			out = bufpt;
			nsd = 16 + flag_altform2*10;
			flag_dp = (precision>0?1:0)|flag_alternateform|flag_altform2;
			// The sign in front of the number
			if (prefix) *(bufpt++) = prefix;
			// Digits prior to the decimal point
			if (e2 < 0) *(bufpt++) = '0';
			else for (; e2 >= 0; e2--) *(bufpt++) = getDigit(&realvalue, &nsd);
			// The decimal point
			if (flag_dp) *(bufpt++) = '.';
			// "0" digits after the decimal point but before the first significant digit of the number
			for (e2++; e2 < 0; precision--, e2++) { assert(precision > 0); *(bufpt++) = '0'; }
			// Significant digits after the decimal point
			while (precision-- > 0) *(bufpt++) = getDigit(&realvalue, &nsd);
			// Remove trailing zeros and the "." if no digits follow the "."
			if (flag_rtz && flag_dp) {
				while (bufpt[-1] == '0') *(--bufpt) = 0;
				assert(bufpt > out);
				if (bufpt[-1] == '.') {
					if (flag_altform2) *(bufpt++) = '0';
					else *(--bufpt) = 0;
				}
			}
			// Add the "eNNN" suffix
			if (type == TYPE_EXP) {
				*(bufpt++) = _digits[info->charset];
				if (exp < 0) { *(bufpt++) = '-'; exp = -exp; }
				else *(bufpt++) = '+';
				if (exp >= 100) { *(bufpt++) = (char)(exp/100+'0'); exp %= 100; } // 100's digit
				*(bufpt++) = (char)(exp/10+'0'); // 10's digit
				*(bufpt++) = (char)(exp%10+'0'); // 1's digit
			}
			*bufpt = 0;

			// The converted number is in buf[] and zero terminated. Output it. Note that the number is in the usual order, not reversed as with integer conversions.
			length = (int)(bufpt-out);
			bufpt = out;

			// Special case:  Add leading zeros if the flag_zeropad flag is set and we are not left justified
			if (flag_zeropad && !flag_leftjustify && length < width) {
				int pad = width - length;
				for (idx = width; idx >= pad; idx--) bufpt[idx] = bufpt[idx-pad];
				idx = (prefix!=0);
				while (pad--) bufpt[idx++] = '0';
				length = width;
			}
#endif // !defined(OMIT_FLOATING_POINT)
			break;
		case TYPE_SIZE:
			if (noArgs)
				*(va_arg(va, int*)) = (int)b->size;
			length = width = 0;
			break;
		case TYPE_PERCENT:
			buf[0] = '%';
			bufpt = buf;
			length = 1;
			break;
		case TYPE_CHARX:
			if (noArgs) c = va_arg(va, int);
			else { bufpt = __extsystem.getStringArg(args); c = bufpt ? bufpt[0] : 0; }
			if (precision > 1) {
				width -= precision-1;
				if (width > 1 && !flag_leftjustify) {
					strbldAppendChar(b, width-1, ' ');
					width = 0;
				}
				strbldAppendChar(b, precision-1, c);
			}
			length = 1;
			buf[0] = (char)c;
			bufpt = buf;
			break;
		case TYPE_STRING:
		case TYPE_DYNSTRING:
			if (noArgs) bufpt = va_arg(va, char*);
			else { bufpt = __extsystem.getStringArg(args); type = TYPE_STRING; }
			if (!bufpt) bufpt = "";
			else if (type == TYPE_DYNSTRING) extra = bufpt;
			if (precision >= 0) for (length = 0; length < precision && bufpt[length]; length++) { }
			else length = strlen(bufpt);
			break;
		case TYPE_SQLESCAPE:
		case TYPE_SQLESCAPE2:
		case TYPE_SQLESCAPE3: {
			char q = (type == TYPE_SQLESCAPE3 ? '"' : '\''); // Quote character
			char *escarg = noArgs ? va_arg(va, char*) : __extsystem.getStringArg(args);
			bool isnull = !escarg;
			if (isnull) escarg = (type == TYPE_SQLESCAPE2 ? "NULL" : "(NULL)");
			int k = precision;
			int i, j, n;
			char ch;
			for (i = n = 0; k != 0 && (ch = escarg[i]) != 0; i++, k--)
				if (ch == q) n++;
			bool needQuote = !isnull && type == TYPE_SQLESCAPE2;
			n += i + 3;
			if (n > BUFSIZE) {
				bufpt = extra = (char *)malloc(n);
				if (!bufpt) {
					strbldSetError(b, STRACCUM_NOMEM);
					return;
				}
			}
			else bufpt = buf;
			j = 0;
			if (needQuote) bufpt[j++] = q;
			k = i;
			for (i = 0; i < k; i++) {
				bufpt[j++] = ch = escarg[i];
				if (ch == q) bufpt[j++] = ch;
			}
			if (needQuote) bufpt[j++] = q;
			bufpt[j] = 0;
			length = j;
			// The precision in %q and %Q means how many input characters to consume, not the length of the output...
			// if (precision>=0 && precision<length) length = precision;
			break; }
		case TYPE_TOKEN: {
			if (!(b->flags & PRINTF_INTERNAL)) return;
			__extsystem.appendFormat[0](b, &va);
			length = width = 0;
			break; }
		case TYPE_SRCLIST: {
			if (!(b->flags & PRINTF_INTERNAL)) return;
			__extsystem.appendFormat[1](b, &va);
			length = width = 0;
			break; }
		default: {
			assert(type == TYPE_INVALID);
			return; }
		}
		// The text of the conversion is pointed to by "bufpt" and is "length" characters long.  The field width is "width".  Do the output.
		width -= length;
		if (width > 0) {
			if (!flag_leftjustify) strbldAppendChar(b, width, ' ');
			strbldAppend(b, bufpt, length);
			if (flag_leftjustify) strbldAppendChar(b, width, ' ');
		}
		else strbldAppend(b, bufpt, length);
		//
		if (extra) { free(extra); extra = nullptr; }
	}
}

/*
** Enlarge the memory allocation on a StrAccum object so that it is able to accept at least N more bytes of text.
**
** Return the number of bytes of text that StrAccum is able to accept after the attempted enlargement.  The value returned might be zero.
*/
static __device__ int strbldEnlarge(strbld_t *b, int n)
{
	assert(b->index+(int64_t)n >= b->size); // Only called if really needed
	if (b->error) {
		ASSERTCOVERAGE(b->error == STRACCUM_TOOBIG);
		ASSERTCOVERAGE(b->error == STRACCUM_NOMEM);
		return 0;
	}
	if (!b->maxSize) {
		n = b->size - b->index - 1;
		strbldSetError(b, STRACCUM_TOOBIG);
		return n;
	}
	char *oldText = PRINTF_ISMALLOCED(b) ? b->text : nullptr;
	int64_t sizeNew = b->index;
	assert((!b->text || b->text == b->base) == !PRINTF_ISMALLOCED(b));
	sizeNew += n + 1;
	if (sizeNew+b->index <= b->maxSize) // Force exponential buffer size growth as long as it does not overflow, to avoid having to call this routine too often
		sizeNew += b->index;
	if (sizeNew > b->maxSize) {
		strbldReset(b);
		strbldSetError(b, STRACCUM_TOOBIG);
		return 0;
	}
	else b->size = (int)sizeNew;
	char *newText = (char *)(b->tag ? __extsystem.tagrealloc(b->tag, oldText, b->size) : realloc(oldText, b->size));
	if (newText) {
		assert( b->text || !b->index);
		if (!PRINTF_ISMALLOCED(b) && b->index > 0) memcpy(newText, b->text, b->index);
		b->text = newText;
		b->size = b->tag ? (size_t)__extsystem.tagallocSize(b->tag, newText) : _msize(newText);
		b->flags |= PRINTF_MALLOCED;
	}
	else {
		strbldReset(b);
		strbldSetError(b, STRACCUM_NOMEM);
		return 0;
	}
	return n;
}

/* Append N copies of character c to the given string buffer. */
__device__ void strbldAppendChar(strbld_t *b, int n, int c) //: sqlite3AppendChar
{
	ASSERTCOVERAGE(b->size+(int64_t)n > 0x7fffffff);
	if (b->index+(int64_t)n >= b->size && (n = strbldEnlarge(b, n)) <= 0)
		return;
	assert((b->text == b->base) == !PRINTF_ISMALLOCED(b));
	while (n-- > 0) b->text[b->index++] = c;
}

/*
** The StrAccum "b" is not large enough to accept N new bytes of str[]. So enlarge if first, then do the append.
**
** This is a helper routine to sqlite3StrAccumAppend() that does special-case work (enlarging the buffer) using tail recursion, so that the
** sqlite3StrAccumAppend() routine can use fast calling semantics.
*/
static __device__ void enlargeAndAppend(strbld_t *b, const char *str, int length)
{
	length = strbldEnlarge(b, length);
	if (length > 0) {
		memcpy(&b->text[b->index], str, length);
		b->index += length;
	}
	assert((!b->text || b->text == b->base) == !PRINTF_ISMALLOCED(b));
}

/* Append N bytes of text from str to the StrAccum object.  Increase the size of the memory allocation for StrAccum if necessary. */
__device__ void strbldAppend(strbld_t *b, const char *str, int length) //: sqlite3StrAccumAppend
{
	assert(str || length == 0);
	assert(b->text || !b->index || b->error);
	assert(length >= 0);
	assert(!b->error || !b->size);
	if (b->index+length >= b->size)
		enlargeAndAppend(b, str, length);
	else if (length) {
		assert(b->text);
		b->index += length;
		memcpy(&b->text[b->index-length], str, length);
	}
}

/* Append the complete text of zero-terminated string str[] to the b string. */
__device__ void strbldAppendAll(strbld_t *b, const char *str) //: sqlite3StrAccumAppendAll
{
	strbldAppend(b, str, strlen(str));
}

/*
** Finish off a string by making sure it is zero-terminated. Return a pointer to the resulting string.  Return a NULL
** pointer if any kind of error was encountered.
*/
static __device__ char *strbldFinishRealloc(strbld_t *b)
{
	assert(b->maxSize > 0 && !PRINTF_ISMALLOCED(b));
	b->text = (char *)(b->tag ? __extsystem.tagallocRaw(b->tag, b->index+1) : malloc(b->index+1));
	if (b->text) {
		memcpy(b->text, b->base, b->index+1);
		b->flags |= PRINTF_MALLOCED;
	}
	else strbldSetError(b, STRACCUM_NOMEM);
	return b->text;
}
__device__ char *strbldToString(strbld_t *b) //: sqlite3StrAccumFinish
{
	if (b->text) {
		assert((b->text == b->base) == !PRINTF_ISMALLOCED(b));
		b->text[b->index] = 0;
		if (b->maxSize > 0 && !PRINTF_ISMALLOCED(b))
			return strbldFinishRealloc(b);
	}
	return b->text;
}

/*
** Reset an StrAccum string.  Reclaim all malloced memory.
*/
__device__ void strbldReset(strbld_t *b) //: sqlite3StrAccumReset
{
	assert((!b->text || b->text == b->base) == !PRINTF_ISMALLOCED(b));
	if (PRINTF_ISMALLOCED(b)) {
		if (b->tag) __extsystem.tagfree(b->tag, b->text);
		else free(b->text);
		b->flags &= ~PRINTF_MALLOCED;
	}
	b->text = nullptr;
}

/*
** Initialize a string accumulator.
**
** b: The accumulator to be initialized.
** tag: Pointer to a database connection.  May be NULL.  Lookaside memory is used if not NULL. db->mallocFailed is set appropriately when not NULL.
** base: An initial buffer.  May be NULL in which case the initial buffer is malloced.
** capacity: Size of zBase in bytes.  If total space requirements never exceed n then no memory allocations ever occur.
** maxSize: Maximum number of bytes to accumulate.  If mx==0 then no memory allocations will ever occur.
*/
__device__ void strbldInit(strbld_t *b, void *tag, char *base, int capacity, int maxSize)
{
	b->text = b->base = base;
	b->tag = tag;
	b->index = 0;
	b->size = capacity;
	b->maxSize = maxSize;
	b->error = 0;
	b->flags = 0;
}

#pragma endregion

__END_DECLS;