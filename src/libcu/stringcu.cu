#include <stdlibcu.h>
#include <stringcu.h>
#include <ctypecu.h>
#include <limits.h>
#include <assert.h>
#include <cuda_runtimecu.h>

__BEGIN_DECLS;

/* Copy N bytes of SRC to DEST.  */
//builtin: extern __device__ void *memcpy(void *__restrict dest, const void *__restrict src, size_t n);

/* Copy N bytes of SRC to DEST, guaranteeing correct behavior for overlapping strings.  */
__device__ void *memmove(void *dest, const void *src, size_t n)
{
	if (!n)
		return dest;
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
__device__ int memcmp(const void *s1, const void *s2, size_t n)
{
	if (!n)
		return 0;
	register unsigned char *a = (unsigned char *)s1;
	register unsigned char *b = (unsigned char *)s2;
	while (--n > 0 && *a == *b) { a++; b++; }
	return *a - *b;
}

/* Search N bytes of S for C.  */
__device__ void *memchr(const void *s, int c, size_t n)
{
	if (!n)
		return nullptr;
	register const char *p = (const char *)s;
	do {
		if (*p++ == c)
			return (void *)(p - 1);
	} while (--n > 0);
	return nullptr;
}

/* Copy SRC to DEST.  */
__device__ char *strcpy(char *__restrict dest, const char *__restrict src)
{
	register unsigned char *d = (unsigned char *)dest;
	register unsigned char *s = (unsigned char *)src;
	while (*s) { *d++ = *s++; } *d = *s;
	return (char *)d;
}

/* Copy no more than N characters of SRC to DEST.  */
__device__ char *strncpy(char *__restrict dest, const char *__restrict src, size_t n)
{
	register unsigned char *d = (unsigned char *)dest;
	register unsigned char *s = (unsigned char *)src;
	size_t i = 0;
	for (; i < n && *s; ++i, ++d, ++s) *d = *s;
	for (; i < n; ++i, ++d, ++s) *d = 0;
	return (char *)d;
}

/* Append SRC onto DEST.  */
__device__ char *strcat(char *__restrict dest, const char *__restrict src)
{
	register unsigned char *d = (unsigned char *)dest;
	while (*d) d++;
	register unsigned char *s = (unsigned char *)src;
	while (*s) { *d++ = *s++; } *d = *s;
	return (char *)d;
}

/* Append no more than N characters from SRC onto DEST.  */
__device__ char *strncat(char *__restrict dest, const char *__restrict src, size_t n)
{
	register unsigned char *d = (unsigned char *)dest;
	while (*d) d++;
	register unsigned char *s = (unsigned char *)src;
	while (*s && !--n) { *d++ = *s++; } *d = *s;
	return (char *)d;
}

/* Compare S1 and S2.  */
__device__ int strcmp(const char *s1, const char *s2)
{
	register unsigned char *a = (unsigned char *)s1;
	register unsigned char *b = (unsigned char *)s2;
	while (*a && __curtUpperToLower[*a] == __curtUpperToLower[*b]) { a++; b++; }
	return __curtUpperToLower[*a] - __curtUpperToLower[*b];
}

/* Compare N characters of S1 and S2.  */
__device__ int strncmp(const char *s1, const char *s2, size_t n)
{
	register unsigned char *a = (unsigned char *)s1;
	register unsigned char *b = (unsigned char *)s2;
	while (n-- > 0 && *a && __curtUpperToLower[*a] == __curtUpperToLower[*b]) { a++; b++; }
	return (!n ? 0 : __curtUpperToLower[*a] - __curtUpperToLower[*b]);
}

/* Compare the collated forms of S1 and S2.  */
__device__ int strcoll(const char *s1, const char *s2)
{
	panic("Not Implemented");
	return -1;
}

/* Put a transformation of SRC into no more than N bytes of DEST.  */
__device__ size_t strxfrm(char *__restrict dest, const char *__restrict src, size_t n)
{
	panic("Not Implemented");
	return 0;
}

/* Find the first occurrence of C in S.  */
__device__ char *strchr(const char *s, int c)
{
	register unsigned char *s1 = (unsigned char *)s;
	register unsigned char l = (unsigned char)__curtUpperToLower[c];
	while (*s1 && __curtUpperToLower[*s1] != l) s++;
	return (char *)(*s1 ? s1 : nullptr);
}

/* Find the last occurrence of C in S.  */
__device__ char *strrchr(const char *s, int c)
{
	char *save;
	char c1;
	for (save = (char *)0; c1 = *s; s++)
		if (c1 == c)
			save = (char *)s;
	return save;
}

/* Return the length of the initial segment of S which consists entirely of characters not in REJECT.  */
__device__ size_t strcspn(const char *s, const char *reject)
{
	panic("Not Implemented");
	return 0;
}

/* Return the length of the initial segment of S which consists entirely of characters in ACCEPT.  */
__device__ size_t strspn(const char *s, const char *accept)
{
	panic("Not Implemented");
	return 0;
}

/* Find the first occurrence in S of any character in ACCEPT.  */
__device__ char *strpbrk(const char *s, const char *accept)
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
__device__ char *strstr(const char *haystack, const char *needle)
{
	if (!*needle)
		return (char *)haystack;
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
extern __device__ char *strtok(char *__restrict s, const char *__restrict delim)
{
	panic("Not Implemented");
	return nullptr;
}

/* inline: Return the length of S.  */
//__device__ size_t strlen(const char *s)
//{
//	if (!s)
//		return 0;
//	register const char *s2 = s;
//	while (*s2) { s2++; }
//	return 0x3fffffff & (int)(s2 - s);
//}

extern __device__ void *mempcpy(void *__restrict dest, const void *__restrict src, size_t n)
{
	panic("Not Implemented");
	return nullptr;
}

/* Return a string describing the meaning of the `errno' code in ERRNUM.  */
__device__ char *strerror(int errnum)
{
	return "ERROR";
}

#pragma region strbld

#define BUFSIZE PRINT_BUF_SIZE  // Size of the output buffer

enum TYPE : unsigned char
{
	TYPE_RADIX = 1,			// Integer types.  %d, %x, %o, and so forth
	TYPE_FLOAT = 2,			// Floating point.  %f
	TYPE_EXP = 3,			// Exponentional notation. %e and %E
	TYPE_GENERIC = 4,		// Floating or exponential, depending on exponent. %g
	TYPE_SIZE = 5,			// Return number of characters processed so far. %n
	TYPE_STRING = 6,		// Strings. %s
	TYPE_DYNSTRING = 7,		// Dynamically allocated strings. %z
	TYPE_PERCENT = 8,		// Percent symbol. %%
	TYPE_CHARX = 9,			// Characters. %c
	// The rest are extensions, not normally found in printf()
	TYPE_SQLESCAPE = 10,	// Strings with '\'' doubled.  %q
	TYPE_SQLESCAPE2 = 11,	// Strings with '\'' doubled and enclosed in '', NULL pointers replaced by SQL NULL.  %Q
	TYPE_TOKEN = 12,		// a pointer to a Token structure
	TYPE_SRCLIST = 13,		// a pointer to a SrcList
	TYPE_POINTER = 14,		// The %p conversion
	TYPE_SQLESCAPE3 = 15,	// %w -> Strings with '\"' doubled
	TYPE_ORDINAL = 16,		// %r -> 1st, 2nd, 3rd, 4th, etc.  English only
	//
	TYPE_INVALID = 0,		// Any unrecognized conversion type
};

enum FLAG : unsigned char
{
	FLAG_SIGNED = 1,	// True if the value to convert is signed
	FLAG_INTERN = 2,	// True if for internal use only
	FLAG_STRING = 4,	// Allow infinity precision
};

// Each builtin conversion character (ex: the 'd' in "%d") is described by an instance of the following structure
typedef struct info_t
{   // Information about each format field
	char fmtType; // The format field code letter
	unsigned char base; // The base for radix conversion
	FLAG flags; // One or more of FLAG_ constants below
	TYPE type; // Conversion paradigm
	unsigned char charset; // Offset into aDigits[] of the digits string
	unsigned char prefix; // Offset into aPrefix[] of the prefix string
} info_t;

// The following table is searched linearly, so it is good to put the most frequently used conversion types first.
__device__ static const char _digits[] = "0123456789ABCDEF0123456789abcdef";
__device__ static const char _prefix[] = "-x0\000X0";
__device__ static const info_t _info[] = {
	{ 'd', 10, (FLAG)1, TYPE_RADIX,      0,  0 },
	{ 's',  0, (FLAG)4, TYPE_STRING,     0,  0 },
	{ 'g',  0, (FLAG)1, TYPE_GENERIC,    30, 0 },
	{ 'z',  0, (FLAG)4, TYPE_DYNSTRING,  0,  0 },
	{ 'q',  0, (FLAG)4, TYPE_SQLESCAPE,  0,  0 },
	{ 'Q',  0, (FLAG)4, TYPE_SQLESCAPE2, 0,  0 },
	{ 'w',  0, (FLAG)4, TYPE_SQLESCAPE3, 0,  0 },
	{ 'c',  0, (FLAG)0, TYPE_CHARX,      0,  0 },
	{ 'o',  8, (FLAG)0, TYPE_RADIX,      0,  2 },
	{ 'u', 10, (FLAG)0, TYPE_RADIX,      0,  0 },
	{ 'x', 16, (FLAG)0, TYPE_RADIX,      16, 1 },
	{ 'X', 16, (FLAG)0, TYPE_RADIX,      0,  4 },
#ifndef OMIT_FLOATING_POINT
	{ 'f',  0, (FLAG)1, TYPE_FLOAT,      0,  0 },
	{ 'e',  0, (FLAG)1, TYPE_EXP,        30, 0 },
	{ 'E',  0, (FLAG)1, TYPE_EXP,        14, 0 },
	{ 'G',  0, (FLAG)1, TYPE_GENERIC,    14, 0 },
#endif
	{ 'i', 10, (FLAG)1, TYPE_RADIX,      0,  0 },
	{ 'n',  0, (FLAG)0, TYPE_SIZE,       0,  0 },
	{ '%',  0, (FLAG)0, TYPE_PERCENT,    0,  0 },
	{ 'p', 16, (FLAG)0, TYPE_POINTER,    0,  1 },
	// All the rest have the FLAG_INTERN bit set and are thus for internal use only
	{ 'T',  0, (FLAG)2, TYPE_TOKEN,      0,  0 },
	{ 'S',  0, (FLAG)2, TYPE_SRCLIST,    0,  0 },
	{ 'r', 10, (FLAG)3, TYPE_ORDINAL,    0,  0 },
};

#ifndef OMIT_FLOATING_POINT
__device__ static char GetDigit(double64 *val, int *cnt)
{
	if (*cnt <= 0) return '0';
	(*cnt)--;
	int digit = (int)*val;
	double64 d = digit;
	digit += '0';
	*val = (*val - d)*10.0;
	return (char)digit;
}
#endif

__device__ void strbldInit(strbld_t *b, char *text, int capacity, int maxSize)
{
	b->text = b->base = text; //: zText
	b->tag = nullptr; //: db
	b->index = 0; //: nChar
	b->size = capacity; //: nAlloc
	b->maxSize = maxSize; //: mxAlloc
	b->allocType = 1; //: useMalloc
	b->overflowed = false; //: tooBig
	b->allocFailed = false; //: mallocFailed
}

static __constant__ const char _spaces[] = "                             ";
__device__ void strbldAppendSpace(strbld_t *b, int length)
{
	while (length >= (int)sizeof(_spaces)-1) {
		strbldAppend(b, _spaces, sizeof(_spaces)-1);
		length -= sizeof(_spaces)-1;
	}
	if (length > 0)
		strbldAppend(b, _spaces, length);
}

static __constant__ const char _ord[] = "thstndrd";
__device__ void strbldAppendFormat(strbld_t *b, bool useExtended, const char *fmt, va_list args)
{
	char buf[BUFSIZE]; // Conversion buffer
	char *bufpt = nullptr; // Pointer to the conversion buffer
	int c; // Next character in the format string
	bool flag_leftjustify = false; // True if "-" flag is present
	int width = 0; // Width of the current field
	int length = 0; // Length of the field
	for (; (c = *fmt); ++fmt) {
		if (c != '%') {
			bufpt = (char *)fmt;
			int amt = 1;
			while ((c = *++fmt) != '%' && c) amt++;
			strbldAppend(b, bufpt, amt);
			if (!c) break;
		}
		if (!(c = (*++fmt))) {
			strbldAppend(b, "%", 1);
			break;
		}
		// Find out what flags are present
		flag_leftjustify = false; // True if "-" flag is present
		bool flag_plussign = false; // True if "+" flag is present
		bool flag_blanksign = false; // True if " " flag is present
		bool flag_alternateform = false; // True if "#" flag is present
		bool flag_altform2 = false; // True if "!" flag is present
		bool flag_zeropad = false; // True if field width constant starts with zero
		bool done = false; // Loop termination flag
		do {
			switch (c) {
			case '-': flag_leftjustify = true; break;
			case '+': flag_plussign = true; break;
			case ' ': flag_blanksign = true; break;
			case '#': flag_alternateform = true; break;
			case '!': flag_altform2 = true; break;
			case '0': flag_zeropad = true; break;
			default: done = true; break;
			}
		} while (!done && (c = *++fmt));
		// Get the field width
		width = 0; // Width of the current field
		if (c == '*') {
			width = va_arg(args, int);
			if (width < 0) {
				flag_leftjustify = true;
				width = -width;
			}
			c = *++fmt;
		}
		else while (c >= '0' && c <= '9') {
			width = width*10 + c - '0';
			c = *++fmt;
		}
		// Get the precision
		int precision; // Precision of the current field
		if (c == '.') {
			precision = 0;
			c = *++fmt;
			if (c == '*') {
				precision = va_arg(args, int);
				if (precision < 0) precision = -precision;
				c = *++fmt;
			}
			else while (c >= '0' && c <= '9') {
				precision = precision*10 + c - '0';
				c = *++fmt;
			}
		}
		else
			precision = -1;
		// Get the conversion type modifier
		bool flag_long; // True if "l" flag is present
		bool flag_longlong; // True if the "ll" flag is present
		if (c == 'l') {
			flag_long = true;
			c = *++fmt;
			if (c == 'l') {
				flag_longlong = true;
				c = *++fmt;
			}
			else
				flag_longlong = false;
		}
		else
			flag_long = flag_longlong = false;
		// Fetch the info entry for the field
		const info_t *info = &_info[0]; // Pointer to the appropriate info structure
		TYPE type = TYPE_INVALID; // Conversion paradigm
		int i;
		for (i = 0; i < _LENGTHOF(_info); i++) {
			if (c == _info[i].fmtType) {
				info = &_info[i];
				if (useExtended || (info->flags & FLAG_INTERN) == 0) type = info->type;
				else return;
				break;
			}
		}

		char prefix; // Prefix character.  "+" or "-" or " " or '\0'.
		unsigned long long longvalue; // Value for integer types
		double64 realvalue; // Value for real types
#ifndef OMIT_FLOATING_POINT
		int exp, e2; // exponent of real numbers
		int nsd; // Number of significant digits returned
		double rounder; // Used for rounding floating point values
		bool flag_dp; // True if decimal point should be shown
		bool flag_rtz; // True if trailing zeros should be removed
#endif

		// At this point, variables are initialized as follows:
		//   flag_alternateform          TRUE if a '#' is present.
		//   flag_altform2               TRUE if a '!' is present.
		//   flag_plussign               TRUE if a '+' is present.
		//   flag_leftjustify            TRUE if a '-' is present or if the field width was negative.
		//   flag_zeropad                TRUE if the width began with 0.
		//   flag_long                   TRUE if the letter 'l' (ell) prefixed the conversion character.
		//   flag_longlong               TRUE if the letter 'll' (ell ell) prefixed the conversion character.
		//   flag_blanksign              TRUE if a ' ' is present.
		//   width                       The specified field width.  This is always non-negative.  Zero is the default.
		//   precision                   The specified precision.  The default is -1.
		//   type                        The class of the conversion.
		//   info                        Pointer to the appropriate info struct.
		char *extra = nullptr; // Malloced memory used by some conversion
		char *out_; // Rendering buffer
		int outLength; // Size of the rendering buffer
		switch (type) {
		case TYPE_POINTER:
			flag_longlong = (sizeof(char *) == sizeof(long long));
			flag_long = (sizeof(char *) == sizeof(long int));
			// Fall through into the next case
		case TYPE_ORDINAL:
		case TYPE_RADIX:
			if (info->flags & FLAG_SIGNED) {
				long long v;
				if (flag_longlong) v = va_arg(args, long long);
				else if (flag_long) v = va_arg(args, long int);
				else v = va_arg(args, int);
				if (v < 0) {
					longvalue = (v == LLONG_MIN ? ((unsigned long long)1)<<63 : -v);
					prefix = '-';
				}
				else {
					longvalue = v;
					if (flag_plussign) prefix = '+';
					else if (flag_blanksign) prefix = ' ';
					else prefix = '\0';
				}
			}
			else {
				if (flag_longlong) longvalue = va_arg(args, unsigned long long);
				else if (flag_long) longvalue = va_arg(args, unsigned long int);
				else longvalue = va_arg(args, unsigned int);
				prefix = 0;
			}
			if (longvalue == 0) flag_alternateform = false;
			if (flag_zeropad && precision < width - (prefix != '\0'))
				precision = width-(prefix!=0);
			if (precision < BUFSIZE-10) {
				outLength = BUFSIZE;
				out_ = buf;
			}
			else {
				outLength = precision + 10;
				out_ = extra = (char *)malloc(outLength);
				if (!out_) {
					b->allocFailed = true;
					return;
				}
			}
			bufpt = &out_[outLength-1];
			if (type == TYPE_ORDINAL) {
				int x = (int)(longvalue % 10);
				if (x >= 4 || (longvalue/10)%10 == 1) x = 0;
				*--bufpt = _ord[x*2+1];
				*--bufpt = _ord[x*2];
			}
			{
				register const char *cset = &_digits[info->charset]; // Use registers for speed
				register int base = info->base;
				do { // Convert to ascii
					*--bufpt = cset[longvalue % base];
					longvalue = longvalue / base;
				} while (longvalue > 0);
			}
			length = (int)(&out_[outLength-1]-bufpt);
			for (i = precision - length; i > 0; i--) *--bufpt = '0'; // Zero pad
			if (prefix) *--bufpt = prefix; // Add sign
			if (flag_alternateform && info->prefix) { // Add "0" or "0x"
				char x;
				const char *pre = &_prefix[info->prefix];
				for (; (x = *pre); pre++) *--bufpt = x;
			}
			length = (int)(&out_[outLength-1]-bufpt);
			break;
		case TYPE_FLOAT:
		case TYPE_EXP:
		case TYPE_GENERIC:
			realvalue = va_arg(args, double);
#ifdef OMIT_FLOATING_POINT
			length = 0;
#else
			if (precision < 0) precision = 6; // Set default precision
			if (realvalue < 0.0) {
				realvalue = -realvalue;
				prefix = '-';
			}
			else {
				if (flag_plussign) prefix = '+';
				else if (flag_blanksign) prefix = ' ';
				else prefix = 0;
			}
			if (type == TYPE_GENERIC && precision > 0) precision--;
#if 0
			// Rounding works like BSD when the constant 0.4999 is used.  Wierd!
			for (i = precision, rounder = 0.4999; i > 0; i--, rounder *= 0.1);
#else
			// It makes more sense to use 0.5
			for (i = precision, rounder = 0.5; i > 0; i--, rounder *= 0.1) { }
#endif
			if (type == TYPE_FLOAT) realvalue += rounder;
			// Normalize realvalue to within 10.0 > realvalue >= 1.0
			exp = 0;
			if (isnan((double)realvalue)) {
				bufpt = "NaN";
				length = 3;
				break;
			}
			if (realvalue > 0.0) {
				double64 scale = 1.0;
				while (realvalue >= 1e100*scale && exp <= 350) { scale *= 1e100; exp += 100; }
				while (realvalue >= 1e64*scale && exp <= 350) { scale *= 1e64; exp += 64; }
				while (realvalue >= 1e8*scale && exp <= 350) { scale *= 1e8; exp += 8; }
				while (realvalue >= 10.0*scale && exp <= 350) { scale *= 10.0; exp++; }
				realvalue /= scale;
				while (realvalue < 1e-8) { realvalue *= 1e8; exp -= 8; }
				while (realvalue < 1.0) { realvalue *= 10.0; exp--; }
				if (exp > 350) {
					if (prefix == '-') bufpt = "-Inf";
					else if (prefix == '+') bufpt = "+Inf";
					else bufpt = "Inf";
					length = strlen(bufpt);
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
			else
				flag_rtz = flag_altform2;
			e2 = (type == TYPE_EXP ? 0 : exp);
			if (e2+precision+width > BUFSIZE - 15) {
				bufpt = extra = (char *)malloc(e2+precision+width+15);
				if (!bufpt) {
					b->allocFailed = true;
					return;
				}
			}
			out_ = bufpt;
			nsd = 16 + flag_altform2*10;
			flag_dp = (precision > 0) | flag_alternateform | flag_altform2;
			// The sign in front of the number
			if (prefix) *bufpt++ = prefix;
			// Digits prior to the decimal point
			if (e2 < 0) *bufpt++ = '0';
			else for (; e2 >= 0; e2--) *bufpt++ = GetDigit(&realvalue, &nsd);
			// The decimal point
			if (flag_dp) *(bufpt++) = '.';
			// "0" digits after the decimal point but before the first significant digit of the number
			for (e2++; e2 < 0; precision--, e2++) { assert(precision > 0); *bufpt++ = '0'; }
			// Significant digits after the decimal point
			while (precision-- > 0) *bufpt++ = GetDigit(&realvalue, &nsd);
			// Remove trailing zeros and the "." if no digits follow the "."
			if (flag_rtz && flag_dp) {
				while (bufpt[-1] == '0') *--bufpt = 0;
				assert(bufpt > out_);
				if (bufpt[-1] == '.') {
					if (flag_altform2) *bufpt++ = '0';
					else *(--bufpt) = 0;
				}
			}
			// Add the "eNNN" suffix
			if (type == TYPE_EXP) {
				*bufpt++ = _digits[info->charset];
				if (exp < 0) { *bufpt++ = '-'; exp = -exp; }
				else *bufpt++ = '+';
				if (exp >= 100) { *bufpt++ = (char)(exp/100+'0'); exp %= 100; } // 100's digit
				*bufpt++ = (char)(exp/10+'0'); // 10's digit
				*bufpt++ = (char)(exp%10+'0'); // 1's digit
			}
			*bufpt = 0;

			// The converted number is in buf[] and zero terminated. Output it. Note that the number is in the usual order, not reversed as with integer conversions.
			length = (int)(bufpt-out_);
			bufpt = out_;

			// Special case:  Add leading zeros if the flag_zeropad flag is set and we are not left justified
			if (flag_zeropad && !flag_leftjustify && length < width) {
				int pad = width - length;
				for (i = width; i >= pad; i--) bufpt[i] = bufpt[i-pad];
				i = (prefix != '\0');
				while (pad--) bufpt[i++] = '0';
				length = width;
			}
#endif
			break;
		case TYPE_SIZE:
			*(va_arg(args, int*)) = (int)b->size;
			length = width = 0;
			break;
		case TYPE_PERCENT:
			buf[0] = '%';
			bufpt = buf;
			length = 1;
			break;
		case TYPE_CHARX:
			c = va_arg(args, int);
			buf[0] = (char)c;
			if (precision >= 0) {
				for (i = 1; i < precision; i++) buf[i] = (char)c;
				length = precision;
			}
			else length =1;
			bufpt = buf;
			break;
		case TYPE_STRING:
		case TYPE_DYNSTRING:
			bufpt = va_arg(args, char*);
			if (!bufpt) bufpt = "";
			else if (type == TYPE_DYNSTRING) extra = bufpt;
			if (precision >= 0) for (length = 0; length < precision && bufpt[length]; length++) { }
			else length = strlen(bufpt);
			break;
		case TYPE_SQLESCAPE:
		case TYPE_SQLESCAPE2:
		case TYPE_SQLESCAPE3: {
			char q = (type == TYPE_SQLESCAPE3 ? '"' : '\''); // Quote character
			char *escarg = va_arg(args, char*);
			bool isnull = (!escarg);
			if (isnull) escarg = (type == TYPE_SQLESCAPE2 ? "NULL" : "(NULL)");
			int k = precision;
			int j, n;
			char ch;
			for (i = n = 0; k != 0 && (ch = escarg[i]) != 0; i++, k--)
				if (ch == q) n++;
			bool needQuote = (!isnull && type == TYPE_SQLESCAPE2);
			n += i + 1 + needQuote*2;
			if (n > BUFSIZE) {
				bufpt = extra = (char *)malloc(n);
				if (!bufpt) {
					b->allocFailed = true;
					return;
				}
			}
			else
				bufpt = buf;
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
			//TagBase_RuntimeStatics.AppendFormat[0](this, args);
			length = width = 0;
			break; }
		case TYPE_SRCLIST: {
			//TagBase_RuntimeStatics.AppendFormat[1](this, args);
			length = width = 0;
			break; }
		default: {
			assert(type == TYPE_INVALID);
			return; }
		}
		// The text of the conversion is pointed to by "bufpt" and is "length" characters long.  The field width is "width".  Do the output.
		if (!flag_leftjustify) {
			register int nspace = width-length;
			if (nspace > 0) strbldAppendSpace(b, nspace);
		}
		if (length > 0) strbldAppend(b, bufpt, length);
		if (flag_leftjustify) {
			register int nspace = width-length;
			if (nspace > 0) strbldAppendSpace(b, nspace);
		}
		if (extra) free(extra);
	}
}

__device__ void strbldAppend(strbld_t *b, const char *str, int length)
{
	assert(str != nullptr || length == 0);
	if (b->overflowed | b->allocFailed) {
		ASSERTCOVERAGE(b->overflowed);
		ASSERTCOVERAGE(b->allocFailed);
		return;
	}
	assert(b->text != nullptr || b->index == 0);
	if (length < 0)
		length = strlen(str);
	if (length == 0 || _NEVER(str == nullptr))
		return;
	if (b->index + length >= b->size) {
		char *newText;
		if (!b->allocType) {
			b->overflowed = true;
			length = (int)(b->size - b->index - 1);
			if (length <= 0)
				return;
		}
		else {
			char *oldText = (b->text == b->base ? nullptr : b->text);
			long long newSize = b->index;
			newSize += length + 1;
			if (newSize > b->maxSize) {
				strbldReset(b);
				b->overflowed = true;
				return;
			}
			else
				b->size = (int)newSize;
			newText = (char *)(b->allocType == 1 ? tagrealloc(b->tag, oldText, b->size) : realloc(oldText, b->size));
			if (newText) {
				if (!oldText && b->index > 0) memcpy(newText, b->text, b->index);
				b->text = newText;
			}
			else {
				b->allocFailed = true;
				strbldReset(b);
				return;
			}
		}
	}
	assert(b->text != nullptr);
	memcpy(&b->text[b->index], str, length);
	b->index += length;
}

__device__ char *strbldToString(strbld_t *b)
{
	if (b->text) {
		b->text[b->index] = 0;
		if (b->allocType && b->text == b->base) {
			b->text = (char *)(b->allocType == 1 ? tagalloc(b->tag, b->index + 1) : malloc(b->index + 1));
			if (b->text) memcpy(b->text, b->base, b->index + 1);
			else b->allocFailed = true;
		}
	}
	return b->text;
}

__device__ void strbldReset(strbld_t *b)
{
	if (b->text != b->base) {
		if (b->allocType == 1)
			tagfree(b->tag, b->text);
		else
			free(b->text);
	}
	b->text = nullptr;
}

#pragma endregion

__END_DECLS;