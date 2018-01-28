#include <ctypecu.h> //: util.c
#include <stringcu.h>
#include <ext/global.h>
#include <ext/convert.h>
#include <assert.h>

#pragma region ATOX

/* The string z[] is an text representation of a real number. Convert this string to a double and write it into *r.
**
** The string z[] is length bytes in length (bytes, not characters) and uses the encoding enc.  The string is not necessarily zero-terminated.
**
** Return TRUE if the result is a valid real number (or integer) and FALSE if the string is empty or contains extraneous text.  Valid numbers
** are in one of these formats:
**
**    [+-]digits[E[+-]digits]
**    [+-]digits.[digits][E[+-]digits]
**    [+-].digits[E[+-]digits]
**
** Leading and trailing whitespace is ignored for the purpose of determining validity.
**
** If some prefix of the input string is a valid number, this routine returns FALSE but it still converts the prefix and writes the result into *r.
*/
__device__ bool convert_atofe(const char *z, double *r, int length, TEXTENCODE encode) //: sqlite3AtoF
{
#ifndef OMIT_FLOATING_POINT
	assert(encode == TEXTENCODE_UTF8 || encode == TEXTENCODE_UTF16LE || encode == TEXTENCODE_UTF16BE);
	*r = 0.0; // Default return value, in case of an error
	const char *end = z + length; int incr; bool nonNum = false;
	if (encode == TEXTENCODE_UTF8)
		incr = 1;
	else {
		incr = 2;
		assert(TEXTENCODE_UTF16LE == 2 && TEXTENCODE_UTF16BE == 3);
		int i; for (i = 3 - encode; i < length && !z[i]; i += 2) { }
		nonNum = i < length;
		end = &z[i ^ 1];
		z += encode & 1;
	}

	// skip leading spaces
	while (z < end && isspace(*z)) z += incr;
	if (z >= end) return false;

	// get sign of significand
	int sign = 1; // sign of significand
	if (*z == '-') { sign = -1; z += incr; }
	else if (*z == '+') z += incr;

	// copy max significant digits to significand
	int64_t s = 0; // significand
	int digits = 0;
	while (z < end && isdigit(*z) && s < ((INT64_MAX - 9) / 10)) { s = s*10 + (*z - '0'); z += incr, digits++; }

	// skip non-significant significand digits (increase exponent by d to shift decimal left)
	int d = 0; // adjust exponent for shifting decimal point
	while (z < end && isdigit(*z)) { z += incr, digits++, d++; }
	if (z >= end) goto do_atof_calc;

	// if decimal point is present
	if (*z == '.') {
		z += incr;
		// copy digits from after decimal to significand (decrease exponent by d to shift decimal right)
		while (z < end && isdigit(*z)) {
			if (s < (INT64_MAX - 9) / 10) { s = s*10 + (*z - '0'); d--; }
			z += incr, digits++;
		}
	}
	if (z >= end) goto do_atof_calc;

	// if exponent is present
	int e = 0; // exponent
	int esign = 1; // sign of exponent
	bool evalid = true;  // True exponent is either not used or is well-formed
	if (*z == 'e' || *z == 'E') {
		z += incr;
		evalid = false;
		// This branch is needed to avoid a (harmless) buffer overread.  The special comment alerts the mutation tester that the correct answer
		// is obtained even if the branch is omitted
		if (z >= end) goto do_atof_calc; /* PREVENTS-HARMLESS-OVERREAD */

		// get sign of exponent
		if (*z == '-') { esign = -1; z += incr; }
		else if (*z == '+') z += incr;

		// copy digits to exponent
		while (z < end && isdigit(*z)) { e = e < 10000 ? e*10 + (*z - '0') : 10000; z += incr; evalid = true; }
	}

	// skip trailing spaces
	while (z < end && isspace(*z)) z += incr;

do_atof_calc:
	// adjust exponent by d, and update sign
	e = (e * esign) + d;
	if (e < 0) { esign = -1; e *= -1; }
	else esign = 1;

	double result;
	if (!s) result = sign < 0 ? -(double)0 : (double)0; // In the IEEE 754 standard, zero is signed.
	else {
		// Attempt to reduce exponent.
		//
		// Branches that are not required for the correct answer but which only help to obtain the correct answer faster are marked with special
		// comments, as a hint to the mutation tester.
		while (e > 0) {
			if (esign > 0) { if (s >= (INT64_MAX / 10)) break; s *= 10; }
			else { if (s % 10) break; s /= 10; }
			e--;
		}

		// adjust the sign of significand
		s = sign < 0 ? -s : s;

		if (!e) result = (double)s;
		else {
			long_double scale = 1.0;
			// attempt to handle extremely small/large numbers better
			if (e > 307) {
				if (e < 342) {
					while (e % 308) { scale *= 1.0e+1; e -= 1; }
					if (esign < 0) { result = s/scale; result /= 1.0e+308; }
					else { result = s*scale; result *= 1.0e+308; }
				}
				else {
					assert(e >= 342);
					result = esign < 0 ? 0.0*s :
#ifdef INFINITY
						INFINITY*s;
#else
						1e308*1e308*s; // Infinity
#endif
				}
			}
			else {
				// 1.0e+22 is the largest power of 10 than can be represented exactly.
				while (e % 22) { scale *= 1.0e+1; e -= 1; }
				while (e > 0) { scale *= 1.0e+22; e -= 22; }
				result = esign < 0 ? s/scale : s*scale;
			}
		}
	}

	// store the result
	*r = result;

	// return true if number and no extra non-whitespace chracters after
	return z == end && digits > 0 && evalid && !nonNum;
#else
	return !convert_atoi64e(z, r, length, encode);
#endif /* OMIT_FLOATING_POINT */
}

/* Compare the 19-character string z against the text representation value 2^63:  9223372036854775808.  Return negative, zero, or positive
** if z is less than, equal to, or greater than the string. Note that z must contain exactly 19 characters.
**
** Unlike memcmp() this routine is guaranteed to return the difference in the values of the last digit if the only difference is in the
** last digit.  So, for example,
**
**      compare2pow63("9223372036854775800", 1)
**
** will return -8.
*/
static __device__ int compare2pow63(const char *z, int incr)
{
	const char *pow63 = "922337203685477580"; // 012345678901234567
	int c = 0;
	for (int i = 0; !c && i < 18; i++) c = (z[i * incr] - pow63[i])*10;
	if (!c) {
		c = z[18 * incr] - '8';
		ASSERTCOVERAGE(c == -1);
		ASSERTCOVERAGE(c == 0);
		ASSERTCOVERAGE(c == +1);
	}
	return c;
}

/* Convert z to a 64-bit signed integer.  z must be decimal. This routine does *not* accept hexadecimal notation.
**
** If the z value is representable as a 64-bit twos-complement integer, then write that value into *pNum and return 0.
**
** If z is exactly 9223372036854775808, return 2.  This special case is broken out because while 9223372036854775808 cannot be a 
** signed 64-bit integer, its negative -9223372036854775808 can be.
**
** If z is too big for a 64-bit integer and is not 9223372036854775808  or if z contains any non-numeric text,
** then return 1.
**
** length is the number of bytes in the string (bytes, not characters). The string is not necessarily zero-terminated.  The encoding is given by enc.
*/
__device__ int convert_atoi64e(const char *z, int64_t *r, int length, TEXTENCODE encode) //: sqlite3Atoi64
{
	const char *start;
	const char *end = z + length;
	assert(encode == TEXTENCODE_UTF8 || encode == TEXTENCODE_UTF16LE || encode == TEXTENCODE_UTF16BE);
	int incr;
	bool nonNum = false;
	if (encode == TEXTENCODE_UTF8)
		incr = 1;
	else {
		incr = 2;
		assert(TEXTENCODE_UTF16LE == 2 && TEXTENCODE_UTF16BE == 3);
		int i; for (i = 3 - encode; i < length && !z[i]; i += 2) { }
		nonNum = i < length;
		end = &z[i ^ 1];
		z += encode & 1;
	}

	// skip leading spaces
	while (z < end && isspace(*z)) z += incr;

	// get sign of significand
	int neg = 0; // assume positive
	if (z < end) {
		if (*z == '-') { neg = 1; z += incr; }
		else if (*z == '+') z += incr;
	}
	start = z;

	// skip leading zeros
	while (z < end && z[0] == '0') z += incr;

	// convert
	uint64_t u = 0; int c = 0; int i; for (i = 0; &z[i] < end && (c = z[i]) >= '0' && c <= '9'; i += incr) { u = u*10 + c - '0'; }
	if (u > INT64_MAX) *r = neg ? INT64_MIN : INT64_MAX;
	else *r = neg ? -(int64_t)u : (int64_t)u;
	ASSERTCOVERAGE(i == 18);
	ASSERTCOVERAGE(i == 19);
	ASSERTCOVERAGE(i == 20);

	int rc = &z[i] < end || (!i && start == z) || nonNum ? 1 : 0;
	// z is empty or contains non-numeric text or is longer than 19 digits (thus guaranteeing that it is too large)
	if (i > 19 * incr) return 2; // Too many digits
	// Less than 19 digits, so we know that it fits in 64 bits
	else if (i < 19 * incr) { assert(u <= INT64_MAX); return rc; } 
	else {
		// z is a 19-digit numbers.  Compare it against 9223372036854775808.
		c = compare2pow63(z, incr);
		// z is less than 9223372036854775808 so it fits
		if (c < 0) { assert(u <= INT64_MAX); return rc; }
		// z is greater than 9223372036854775808 so it overflows
		else if (c > 0) return 2;
		// z is exactly 9223372036854775808. Fits if negative. The special case 2 overflow if positive
		else { assert(u - 1 == INT64_MAX); return neg ? rc : 3; }
	}
}

/* Transform a UTF-8 integer literal, in either decimal or hexadecimal, into a 64-bit signed integer.  This routine accepts hexadecimal literals, whereas convert_atoi64e() does not.
**
** Returns:
**
**     0    Successful transformation.  Fits in a 64-bit signed integer.
**     1    Integer too large for a 64-bit signed integer or is malformed
**     2    Special case of 9223372036854775808
*/
__device__ int convert_axtoi64e(const char *z, int64_t *r) { //: sqlite3DecOrHexToI64
#ifndef LIBCU_OMIT_HEX_INTEGER
	if (z[0] == '0' && (z[1] == 'x' || z[1] == 'X')) {
		int i; for (i = 2; z[i] == '0'; i++) { }
		uint64_t u = 0; int k; for (k = i; isxdigit(z[k]); k++) { u = u*16 + convert_xtoi(z[k]); }
		memcpy(r, &u, 8);
		return !z[k] && k - i <= 16 ? 0 : 2;
	} else
#endif /* LIBCU_OMIT_HEX_INTEGER */
	{ return convert_atoi64e(z, r, strlen(z), TEXTENCODE_UTF8); }
}

/* If z represents an integer that will fit in 32-bits, then set *r to that integer and return true.  Otherwise return false.
**
** This routine accepts both decimal and hexadecimal notation for integers.
**
** Any non-numeric characters that following z are ignored. This is different from convert_atoi64e() which requires the
** input number to be zero-terminated.
*/
__device__ bool convert_atoie(const char *z, int *r) //: sqlite3GetInt32
{
	int i, c;
	// get sign of significand
	int neg = 0;
	if (z[0] == '-') { neg = 1; z++; }
	else if (z[0] == '+') z++;
#ifndef LIBCU_OMIT_HEX_INTEGER
	else if (z[0] == '0' && (z[1] == 'x' || z[1] == 'X') && isxdigit(z[2])) {
		z += 2; while (z[0] == '0') z++;
		uint32_t u = 0; for (i = 0; isxdigit(z[i]) && i < 8; i++) { u = u*16 + convert_xtoi(z[i]); }
		if (!(u & 0x80000000) && !isxdigit(z[i])) { memcpy(r, &u, 4); return true; }
		else return false;
	}
#endif /* LIBCU_OMIT_HEX_INTEGER */
	if (!isdigit(z[0])) return false;
	// skip leading zeros
	while (z[0] == '0') z++;
	int64_t v = 0; for (i = 0; i < 11 && (c = z[i] - '0') >= 0 && c <= 9; i++) { v = v*10 + c; }
	// The longest decimal representation of a 32 bit integer is 10 digits:
	//             1234567890
	//     2^31 -> 2147483648
	ASSERTCOVERAGE(i == 10);
	if (i > 10) return false;
	ASSERTCOVERAGE(v - neg == 2147483647);
	if (v - neg > 2147483647) return false;
	*r = (int)(neg ? -v : v);
	return true;
}

/* Return a 32-bit integer value extracted from a string.  If the string is not an integer, just return 0. */
// inline: convert_atoi(z)
__device__ int convert_atoi(const char *z) { int r = 0; if (z) convert_atoie(z, &r); return r; } //: sqlite3Atoi

/* Translate a single byte of Hex into an integer. This routine only works if h really is a valid hexadecimal character:  0..9a..fA..F */
__device__ uint8_t convert_xtoi(int h) { //: sqlite3HexToInt
	assert((h >= '0' && h <= '9') || (h >= 'a' && h <= 'f') || (h >= 'A' && h <= 'F'));
#ifdef LIBCU_ASCII
	h += 9 * (1 & (h >> 6));
#endif
#ifdef LIBCU_EBCDIC
	h += 9 * (1 & ~(h >> 4));
#endif
	return (uint8_t)(h & 0xf);
}

#if !defined(OMIT_BLOB_LITERAL) || defined(LIBCU_HAS_CODEC)
/* Convert a BLOB literal of the form "x'hhhhhh'" into its binary value.  Return a pointer to its binary value.  Space to hold the
** binary value has been obtained from malloc and must be freed by the calling routine.
*/
__device__ void *convert_taghextoblob(tagbase_t *tag, const char *z, int size) // (size_t size) //: sqlite3HexToBlob
{
	char *b = (char *)tagallocRawNN(tag, size / 2 + 1);
	size--;
	if (b) {
		int i; for (i = 0; i < size; i += 2)
			b[i/2] = (convert_xtoi(z[i]) << 4) | convert_xtoi(z[i + 1]);
		b[i/2] = 0;
	}
	return b;
}
#endif /* !OMIT_BLOB_LITERAL || LIBCU_HAS_CODEC */

//static __constant__ char const __convert_digits[] = "0123456789";
//__device__ char *convert_itoa64(int64_t i, char *b) //: sky
//{
//	char *p = b;
//	if (i < 0) { *p++ = '-'; i *= -1; }
//	int64_t shifter = i;
//	do { ++p; shifter = shifter/10; } while(shifter); // Move to where representation ends
//	*p = '\0';
//	do { *--p = __convert_digits[i%10]; i = i/10; } while(i); // Move back, inserting digits as u go
//	return b;
//}

#pragma endregion

#pragma region VARINT

/* The variable-length integer encoding is as follows:
**
** KEY:
**         A = 0xxxxxxx    7 bits of data and one flag bit
**         B = 1xxxxxxx    7 bits of data and one flag bit
**         C = xxxxxxxx    8 bits of data
**
**  7 bits - A
** 14 bits - BA
** 21 bits - BBA
** 28 bits - BBBA
** 35 bits - BBBBA
** 42 bits - BBBBBA
** 49 bits - BBBBBBA
** 56 bits - BBBBBBBA
** 64 bits - BBBBBBBBC
*/

/* Write a 64-bit variable-length integer to memory starting at p[0]. The length of data write will be between 1 and 9 bytes.  The number
** of bytes written is returned.
**
** A variable-length integer consists of the lower 7 bits of each byte for all bytes that have the 8th bit set and one byte with the 8th
** bit clear.  Except, if we get to the 9th byte, it stores the full 8 bits and is the last byte.
*/
__device__ int convert_putvarint(unsigned char *p, uint64_t v) //: sqlite3PutVarint
{
	if (v <= 0x7f) {
		p[0] = v & 0x7f;
		return 1;
	}
	if (v <= 0x3fff) {
		p[0] = ((v >> 7) & 0x7f) | 0x80;
		p[1] = v & 0x7f;
		return 2;
	}
	//: from putVarint64
	int i, j, n;
	if (v & (((uint64_t)0xff000000) << 32)) {
		p[8] = (uint8_t)v;
		v >>= 8;
		for (i = 7; i >= 0; i--) {
			p[i] = (uint8_t)((v & 0x7f) | 0x80);
			v >>= 7;
		}
		return 9;
	}    
	n = 0;
	uint8_t b[10];
	do {
		b[n++] = (uint8_t)((v & 0x7f) | 0x80);
		v >>= 7;
	} while (v != 0);
	b[0] &= 0x7f;
	assert(n <= 9);
	for (i = 0, j = n - 1; j >= 0; j--, i++)
		p[i] = b[j];
	return n;
}

/* Bitmasks used by convert_getvarint().  These precomputed constants are defined here rather than simply putting the constant expressions
** inline in order to work around bugs in the RVT compiler.
**
** SLOT_2_0     A mask for  (0x7f<<14) | 0x7f
**
** SLOT_4_2_0   A mask for  (0x7f<<28) | SLOT_2_0
*/
#define SLOT_2_0     0x001fc07f
#define SLOT_4_2_0   0xf01fc07f

/* Read a 64-bit variable-length integer from memory starting at p[0]. Return the number of bytes read.  The value is stored in *v. */
__device__ uint8_t convert_getvarint(const unsigned char *p, uint64_t *v) //: sqlite3GetVarint
{
	uint32_t a, b, s;
	a = *p;
	// a: p0 (unmasked)
	if (!(a & 0x80)) {
		*v = a;
		return 1;
	}

	p++;
	b = *p;
	// b: p1 (unmasked)
	if (!(b & 0x80)) {
		a &= 0x7f;
		a = a << 7;
		a |= b;
		*v = a;
		return 2;
	}

	// Verify that constants are precomputed correctly
	assert(SLOT_2_0 == ((0x7f << 14) | 0x7f));
	assert(SLOT_4_2_0 == ((0xfU << 28) | (0x7f << 14) | 0x7f));

	p++;
	a = a << 14;
	a |= *p;
	// a: p0 << 14 | p2 (unmasked)
	if (!(a & 0x80)) {
		a &= SLOT_2_0;
		b &= 0x7f;
		b = b << 7;
		a |= b;
		*v = a;
		return 3;
	}

	// CSE1 from below
	a &= SLOT_2_0;
	p++;
	b = b << 14;
	b |= *p;
	// b: p1 << 14 | p3 (unmasked)
	if (!(b & 0x80)) {
		b &= SLOT_2_0;
		// moved CSE1 up
		// a &= (0x7f << 14) | 0x7f;
		a = a << 7;
		a |= b;
		*v = a;
		return 4;
	}
	
	// a: p0 << 14 | p2 (masked)
	// b: p1 << 14 | p3 (unmasked)
	// 1: save off p0 << 21 | p1 << 14 | p2 << 7 | p3 (masked)
	// moved CSE1 up
	// a &= (0x7f << 14) | 0x7f;
	b &= SLOT_2_0;
	s = a;
	// s: p0 << 14 | p2 (masked)

	p++;
	a = a << 14;
	a |= *p;
	// a: p0 << 28 | p2 << 14 | p4 (unmasked)
	if (!(a & 0x80)) {
		// we can skip these cause they were (effectively) done above in calc'ing s
		// a &= (0x7f << 28) | (0x7f << 14) | 0x7f;
		// b &= (0x7f << 14) | 0x7f;
		b = b << 7;
		a |= b;
		s = s >> 18;
		*v = ((uint64_t)s) << 32 | a;
		return 5;
	}

	// 2: save off p0 << 21 | p1 << 14 | p2 << 7 | p3 (masked)
	s = s << 7;
	s |= b;
	// s: p0 << 21 | p1 << 14 | p2 << 7 | p3 (masked)

	p++;
	b = b << 14;
	b |= *p;
	/* b: p1 << 28 | p3 << 14 | p5 (unmasked) */
	if (!(b & 0x80)) {
		// we can skip this cause it was (effectively) done above in calc'ing s
		// b &= (0x7f << 28)|(0x7f << 14) | 0x7f;
		a &= SLOT_2_0;
		a = a << 7;
		a |= b;
		s = s >> 18;
		*v = ((uint64_t)s) << 32 | a;
		return 6;
	}

	p++;
	a = a << 14;
	a |= *p;
	// a: p2 << 28 | p4 << 14 | p6 (unmasked)
	if (!(a & 0x80)) {
		a &= SLOT_4_2_0;
		b &= SLOT_2_0;
		b = b << 7;
		a |= b;
		s = s >> 11;
		*v = ((uint64_t)s) << 32 | a;
		return 7;
	}

	// CSE2 from below
	a &= SLOT_2_0;
	p++;
	b = b << 14;
	b |= *p;
	// b: p3 << 28 | p5 << 14 | p7 (unmasked)
	if (!(b & 0x80)) {
		b &= SLOT_4_2_0;
		// moved CSE2 up
		// a &= (0x7f << 14) | 0x7f;
		a = a << 7;
		a |= b;
		s = s >> 4;
		*v = ((uint64_t)s) << 32 | a;
		return 8;
	}

	p++;
	a = a << 15;
	a |= *p;
	// a: p4 << 29 | p6 << 15 | p8 (unmasked)

	// moved CSE2 up
	// a &= (0x7f << 29) | (0x7f << 15) | 0xff;
	b &= SLOT_2_0;
	b = b << 8;
	a |= b;

	s = s << 4;
	b = p[-4];
	b &= 0x7f;
	b = b >> 3;
	s |= b;

	*v = ((uint64_t)s) << 32 | a;
	return 9;
}

/* Read a 32-bit variable-length integer from memory starting at p[0]. Return the number of bytes read.  The value is stored in *v.
**
** If the varint stored in p[0] is larger than can fit in a 32-bit unsigned integer, then set *v to 0xffffffff.
**
** A MACRO version, getVarint32, is provided which inlines the single-byte case.  All code should use the MACRO version as 
** this function assumes the single-byte case has already been handled.
*/
__device__ uint8_t convert_getvarint32(const unsigned char *p, uint32_t *v) //: sqlite3GetVarint32
{
	uint32_t a, b;
	// The 1-byte case.  Overwhelmingly the most common.  Handled inline by the getVarin32() macro
	a = *p;
	// a: p0 (unmasked)
#ifndef _getvarint32
	if (!(a & 0x80)) {
		// Values between 0 and 127
		*v = a;
		return 1;
	}
#endif

	// The 2-byte case
	p++;
	b = *p;
	// b: p1 (unmasked)
	if (!(b & 0x80)) {
		// Values between 128 and 16383
		a &= 0x7f;
		a = a << 7;
		*v = a | b;
		return 2;
	}

	// The 3-byte case
	p++;
	a = a << 14;
	a |= *p;
	// a: p0 << 14 | p2 (unmasked)
	if (!(a & 0x80)) {
		// Values between 16384 and 2097151
		a &= (0x7f << 14) | 0x7f;
		b &= 0x7f;
		b = b << 7;
		*v = a | b;
		return 3;
	}

	// A 32-bit varint is used to store size information in btrees. Objects are rarely larger than 2MiB limit of a 3-byte varint.
	// A 3-byte varint is sufficient, for example, to record the size of a 1048569-byte BLOB or string.
	//
	// We only unroll the first 1-, 2-, and 3- byte cases.  The very rare larger cases can be handled by the slower 64-bit varint routine.
#if 1
	{
		p -= 2;
		uint64_t v64; uint8_t n = convert_getvarint(p, &v64);
		assert(n > 3 && n <= 9);
		*v = (v64 & UINT32_MAX) != v64 ? 0xffffffff : (uint32_t)v64;
		return n;
	}
#else
	// For following code (kept for historical record only) shows an unrolling for the 3- and 4-byte varint cases.  This code is
	// slightly faster, but it is also larger and much harder to test.
	p++;
	b = b << 14;
	b |= *p;
	// b: p1 << 14 | p3 (unmasked)
	if (!(b & 0x80)) {
		// Values between 2097152 and 268435455
		b &= (0x7f << 14) | 0x7f;
		a &= (0x7f << 14) | 0x7f;
		a = a << 7;
		*v = a | b;
		return 4;
	}

	p++;
	a = a << 14;
	a |= *p;
	// a: p0 << 28 | p2 << 14 | p4 (unmasked)
	if (!(a & 0x80)) {
		// Values  between 268435456 and 34359738367
		a &= SLOT_4_2_0;
		b &= SLOT_4_2_0;
		b = b << 7;
		*v = a | b;
		return 5;
	}

	// We can only reach this point when reading a corrupt database file.  In that case we are not in any hurry.
	// Use the (relatively slow) general-purpose convert_getvarint() routine to extract the value.
	{
		p -= 4;
		uint64_t v64; uint8_t n = convert_getvarint(p, &v64);
		assert(n > 5 && n <= 9);
		*v = (uint32)v64;
		return n;
	}
#endif
}

__device__ int convert_getvarintLength(uint64_t v) //: sqlite3VarintLen
{
	int i; for (i = 1; v >>= 7; i++) { assert(i < 10); }
	return i;
}

#pragma endregion

/* not sure where these are */ //: Sky
__device__ uint16_t convert_get2nz(const uint8_t *p) { return ((((int)((p[0]<<8) | p[1]) -1)&0xffff)+1); }
__device__ uint16_t convert_get2(const uint8_t *p) { return (p[0]<<8) | p[1]; }
__device__ void convert_put2(unsigned char *p, uint32_t v) { p[0] = (uint8_t)(v>>8); p[1] = (uint8_t)v; }

/* Read or write a four-byte big-endian integer value. */
__device__ uint32_t convert_get4(const uint8_t *p) //: sqlite3Get4byte
{
#if LIBCU_BYTEORDER==4321
  uint32_t x; memcpy(&x, p, 4); return x;
#elif LIBCU_BYTEORDER==1234 && GCC_VERSION>=4003000
  uint32_t x; memcpy(&x, p, 4); return __builtin_bswap32(x);
#elif LIBCU_BYTEORDER==1234 && MSVC_VERSION>=1300
  uint32_t x; memcpy(&x, p, 4); return _byteswap_ulong(x);
#else
  ASSERTCOVERAGE(p[0] & 0x80); return ((unsigned)p[0]<<24) | (p[1]<<16) | (p[2]<<8) | p[3];
#endif
}
//__device__ uint32_t convert_get4(const uint8_t *p) { return (p[0]<<24) | (p[1]<<16) | (p[2]<<8) | p[3]; }
__device__ void convert_put4(unsigned char *p, uint32_t v) //: sqlite3Put4byte
{
#if LIBCU_BYTEORDER==4321
  memcpy(p, &v, 4);
#elif LIBCU_BYTEORDER==1234 && GCC_VERSION>=4003000
  uint32_t x = __builtin_bswap32(v); memcpy(p, &x, 4);
#elif LIBCU_BYTEORDER==1234 && MSVC_VERSION>=1300
  uint32_t x = _byteswap_ulong(v); memcpy(p, &x, 4);
#else
  p[0] = (uint8_t)(v>>24);
  p[1] = (uint8_t)(v>>16);
  p[2] = (uint8_t)(v>>8);
  p[3] = (uint8_t)v;
#endif
}
//__device__ void convert_put4(unsigned char *p, uint32 v) { p[0] = (uint8_t)(v>>24); p[1] = (uint8_t)(v>>16); p[2] = (uint8_t)(v>>8); p[3] = (uint8_t)v; }

#pragma region From: pragma.c

static __constant__ const char __safetyLevelText[] = "onoffalseyestruefull";
static __constant__ const uint8_t __safetyLevelOffset[] = {0, 1, 2, 4, 9, 12, 16};
static __constant__ const uint8_t __safetyLevelLength[] = {2, 2, 3, 5, 3, 4, 4};
static __constant__ const uint8_t __safetyLevelValue[] =  {1, 0, 0, 0, 1, 1, 2};

__device__ uint8_t convert_atolevel(const char *z, int omitFull, uint8_t dflt)
{
	if (isdigit(*z))
		return (uint8_t)convert_atoi(z);
	int n = strlen(z);
	for (int i = 0; i < _LENGTHOF(__safetyLevelLength) - omitFull; i++)
		if (__safetyLevelLength[i] == n && !strncmp(&__safetyLevelText[__safetyLevelOffset[i]], z, n))
			return __safetyLevelValue[i];
	return dflt;
}

__device__ bool convert_atob(const char *z, uint8_t dflt)
{
	return convert_atolevel(z, 1, dflt) != 0;
}

#pragma endregion