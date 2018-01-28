#include <ext/utf.h> //: utf.c
#include <assert.h>

#pragma region UTF Macros

static __constant__ const unsigned char __utf8Trans1[] =
{
	0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
	0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
	0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
	0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f,
	0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
	0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
	0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
	0x00, 0x01, 0x02, 0x03, 0x00, 0x01, 0x00, 0x00,
};

#define WRITE_UTF8(z, c) { \
	if (c < 0x00080) { \
	*z++ = (unsigned char)(c&0xFF); \
	} else if (c < 0x00800) { \
	*z++ = 0xC0 + (unsigned char)((c>>6)&0x1F); \
	*z++ = 0x80 + (unsigned char)(c&0x3F); \
	} else if (c < 0x10000) { \
	*z++ = 0xE0 + (unsigned char)((c>>12)&0x0F); \
	*z++ = 0x80 + (unsigned char)((c>>6)&0x3F); \
	*z++ = 0x80 + (unsigned char)(c&0x3F); \
	} else { \
	*z++ = 0xF0 + (unsigned char)((c>>18)&0x07); \
	*z++ = 0x80 + (unsigned char)((c>>12)&0x3F); \
	*z++ = 0x80 + (unsigned char)((c>>6)&0x3F); \
	*z++ = 0x80 + (unsigned char)(c&0x3F); \
	} \
	}

#define WRITE_UTF16LE(z, c) { \
	if (c <= 0xFFFF) { \
	*z++ = (unsigned char)(c&0x00FF); \
	*z++ = (unsigned char)((c>>8)&0x00FF); \
	} else { \
	*z++ = (unsigned char)(((c>>10)&0x003F) + (((c-0x10000)>>10)&0x00C0)); \
	*z++ = (unsigned char)(0x00D8 + (((c-0x10000)>>18)&0x03)); \
	*z++ = (unsigned char)(c&0x00FF); \
	*z++ = (unsigned char)(0x00DC + ((c>>8)&0x03)); \
	} \
	}

#define WRITE_UTF16BE(z, c) { \
	if (c <= 0xFFFF) { \
	*z++ = (unsigned char)((c>>8)&0x00FF); \
	*z++ = (unsigned char)(c&0x00FF); \
	} else { \
	*z++ = (unsigned char)(0x00D8 + (((c-0x10000)>>18)&0x03)); \
	*z++ = (unsigned char)(((c>>10)&0x003F) + (((c-0x10000)>>10)&0x00C0)); \
	*z++ = (unsigned char)(0x00DC + ((c>>8)&0x03)); \
	*z++ = (unsigned char)(c&0x00FF); \
	} \
	}

#define READ_UTF16LE(z, TERM, c) { \
	c = (*z++); \
	c += ((*z++)<<8); \
	if (c >= 0xD800 && c < 0xE000 && TERM) { \
	int c2 = (*z++); \
	c2 += ((*z++)<<8); \
	c = (c2&0x03FF) + ((c&0x003F)<<10) + (((c&0x03C0)+0x0040)<<10); \
	} \
	}

#define READ_UTF16BE(z, TERM, c) { \
	c = ((*z++)<<8); \
	c += (*z++); \
	if (c >= 0xD800 && c < 0xE000 && TERM) { \
	int c2 = ((*z++)<<8); \
	c2 += (*z++); \
	c = (c2&0x03FF) + ((c&0x003F)<<10) + (((c&0x03C0)+0x0040)<<10); \
	} \
	}

#define READ_UTF8(z, term, c) \
	c = *(z++); \
	if (c >= 0xc0) { \
	c = __utf8Trans1[c-0xc0]; \
	while (z != term && (*z & 0xc0) == 0x80) { \
	c = (c<<6) + (0x3f & *(z++)); \
	} \
	if (c < 0x80 || (c&0xFFFFF800) == 0xD800 || (c&0xFFFFFFFE) == 0xFFFE) c = 0xFFFD; \
	}

#pragma endregion

/* Translate a single UTF-8 character.  Return the unicode value.
**
** During translation, assume that the byte that zTerm points is a 0x00.
**
** Write a pointer to the next unread byte back into *pzNext.
*/
__device__ unsigned int utf8read(const unsigned char **z) //: sqlite3Utf8Read
{
	// Same as READ_UTF8() above but without the zTerm parameter. For this routine, we assume the UTF8 string is always zero-terminated.
	unsigned int c = *((*z)++);
	if (c >= 0xc0) {
		c = __utf8Trans1[c-0xc0];
		while ((*(*z) & 0xc0) == 0x80)
			c = (c<<6) + (0x3f & *((*z)++));
		if (c < 0x80 || (c&0xFFFFF800) == 0xD800 || (c&0xFFFFFFFE) == 0xFFFE) c = 0xFFFD;
	}
	return c;
}

/* pZ is a UTF-8 encoded unicode string. If nByte is less than zero, return the number of unicode characters in pZ up to (but not including)
** the first 0x00 byte. If nByte is not less than zero, return the number of unicode characters in the first nByte of pZ (or up to 
** the first 0x00, whichever comes first).
*/
__device__ int utf8charlen(const char *z, int bytes) //: sqlite3Utf8CharLen
{
	const char *term = bytes >= 0 ? &z[bytes] : (const char *)-1;
	assert(z <= term);
	int r = 0; while (*z != 0 && z < term) { _STRSKIPUTF8(z); r++; }
	return r;
}

#if defined(_TEST) && defined(_DEBUG)
/* Translate UTF-8 to UTF-8.
**
** This has the effect of making sure that the string is well-formed UTF-8.  Miscoded characters are removed.
**
** The translation is done in-place and aborted if the output overruns the input.
*/
__device__ int utf8to8(unsigned char *z) //: sqlite3Utf8To8
{
	unsigned char *r = z;
	unsigned char *start = z;
	while (z[0] && r <= z) {
		unsigned int c = utf8read((const unsigned char **)&z);
		if (c != 0xfffd) WRITE_UTF8(r, c);
	}
	*r = 0;
	return (int)(r - start);
}
#endif

#ifndef OMIT_UTF16
/*
** Convert a UTF-16 string in the native encoding into a UTF-8 string.
** Memory to hold the UTF-8 string is obtained from sqlite3_malloc and must
** be freed by the calling function.
**
** NULL is returned if there is an allocation error.
*/
//__device__ char *utf16to8(sqlite3 *db, const void *z, int bytes, TEXTENCODE encode)
//{
//	Mem m;
//	memset(&m, 0, sizeof(m));
//	m.db = db;
//	sqlite3VdbeMemSetStr(&m, z, bytes, encode, SQLITE_STATIC);
//	sqlite3VdbeChangeEncoding(&m, TEXTENCODE_UTF8);
//	if (db->mallocFailed) {
//		sqlite3VdbeMemRelease(&m);
//		m.z = 0;
//	}
//	assert((m.flags & MEM_Term) || db->mallocFailed);
//	assert((m.flags & MEM_Str) || db->mallocFailed);
//	assert(m.z || db->mallocFailed);
//	return m.z;
//}


/* zIn is a UTF-16 encoded unicode string at least nChar characters long. Return the number of bytes in the first nChar unicode characters
** in pZ.  nChar must be non-negative.
*/
__device__ int utf16bytelen(const void *z, int chars) //: sqlite3Utf16ByteLen
{
	int c;
	unsigned char const *r = (unsigned char const *)z;
	int n = 0;
	if (LIBCU_UTF16NATIVE == TEXTENCODE_UTF16BE) while (n < chars) { READ_UTF16BE(r, 1, c); n++; }
	else while (n < chars) { READ_UTF16LE(r, 1, c); n++; }
	return (int)(r - (unsigned char const *)z);
}

#ifdef _TEST
/* This routine is called from the TCL test function "translate_selftest". It checks that the primitives for serializing and deserializing
** characters in each encoding are inverses of each other.
*/
__device__ void utfselftest() //: sqlite3UtfSelfTest
{
	unsigned int i, t;
	unsigned char buf[20];
	unsigned char *z;
	int n;
	unsigned int c;
	for (i = 0; i < 0x00110000; i++) {
		z = buf;
		WRITE_UTF8(z, i);
		n = (int)(z - buf);
		assert(n > 0 && n <= 4);
		z[0] = 0;
		z = buf;
		c = utf8read((const unsigned char **)&z);
		t = i;
		if (i >= 0xD800 && i <= 0xDFFF) t = 0xFFFD;
		if ((i&0xFFFFFFFE) == 0xFFFE) t = 0xFFFD;
		assert(c == t);
		assert((z - buf) == n);
	}
	for (i = 0; i < 0x00110000; i++) {
		if (i >= 0xD800 && i < 0xE000) continue;
		z = buf;
		WRITE_UTF16LE(z, i);
		n = (int)(z - buf);
		assert(n > 0 && n <= 4);
		z[0] = 0;
		z = buf;
		READ_UTF16LE(z, 1, c);
		assert(c == i);
		assert((z - buf) == n);
	}
	for (i = 0; i < 0x00110000; i++) {
		if (i >= 0xD800 && i < 0xE000) continue;
		z = buf;
		WRITE_UTF16BE(z, i);
		n = (int)(z - buf);
		assert(n > 0 && n <= 4);
		z[0] = 0;
		z = buf;
		READ_UTF16BE(z, 1, c);
		assert(c == i);
		assert((z - buf) == n);
	}
}
#endif
#endif

