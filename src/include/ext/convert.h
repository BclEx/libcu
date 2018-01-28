/*
convert.h - xxx
The MIT License

Copyright (c) 2016 Sky Morey

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#pragma once
#ifndef _EXT_CONVERT_H
#define _EXT_CONVERT_H
#include <stdint.h>
__BEGIN_DECLS;

#define TEXTENCODE uint8_t
#define TEXTENCODE_UTF8	1
#define TEXTENCODE_UTF16LE 2
#define TEXTENCODE_UTF16BE 3
#define TEXTENCODE_UTF16 4 // Use native byte order
#define TEXTENCODE_ANY 5 // libcu_create_function only
#define TEXTENCODE_UTF16_ALIGNED 8 // libcu_create_collation only

//////////////////////
// ATOX
#pragma region ATOX

/* The string z[] is an text representation of a real number. Convert this string to a double and write it into *r. */
extern __device__ bool convert_atofe(const char *z, double *r, int length, TEXTENCODE encode); //: sqlite3AtoF
__forceinline __device__ double convert_atof(const char *z) { double r = 0; if (z) convert_atofe(z, &r, -1, TEXTENCODE_UTF8); return r; } //: sky
/* Convert z to a 64-bit signed integer.  z must be decimal. This routine does *not* accept hexadecimal notation. */
extern __device__ int convert_atoi64e(const char *z, int64_t *r, int length, TEXTENCODE encode); //: sqlite3Atoi64
/* Transform a UTF-8 integer literal, in either decimal or hexadecimal, into a 64-bit signed integer.  This routine accepts hexadecimal literals, whereas convert_atoi64e() does not. */
extern __device__ int convert_axtoi64e(const char *z, int64_t *r); //: sqlite3DecOrHexToI64
/* If z represents an integer that will fit in 32-bits, then set *r to that integer and return true.  Otherwise return false. */
extern __device__ bool convert_atoie(const char *z, int *r); //: sqlite3GetInt32
/* Return a 32-bit integer value extracted from a string.  If the string is not an integer, just return 0. */
//__forceinline __device__ int convert_atoi(const char *z) { int r = 0; if (z) convert_atoie(z, &r); return r; } //: sqlite3Atoi
extern __device__ int convert_atoi(const char *z); //: sqlite3Atoi
/* sqlite3HexToInt: Translate a single byte of Hex into an integer. */
extern __device__ uint8_t convert_xtoi(int h); //: sqlite3HexToInt

//#define convert_itoa(i, b) convert_itoa64((int64_t)i, b)
//?extern __device__ char *convert_itoa64(int64_t i, char *b);

#pragma endregion

//////////////////////
// VARINT
#pragma region VARINT

extern __device__ int convert_putvarint(unsigned char *p, uint64_t v);
#define _putvarint32(A, B) (unsigned char)(((uint32_t)(B)<(uint32_t)0x80)?(*(A)=(unsigned char)(B)),1:convert_putvarint((A),(B)))
#define _putvarint convert_putvarint
extern __device__ uint8_t convert_getvarint(const unsigned char *p, uint64_t *v);
extern __device__ uint8_t convert_getvarint32(const unsigned char *p, uint32_t *v);
#define _getvarint32(A,B) (unsigned char)((*(A)<(uint8_t)0x80)?((B)=(uint32_t)*(A)),1:convert_getvarint32((A),(uint32_t *)&(B)))
#define _getvarint convert_getvarint
extern __device__ int convert_getvarintLength(uint64_t v);

#pragma endregion

#if 0
__forceinline __device__ uint16_t convert_get2nz(const uint8_t *p) { return ((((int)((p[0]<<8) | p[1]) -1)&0xffff)+1); }
__forceinline __device__ uint16_t convert_get2(const uint8_t *p) { return (p[0]<<8) | p[1]; }
__forceinline __device__ void convert_put2(unsigned char *p, uint32_t v) { p[0] = (uint8_t)(v>>8); p[1] = (uint8_t)v; }
__forceinline __device__ uint32_t convert_get4(const uint8_t *p) { return (p[0]<<24) | (p[1]<<16) | (p[2]<<8) | p[3]; }
__forceinline __device__ void convert_put4(unsigned char *p, uint32_t v) { p[0] = (uint8_t)(v>>24); p[1] = (uint8_t)(v>>16); p[2] = (uint8_t)(v>>8); p[3] = (uint8_t)v; }
#else
extern __device__ uint16_t convert_get2nz(const uint8_t *p);
extern __device__ uint16_t convert_get2(const uint8_t *p);
extern __device__ void convert_put2(unsigned char *p, uint32_t v);
extern __device__ uint32_t convert_get4(const uint8_t *p);
extern __device__ void convert_put4(unsigned char *p, uint32_t v);
#endif

extern __device__ uint8_t convert_atolevel(const char *z, int omitFull, uint8_t dflt);
extern __device__ bool convert__atob(const char *z, uint8_t dflt);

__END_DECLS;
#endif	/* _EXT_CONVERT_H */