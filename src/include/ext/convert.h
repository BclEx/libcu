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
#ifdef  __cplusplus
extern "C" {
#endif

	enum TEXTENCODE : uint8_t
	{
		TEXTENCODE_UTF8 = 1,
		TEXTENCODE_UTF16LE = 2,
		TEXTENCODE_UTF16BE = 3,
		TEXTENCODE_UTF16 = 4, // Use native byte order
		TEXTENCODE_ANY = 5, // sqlite3_create_function only
		TEXTENCODE_UTF16_ALIGNED = 8, // sqlite3_create_collation only
	};
	__device__ __forceinline void operator|=(TEXTENCODE &a, int b) { a = (TEXTENCODE)(a | b); }
	__device__ __forceinline void operator&=(TEXTENCODE &a, int b) { a = (TEXTENCODE)(a & b); }

#define convert_getvarint32(A,B) \
	(uint8_t)((*(A)<(uint8_t)0x80)?((B)=(uint32_t)*(A)),1:\
	convert_getvarint32_((A),(uint32_t *)&(B)))
#define convert_putvarint32(A,B) \
	(uint8_t)(((uint32_t)(B)<(uint32_t)0x80)?(*(A)=(unsigned char)(B)),1:\
	convert_putvarint32_((A),(B)))

#pragma region Varint
	extern __device__ int convert_putvarint(unsigned char *p, uint64_t v);
	extern __device__ int convert_putvarint32_(unsigned char *p, uint32_t v);
	extern __device__ uint8_t convert_getvarint(const unsigned char *p, uint64_t *v);
	extern __device__ uint8_t convert_getvarint32_(const unsigned char *p, uint32_t *v);
	extern __device__ int convert_getvarintLength(uint64_t v);
#pragma endregion
#pragma region AtoX
	extern __device__ bool convert_atof(const char *z, double *out, int length, TEXTENCODE encode);
	__forceinline __device__ double convert_atof(const char *z) { double out = 0; if (z) convert_atof(z, &out, -1, TEXTENCODE_UTF8); return out; }
	extern __device__ int convert_atoi64(const char *z, int64_5 *out, int length, TEXTENCODE encode);
	extern __device__ bool convert_atoi(const char *z, int *out);
	__forceinline __device__ int convert_atoi(const char *z) { int out = 0; if (z) convert_atoi(z, &out); return out; }
#define convert_itoa(i, b) convert_itoa64((int64_t)i, b)
	extern __device__ char *convert_itoa64(int64_t i, char *b);
#pragma endregion
#ifndef OMIT_INLINECONVERT
	__forceinline __device__ uint16_t convert_get2nz(const uint8_t *p) { return ((((int)((p[0]<<8) | p[1]) -1)&0xffff)+1); }
	__forceinline __device__ uint16_t convert_get2(const uint8_t *p) { return (p[0]<<8) | p[1]; }
	__forceinline __device__ void convert_put2(unsigned char *p, uint32_t v)
	{
		p[0] = (uint8)(v>>8);
		p[1] = (uint8)v;
	}
	__forceinline __device__ uint32_t convert_get4(const uint8_t *p) { return (p[0]<<24) | (p[1]<<16) | (p[2]<<8) | p[3]; }
	__forceinline __device__ void convert_put4(unsigned char *p, uint32_t v)
	{
		p[0] = (uint8)(v>>24);
		p[1] = (uint8)(v>>16);
		p[2] = (uint8)(v>>8);
		p[3] = (uint8)v;
	}
#else
	extern __device__ uint16_t convert_get2nz(const uint8_t *p);
	extern __device__ uint16_t convert_get2(const uint8_t *p);
	extern __device__ void convert_put2(unsigned char *p, uint32_t v);
	extern __device__ uint32_t convert_get4(const uint8_t *p);
	extern __device__ void convert_put4(unsigned char *p, uint32_t v);
#endif

	extern __device__ uint8_t convert_atolevel(const char *z, int omitFull, uint8_t dflt);
	extern __device__ bool convert__atob(const char *z, uint8_t dflt);

#ifdef  __cplusplus
}
#endif
#endif	/* _EXT_CONVERT_H */