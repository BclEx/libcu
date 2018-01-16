/*
utf.h - xxx
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
#ifndef _EXT_UTF_H
#define _EXT_UTF_H
__BEGIN_DECLS;

#define _strskiputf8(z) { if ((*(z++)) >= 0xc0) while ((*z & 0xc0) == 0x80) { z++; } }
//template <typename T> __device__ inline void _strskiputf8(const T *z) { if (*(z++) >= 0xc0) while ((*z & 0xc0) == 0x80) { z++; } }
extern "C" __device__ unsigned int _utf8read(const unsigned char **z);
extern "C" __device__ int _utf8charlength(const char *z, int bytes);
#if _DEBUG
extern "C" __device__ int _utf8to8(unsigned char *z);
#endif
#ifndef OMIT_UTF16
extern "C" __device__ int _utf16bytelength(const void *z, int chars);
#ifdef _TEST
extern "C" __device__ void _runtime_utfselftest();
#endif
#endif

#ifdef _UNICODE
#define char_t unsigned short
#define MAX_CHAR 0xFFFF
#define _L(c) L##c
//#define _isprint iswprint
//#define _strlen wcslen
//#define _printf wprintf
#else
#define char_t char
#define MAX_CHAR 0xFF
#define _L(c) (c) 
//#define _isprint isprint
//#define _strlen strlen
//#define _printf printf
#endif

// No utf-8 support. 1 byte = 1 char
#define utf8_strlen(S, B) ((B) < 0 ? _strlen(S) : (B))
#define utf8_tounicode(S, CP) (*(CP) = (unsigned char)*(S), 1)
#define utf8_getchars(CP, C) (*(CP) = (C), 1)
#define utf8_upper(C) __toupper(C)
#define utf8_title(C) __toupper(C)
#define utf8_lower(C) __tolower(C)
#define utf8_index(C, I) (I)
#define utf8_charlen(C) 1
#define utf8_prev_len(S, L) 1

__END_DECLS;
#endif	/* _EXT_UTF_H */