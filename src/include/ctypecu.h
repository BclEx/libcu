/*
ctype.h - Character handling
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

#if !__CUDACC__
#include <ctype.h>
#elif !defined(_INC_CTYPE)
#define _INC_CTYPE
#include <crtdefscu.h>

// embed
extern __constant__ unsigned char __curtUpperToLower[256];
extern __constant__ unsigned char __curtCtypeMap[256]; 

/* set bit masks for the possible character types */
#define _DIGIT          0x04     /* digit[0-9] */
#define _HEX            0x08    /* hexadecimal digit */

extern __forceinline __device__ int isctype(int c, int type) { return (__curtCtypeMap[(unsigned char)c]&type)!=0; }
extern __forceinline __device__ int isalnum(int c) { return (__curtCtypeMap[(unsigned char)c]&0x06)!=0; }
extern __forceinline __device__ int isalpha(int c) { return (__curtCtypeMap[(unsigned char)c]&0x02)!=0; }
extern __forceinline __device__ int iscntrl(int c) { return (unsigned char)c<=0x1f||(unsigned char)c==0x7f; }
extern __forceinline __device__ int isdigit(int c) { return (__curtCtypeMap[(unsigned char)c]&0x04)!=0; }
extern __forceinline __device__ int islower(int c) { return __curtUpperToLower[(unsigned char)c]==c; }
extern __forceinline __device__ int isgraph(int c) { return 0; }
extern __forceinline __device__ int isprint(int c) { return (unsigned char)c>0x1f&&(unsigned char)c!=0x7f; }
extern __forceinline __device__ int ispunct(int c) { return 0; }
extern __forceinline __device__ int isspace(int c) { return (__curtCtypeMap[(unsigned char)c]&0x01)!=0; }
extern __forceinline __device__ int isupper(int c) { return (c&~(__curtCtypeMap[(unsigned char)c]&0x20))==c; }
extern __forceinline __device__ int isxdigit(int c) { return (__curtCtypeMap[(unsigned char)c]&0x08)!=0; }

/* Return the lowercase version of C.  */
extern __forceinline __device__ int tolower(int c) { return __curtUpperToLower[(unsigned char)c]; }

/* Return the uppercase version of C.  */
extern __forceinline __device__ int toupper(int c) { return c&~(__curtCtypeMap[(unsigned char)c]&0x20); }

#define _tolower(c) (char)((c)-'A'+'a')
#define _toupper(c) (char)((c)-'a'+'A')

/*C99*/
//extern __forceinline __device__ int isblank(int c);

extern __forceinline __device__ int isidchar(int c) { return (__curtCtypeMap[(unsigned char)c]&0x46)!=0; }

#endif  /* _INC_CTYPE */