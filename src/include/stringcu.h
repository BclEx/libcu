/*
string.h - declarations for string manipulation functions
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

//#pragma once

#if defined(__CUDA_ARCH__) || defined(LIBCUFORCE)
#ifndef _STRINGCU_H
#define _STRING_H
#define _INC_STRING
#include <featurescu.h>
#include <crtdefscu.h>
#include <stdargcu.h>

__BEGIN_DECLS;

// builtin
extern void *__cdecl memset(void *, int, size_t);
extern void *__cdecl memcpy(void *, const void *, size_t);

__BEGIN_NAMESPACE_STD;
/* Copy N bytes of SRC to DEST.  */
//builtin: extern __device__ void *memcpy(void *__restrict dest, const void *__restrict src, size_t n);
#define _memcpy(dest, src, length) if (length) memcpy(dest, src, length)

/* Copy N bytes of SRC to DEST, guaranteeing correct behavior for overlapping strings.  */
extern __device__ void *memmove(void *dest, const void *src, size_t n);
__END_NAMESPACE_STD;

__BEGIN_NAMESPACE_STD;
/* Set N bytes of S to C.  */
//builtin: extern __device__ void *memset(void *s, int c, size_t n);
#define _memset(dest, value, length) if (length) memset(dest, value, length)
/* Compare N bytes of S1 and S2.  */
extern __device__ int memcmp_(const void *s1, const void *s2, size_t n);
#define memcmp memcmp_
/* Search N bytes of S for C.  */
extern __device__ void *memchr(const void *s, int c, size_t n);
__END_NAMESPACE_STD;

__BEGIN_NAMESPACE_STD;
/* Copy SRC to DEST.  */
extern __device__ char *strcpy_(char *__restrict dest, const char *__restrict src);
#define strcpy strcpy_
/* Copy no more than N characters of SRC to DEST.  */
extern __device__ char *strncpy(char *__restrict dest, const char *__restrict src, size_t n);

/* Append SRC onto DEST.  */
extern __device__ char *strcat_(char *__restrict dest, const char *__restrict src);
#define strcat strcat_
/* Append no more than N characters from SRC onto DEST.  */
extern __device__ char *strncat(char *__restrict dest, const char *__restrict src, size_t n);

/* Compare S1 and S2.  */
extern __device__ int strcmp(const char *s1, const char *s2);
/* Compare S1 and S2. Case insensitive.  */
extern __device__ int stricmp(const char *s1, const char *s2);
/* Compare N characters of S1 and S2.  */
extern __device__ int strncmp(const char *s1, const char *s2, size_t n);
/* Compare N characters of S1 and S2. Case insensitive.  */
extern __device__ int strnicmp(const char *s1, const char *s2, size_t n);

/* Compare the collated forms of S1 and S2.  */
extern __device__ int strcoll(const char *s1, const char *s2);
/* Put a transformation of SRC into no more than N bytes of DEST.  */
extern __device__ size_t strxfrm(char *__restrict dest, const char *__restrict src, size_t n);
__END_NAMESPACE_STD;

/* Duplicate S, returning an identical malloc'd string.  */
extern __device__ char *strdup(const char *s);
/* Return a malloc'd copy of at most N bytes of STRING.  The resultant string is terminated even if no null terminator appears before STRING[N].  */
extern __device__ char *strndup(const char *s, size_t n);

__BEGIN_NAMESPACE_STD;
/* Find the first occurrence of C in S.  */
extern __device__ char *strchr(const char *s, int c);
/* Find the last occurrence of C in S.  */
extern __device__ char *strrchr(const char *s, int c);
__END_NAMESPACE_STD;

__BEGIN_NAMESPACE_STD;
/* Return the length of the initial segment of S which consists entirely of characters not in REJECT.  */
extern __device__ size_t strcspn(const char *s, const char *reject);
/* Return the length of the initial segment of S which consists entirely of characters in ACCEPT.  */
extern __device__ size_t strspn(const char *s, const char *accept);
/* Find the first occurrence in S of any character in ACCEPT.  */
extern __device__ char *strpbrk(const char *s, const char *accept);
/* Find the first occurrence of NEEDLE in HAYSTACK.  */
extern __device__ char *strstr(const char *haystack, const char *needle);

/* Divide S into tokens separated by characters in DELIM.  */
extern __device__ char *strtok(char *__restrict s, const char *__restrict delim);
__END_NAMESPACE_STD;

extern __device__ void *mempcpy(void *__restrict dest, const void *__restrict src, size_t n);

__BEGIN_NAMESPACE_STD;
/* Return the length of S.  */
extern __device__ size_t strlen_(const char *s);
#define strlen strlen_
//__forceinline __device__ size_t strlen(const char *s)
//{
//	if (!s) return 0;
//	register const char *s2 = s;
//	while (*s2) { s2++; }
//	return 0x3fffffff & (int)(s2 - s);
//}

/* Return the length of S.  */
//extern __device__ size_t strlen16(const char *s);
__forceinline __device__ size_t strlen16(const void *s)
{
	if (!s) return 0;
	register const char *s2 = (const char *)s;
	int n; for (n = 0; s2[n] || s2[n+1]; n += 2) { }
	return n;
}
__END_NAMESPACE_STD;

/* Find the length of STRING, but scan at most MAXLEN characters. If no '\0' terminator is found in that many characters, return MAXLEN.  */
extern __device__ size_t strnlen(const char *s, size_t maxlen);

__BEGIN_NAMESPACE_STD;
/* Return a string describing the meaning of the `errno' code in ERRNUM.  */
extern __device__ char *strerror(int errnum);
__END_NAMESPACE_STD;

//#ifndef PRINT_BUF_SIZE
//#define PRINT_BUF_SIZE 70
//#endif
//
//typedef struct strbld_t
//{
//	void *tag;			// Optional database for lookaside.  Can be NULL
//	char *base;			// A base allocation.  Not from malloc.
//	char *text;			// The string collected so far
//	int index;			// Length of the string so far
//	size_t size;		// Amount of space allocated in zText
//	int maxSize;		// Maximum allowed string length
//	bool allocFailed;	// Becomes true if any memory allocation fails
//	unsigned char allocType; // 0: none,  1: _tagalloc,  2: _alloc
//	bool overflowed;    // Becomes true if string size exceeds limits
//} strbld_t;
//
//__device__ void strbldInit(strbld_t *b, char *text = nullptr, int capacity = -1, int maxAlloc = -1);
//__device__ void strbldAppendSpace(strbld_t *b, int length);
//__device__ void strbldAppendFormat(strbld_t *b, bool useExtended, const char *fmt, va_list args);
//__device__ void strbldAppend(strbld_t *b, const char *str, int length);
//__device__ __forceinline void strbldAppendElement(strbld_t *b, const char *str) { strbldAppend(b, ", ", 2); strbldAppend(b, str, strlen(str)); }
//__device__ char *strbldToString(strbld_t *b);
//__device__ void strbldReset(strbld_t *b);

__END_DECLS;

#define memcpy_(dest, src, length) if (length) memcpy(dest, src, length)

#endif  /* _STRINGCU_H */
#else
#include <string.h>
#define memcpy_ memcpy
#endif

#ifndef _STRINGCU_H
//#include <crtdefscu.h>
#ifdef __cplusplus
extern "C" {
#endif

#ifndef PRINT_BUF_SIZE
#define PRINT_BUF_SIZE 70
#endif

	typedef struct strbld_t
	{
		void *tag;			// Optional database for lookaside.  Can be NULL
		char *base;			// A base allocation.  Not from malloc.
		char *text;			// The string collected so far
		int index;			// Length of the string so far
		size_t size;		// Amount of space allocated in zText
		int maxSize;		// Maximum allowed string length
		bool allocFailed;	// Becomes true if any memory allocation fails
		unsigned char allocType; // 0: none,  1: _tagalloc,  2: _alloc
		bool overflowed;    // Becomes true if string size exceeds limits
	} strbld_t;

	__device__ void strbldInit(strbld_t *b, char *text = nullptr, int capacity = -1, int maxAlloc = -1);
	__device__ void strbldAppendSpace(strbld_t *b, int length);
	__device__ void strbldAppendFormat(strbld_t *b, bool useExtended, const char *fmt, va_list args);
	__device__ void strbldAppend(strbld_t *b, const char *str, int length);
	__device__ __forceinline void strbldAppendElement(strbld_t *b, const char *str) { strbldAppend(b, ", ", 2); strbldAppend(b, str, (int)strlen(str)); }
	__device__ char *strbldToString(strbld_t *b);
	__device__ void strbldReset(strbld_t *b);

#ifdef  __cplusplus
}
#endif
#endif
#define _STRINGCU_H