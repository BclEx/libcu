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

#pragma once

#if !__CUDACC__
#include <string.h>
#elif !defined(_INC_STRING)
#define _INC_STRING
#include <featurescu.h>
#include <crtdefscu.h>
#include <stdargcu.h>

__BEGIN_DECLS;

__BEGIN_NAMESPACE_STD;
/* Copy N bytes of SRC to DEST.  */
extern void *memcpy(void *__restrict dest, const void *__restrict src, size_t n);
/* Copy N bytes of SRC to DEST, guaranteeing correct behavior for overlapping strings.  */
extern void *memmove(void *dest, const void *src, size_t n);
__END_NAMESPACE_STD;

__BEGIN_NAMESPACE_STD;
/* Set N bytes of S to C.  */
extern void *memset(void *s, int c, size_t n);
/* Compare N bytes of S1 and S2.  */
extern int memcmp(const void *s1, const void *s2, size_t n);
/* Search N bytes of S for C.  */
extern void *memchr(const void *s, int c, size_t n);
__END_NAMESPACE_STD;


__BEGIN_NAMESPACE_STD;
/* Copy SRC to DEST.  */
extern char *strcpy(char *__restrict dest, const char *__restrict src);
/* Copy no more than N characters of SRC to DEST.  */
extern char *strncpy(char *__restrict dest, const char *__restrict src, size_t n);

/* Append SRC onto DEST.  */
extern char *strcat(char *__restrict dest, const char *__restrict src);
/* Append no more than N characters from SRC onto DEST.  */
extern char *strncat(char *__restrict dest, const char *__restrict src, size_t n);

/* Compare S1 and S2.  */
extern int strcmp(const char *s1, const char *s2);
/* Compare N characters of S1 and S2.  */
extern int strncmp(const char *s1, const char *s2, size_t n);

/* Compare the collated forms of S1 and S2.  */
extern int strcoll(const char *s1, const char *s2);
/* Put a transformation of SRC into no more than N bytes of DEST.  */
extern size_t strxfrm(char *__restrict dest, const char *__restrict src, size_t n);
__END_NAMESPACE_STD;

__BEGIN_NAMESPACE_STD;
/* Find the first occurrence of C in S.  */
extern char *strchr(const char *s, int c);
/* Find the last occurrence of C in S.  */
extern char *strrchr(const char *s, int c);
__END_NAMESPACE_STD;

__BEGIN_NAMESPACE_STD
/* Return the length of the initial segment of S which consists entirely of characters not in REJECT.  */
extern size_t strcspn(const char *s, const char *reject);
/* Return the length of the initial segment of S which consists entirely of characters in ACCEPT.  */
extern size_t strspn(const char *s, const char *accept);
/* Find the first occurrence in S of any character in ACCEPT.  */
extern char *strpbrk(const char *s, const char *accept);
/* Find the first occurrence of NEEDLE in HAYSTACK.  */
extern char *strstr(const char *haystack, const char *needle);

/* Divide S into tokens separated by characters in DELIM.  */
extern char *strtok(char *__restrict s, const char *__restrict delim);
__END_NAMESPACE_STD;

extern void *mempcpy(void *__restrict dest, const void *__restrict src, size_t n);

__BEGIN_NAMESPACE_STD;
/* Return the length of S.  */
//extern size_t strlen(const char *s);
__forceinline __device__ size_t strlen(const char *s)
{
	if (!s) return 0;
	register const char *s2 = s;
	while (*s2) { s2++; }
	return 0x3fffffff & (int)(s2 - s);
}
__END_NAMESPACE_STD;

__BEGIN_NAMESPACE_STD;
/* Return a string describing the meaning of the `errno' code in ERRNUM.  */
extern char *strerror(int errnum);
__END_NAMESPACE_STD;

#pragma region PRINT

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
__device__ __forceinline void strbldAppendElement(strbld_t *b, const char *str) { strbldAppend(b, ", ", 2); strbldAppend(b, str, strlen(str)); }
__device__ char *strbldToString(strbld_t *b);
__device__ void strbldReset(strbld_t *b);

__END_DECLS;

#endif  /* _INC_STRING */