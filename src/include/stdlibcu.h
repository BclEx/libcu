/*
stdlibcu.h - declarations/definitions for commonly used library functions
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

#ifndef _STDLIBCU_H
#ifdef  __cplusplus
extern "C" {
#endif

	/* Shorthand for type of comparison functions.  */
#ifndef __COMPAR_FN_T
#define __COMPAR_FN_T
	typedef int (*__compar_fn_t)(const void *, const void *);
#endif

#ifdef  __cplusplus
}
#endif
#endif

#if defined(__CUDA_ARCH__) || defined(LIBCUFORCE)
#ifndef _STDLIBCU_H
#define _STDLIB_H
#define _INC_STDLIB
#include <featurescu.h>
#include <crtdefscu.h>
#include <limits.h>

#include <sentinel-stdlibmsg.h>
__BEGIN_DECLS;

extern __device__ unsigned long _stdlib_strto_l(register const char *__restrict str, char **__restrict endptr, int base, int sflag);
#if defined(ULLONG_MAX)
extern __device__ unsigned long long _stdlib_strto_ll(register const char *__restrict str, char **__restrict endptr, int base, int sflag);
#endif

__BEGIN_NAMESPACE_STD;
/* Returned by `div'.  */
typedef struct
{
	int quot;			/* Quotient.  */
	int rem;			/* Remainder.  */
} div_t;
/* Returned by `ldiv'.  */
typedef struct
{
	long int quot;		/* Quotient.  */
	long int rem;		/* Remainder.  */
} ldiv_t;
__END_NAMESPACE_STD;

#if defined(ULLONG_MAX)
__BEGIN_NAMESPACE_C99;
/* Returned by `lldiv'.  */
typedef struct
{
	long long int quot;		/* Quotient.  */
	long long int rem;		/* Remainder.  */
} lldiv_t;
__END_NAMESPACE_C99;
#endif

/* The largest number rand will return (same as INT_MAX).  */
#define	RAND_MAX	2147483647

/* We define these the same for all machines. Changes from this to the outside world should be done in `_exit'.  */
#define	EXIT_FAILURE	1	/* Failing exit status.  */
#define	EXIT_SUCCESS	0	/* Successful exit status.  */

__BEGIN_NAMESPACE_STD;
/* prototype */
extern __device__ double strtod(const char *__restrict nptr, char **__restrict endptr);

/* Convert a string to a floating-point number.  */
__forceinline __device__ double atof(const char *nptr) { return strtod(nptr, NULL); }
/* Convert a string to an integer.  */
__forceinline __device__ int atoi(const char *nptr) { return (int)_stdlib_strto_l(nptr, (char **)NULL, 10, 1); }
/* Convert a string to a long integer.  */
__forceinline __device__ long int atol(const char *nptr) { return _stdlib_strto_l(nptr, (char **)NULL, 10, 1); }
__END_NAMESPACE_STD;

#if defined(ULLONG_MAX)
__BEGIN_NAMESPACE_C99;
/* Convert a string to a long long integer.  */
__forceinline __device__ long long int atoll(const char *nptr) { return _stdlib_strto_ll(nptr, (char **)NULL, 10, 1); }
__END_NAMESPACE_C99;
#endif

__BEGIN_NAMESPACE_STD;
/* Convert a string to a floating-point number.  */
extern __device__ double strtod(const char *__restrict nptr, char **__restrict endptr);
__END_NAMESPACE_STD;

__BEGIN_NAMESPACE_C99;
/* Likewise for `float' and `long double' sizes of floating-point numbers.  */
extern __device__ float strtof(const char *__restrict nptr, char **__restrict endptr);
extern __device__ long double strtold(const char *__restrict nptr, char **__restrict endptr);
__END_NAMESPACE_C99;

__BEGIN_NAMESPACE_STD;
/* Convert a string to a long integer.  */
__forceinline __device__ long int strtol(const char *__restrict nptr, char **__restrict endptr, int base) { return _stdlib_strto_l(nptr, endptr, base, 1); }
/* Convert a string to an unsigned long integer.  */
__forceinline __device__ unsigned long int strtoul(const char *__restrict nptr, char **__restrict endptr, int base) { return _stdlib_strto_l(nptr, endptr, base, 0); }
__END_NAMESPACE_STD;

#if defined(ULLONG_MAX)
__BEGIN_NAMESPACE_C99;
/* Convert a string to a quadword integer.  */
__forceinline __device__ long long int strtoll(const char *__restrict nptr, char **__restrict endptr, int base) { return _stdlib_strto_ll(nptr, endptr, base, 1); }
/* Convert a string to an unsigned quadword integer.  */
__forceinline __device__ unsigned long long int strtoull(const char *__restrict nptr, char **__restrict endptr, int base) { return _stdlib_strto_ll(nptr, endptr, base, 0); }
__END_NAMESPACE_C99;
#endif

__BEGIN_NAMESPACE_STD;
/* Return a random integer between 0 and RAND_MAX inclusive.  */
extern __device__ int rand(void);
/* Seed the random number generator with the given number.  */
extern __device__ void srand(unsigned int seed);
__END_NAMESPACE_STD;

__BEGIN_NAMESPACE_STD;
/* Allocate SIZE bytes of memory.  */
extern __device__ void *malloc_(size_t size);
/* Allocate NMEMB elements of SIZE bytes each, all initialized to 0.  */
extern __device__ void *calloc_(size_t nmemb, size_t size);
__END_NAMESPACE_STD;

__BEGIN_NAMESPACE_STD;
/* Re-allocate the previously allocated block in PTR, making the new block SIZE bytes long.  */
extern __device__ void *realloc_(void *ptr, size_t size);
/* Free a block allocated by `malloc', `realloc' or `calloc'.  */
extern __device__ void free_(void *ptr);
__END_NAMESPACE_STD;
#define malloc malloc_
#define calloc calloc_
#define realloc realloc_
#define free free_

__BEGIN_NAMESPACE_STD;
/* Abort execution and generate a core-dump.  */
__forceinline __device__ void abort(void) { asm("trap;"); }
/* Register a function to be called when `exit' is called.  */
extern __device__ int atexit(void(*func)(void));
//__forceinline __device__ int atexit(void(*func)(void)) { panic("Not Implemented"); }
__END_NAMESPACE_STD;

__BEGIN_NAMESPACE_STD;
/* Call all functions registered with `atexit' and `on_exit', in the reverse of the order in which they were registered, perform stdio cleanup, and terminate program execution with STATUS.  */
__forceinline __device__ void exit_(int status) { stdlib_exit msg(true, status); }
__END_NAMESPACE_STD;

__BEGIN_NAMESPACE_C99;
/* Terminate the program with STATUS without calling any of the functions registered with `atexit' or `on_exit'.  */
__forceinline __device__ void _Exit(int status) { stdlib_exit msg(false, status); }
__END_NAMESPACE_C99;

__BEGIN_NAMESPACE_STD;
/* Return the value of envariable NAME, or NULL if it doesn't exist.  */
extern __device__ char *getenv(const char *name);
__END_NAMESPACE_STD;

/* Set NAME to VALUE in the environment. If REPLACE is nonzero, overwrite an existing value.  */
extern __device__ int setenv(const char *name, const char *value, int replace);
/* Remove the variable NAME from the environment.  */
extern __device__ int unsetenv(const char *name);

/* Generate a unique temporary file name from TEMPLATE.
The last six characters of TEMPLATE must be "XXXXXX"; they are replaced with a string that makes the file name unique.
Returns TEMPLATE, or a null pointer if it cannot get a unique file name.  */
extern __device__ char *mktemp(char *template_);
/* Generate a unique temporary file name from TEMPLATE.
The last six characters of TEMPLATE must be "XXXXXX"; they are replaced with a string that makes the filename unique.
Returns a file descriptor open on the file for reading and writing, or -1 if it cannot create a uniquely-named file. */
#ifndef __USE_FILE_OFFSET64
extern __device__ int mkstemp(char *template_);
#else
#define mkstemp mkstemp64
#endif
#ifdef __USE_LARGEFILE64
extern __device__ int mkstemp64(char *template_);
#endif

__BEGIN_NAMESPACE_STD;
/* Execute the given line as a shell command.  */
__forceinline __device__ int system(const char *command) { stdlib_system msg(command); return msg.RC; }
__END_NAMESPACE_STD;

__BEGIN_NAMESPACE_STD;
/* Do a binary search for KEY in BASE, which consists of NMEMB elements of SIZE bytes each, using COMPAR to perform the comparisons.  */
extern __device__ void *bsearch(const void *key, const void *base, size_t nmemb, size_t size, __compar_fn_t compar);
/* Sort NMEMB elements of BASE, of SIZE bytes each, using COMPAR to perform the comparisons.  */
extern __device__ void qsort(void *base, size_t nmemb, size_t size, __compar_fn_t compar);

/* Return the absolute value of X.  */
__forceinline __device__ int abs_(int x) { return x >= 0 ? x : -x; }
#define abs abs_
__forceinline __device__ long int labs_(long int x) { return x >= 0 ? x : -x; }
#define labs labs_
__END_NAMESPACE_STD;
#if defined(ULLONG_MAX)
__BEGIN_NAMESPACE_C99;
__forceinline __device__ long long int llabs_(long long int x) { return x >= 0 ? x : -x; }
#define llabs llabs_
__END_NAMESPACE_C99;
#endif

__BEGIN_NAMESPACE_STD;
/* Return the `div_t', `ldiv_t' or `lldiv_t' representation of the value of NUMER over DENOM. */
extern __device__ div_t div(int numer, int denom);
extern __device__ ldiv_t ldiv(long int numer, long int denom);
__END_NAMESPACE_STD;
#if defined(ULLONG_MAX)
__BEGIN_NAMESPACE_C99;
extern __device__ lldiv_t lldiv(long long int numer, long long int denom);
__END_NAMESPACE_C99;
#endif

__BEGIN_NAMESPACE_STD;
/* Return the length of the multibyte character in S, which is no longer than N.  */
extern __device__ int mblen(const char *s, size_t n);
/* Return the length of the given multibyte character, putting its `wchar_t' representation in *PWC.  */
extern __device__ int mbtowc(wchar_t *__restrict __pwc, const char *__restrict s, size_t n);
/* Put the multibyte character represented by WCHAR in S, returning its length.  */
extern __device__ int wctomb(char *s, wchar_t wchar);

/* Convert a multibyte string to a wide char string.  */
extern __device__ size_t mbstowcs(wchar_t *__restrict  pwcs, const char *__restrict s, size_t n);
/* Convert a wide char string to multibyte string.  */
extern __device__ size_t wcstombs(char *__restrict s, const wchar_t *__restrict pwcs, size_t n);
__END_NAMESPACE_STD;

// override
#define _CRT_ATOF_DEFINED

__END_DECLS;

#endif  /* _STDLIBCU_H */
#else
#include <stdlib.h>
#define strtoll
#define strtoull
#define exit_ exit
#endif

#ifndef _STDLIBCU_H
__BEGIN_DECLS;

#if defined(ULLONG_MAX)
/* Returned by `strtoq'.  */
typedef long long int quad_t;
/* Returned by `strtouq'.  */
typedef unsigned long long int u_quad_t;
/* Convert a string to a quadword integer.  */
__forceinline __device__ quad_t strtoq(const char *__restrict nptr, char **__restrict endptr, int base) { return (quad_t)strtol(nptr, endptr, base); }
/* Convert a string to an unsigned quadword integer.  */
__forceinline __device__ u_quad_t strtouq(const char *__restrict nptr, char **__restrict endptr, int base) { return (u_quad_t)strtoul(nptr, endptr, base); }
#endif

__END_DECLS;
#endif
#define _STDLIBCU_H