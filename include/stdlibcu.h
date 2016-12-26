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

#pragma once

#if !__CUDACC__
#include <stdlib.h>
#elif !defined(_INC_STDLIB)
#define _INC_STDLIB
#include <crtdefscu.h>
#include <limits.h>

extern __device__ unsigned long _stdlib_strto_l(register const char * __restrict str, char **__restrict endptr, int base, int sflag);
#if defined(ULLONG_MAX)
extern __device__ unsigned long long _stdlib_strto_ll(register const char * __restrict str, char ** __restrict endptr, int base, int sflag);
#endif

#ifdef  __cplusplus
extern "C" {
#endif

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

#if defined(ULLONG_MAX)
	/* Returned by `lldiv'.  */
	typedef struct
	{
		long long int quot;		/* Quotient.  */
		long long int rem;		/* Remainder.  */
	} lldiv_t;
#endif

	/* The largest number rand will return (same as INT_MAX).  */
#define	RAND_MAX	2147483647

	/* We define these the same for all machines. Changes from this to the outside world should be done in `_exit'.  */
#define	EXIT_FAILURE	1	/* Failing exit status.  */
#define	EXIT_SUCCESS	0	/* Successful exit status.  */

	/* Convert a string to a floating-point number.  */
	extern __device__ double strtod(const char *__restrict nptr, char **__restrict endptr);

	/* Convert a string to a floating-point number.  */
	__forceinline __device__ double atof(const char *nptr) { return strtod(nptr, NULL); }
	/* Convert a string to an integer.  */
	__forceinline __device__ int atoi(const char *nptr) { return (int)_stdlib_strto_l(nptr, (char **)NULL, 10, 1); }
	/* Convert a string to a long integer.  */
	__forceinline __device__ long int atol(const char *nptr) { return _stdlib_strto_l(nptr, (char **)NULL, 10, 1); }
#if defined(ULLONG_MAX)
	/* Convert a string to a long long integer.  */
	__forceinline __device__ long long int atoll(const char *nptr) { return _stdlib_strto_ll(nptr, (char **)NULL, 10, 1); }
#endif

	/* C99: Likewise for `float' and `long double' sizes of floating-point numbers.  */
	extern __device__ float strtof(const char *__restrict nptr, char **__restrict endptr);
#if 0
	extern __device__ long double strtold(const char *__restrict nptr, char **__restrict endptr);
#endif

	/* Convert a string to a long integer.  */
	__forceinline __device__ long int strtol(const char *__restrict nptr, char **__restrict endptr, int base) { return _stdlib_strto_l(nptr, endptr, base, 1); }
	/* Convert a string to an unsigned long integer.  */
	__forceinline __device__ unsigned long int strtoul(const char *__restrict nptr, char **__restrict endptr, int base) { return _stdlib_strto_l(nptr, endptr, base, 0); }

#if defined(ULLONG_MAX)
	/* Convert a string to a quadword integer.  */
	__forceinline __device__ long long int strtoll(const char *__restrict nptr, char **__restrict endptr, int base) { return _stdlib_strto_ll(nptr, endptr, base, 1); }
	/* Convert a string to an unsigned quadword integer.  */
	__forceinline __device__ unsigned long long int strtoull(const char *__restrict nptr, char **__restrict endptr, int base) { return _stdlib_strto_ll(nptr, endptr, base, 0); }
#endif

	/* Return a random integer between 0 and RAND_MAX inclusive.  */
	extern __device__ int rand(void);
	/* Seed the random number generator with the given number.  */
	extern __device__ void srand(unsigned int seed);


	/* Allocate SIZE bytes of memory.  */
	extern __device__ void *malloc_(size_t size);
	/* Allocate NMEMB elements of SIZE bytes each, all initialized to 0.  */
	extern __device__ void *calloc(size_t nmemb, size_t size);
	/* Re-allocate the previously allocated block in PTR, making the new block SIZE bytes long.  */
	extern __device__ void *realloc(void *ptr, size_t size);
	/* Free a block allocated by `malloc', `realloc' or `calloc'.  */
	extern __device__ void free_(void *ptr);

	/* Abort execution and generate a core-dump.  */
	extern __device__ void abort(void);
	/* Register a function to be called when `exit' is called.  */
	extern __device__ int atexit(void(*func)(void));
	/* Call all functions registered with `atexit' and `on_exit', in the reverse of the order in which they were registered, perform stdio cleanup, and terminate program execution with STATUS.  */
	extern __device__ void exit_(int status);

	/* Return the value of envariable NAME, or NULL if it doesn't exist.  */
	extern __device__ char *getenv(const char *name);
	/* Set NAME to VALUE in the environment. If REPLACE is nonzero, overwrite an existing value.  */
	//extern __device__ int setenv(const char *name, const char *value, int replace);
	/* Remove the variable NAME from the environment.  */
	//extern __device__ int unsetenv(const char *name);

	/* Generate a unique temporary file name from TEMPLATE.
	The last six characters of TEMPLATE must be "XXXXXX";
	they are replaced with a string that makes the file name unique.
	Returns TEMPLATE, or a null pointer if it cannot get a unique file name.  */
	//extern char *mktemp(char *template);
	/* Execute the given line as a shell command.  */
	extern __device__ int system(const char *command);

	/* Shorthand for type of comparison functions.  */
#ifndef __COMPAR_FN_T
#define __COMPAR_FN_T
	typedef int (*__compar_fn_t)(const void *, const void *);
#endif

	/* Do a binary search for KEY in BASE, which consists of NMEMB elements of SIZE bytes each, using COMPAR to perform the comparisons.  */
	extern __device__ void *bsearch(const void *key, const void *base, size_t nmemb, size_t size, __compar_fn_t compar);
	/* Sort NMEMB elements of BASE, of SIZE bytes each, using COMPAR to perform the comparisons.  */
	extern __device__ void qsort(void *base, size_t nmemb, size_t size, __compar_fn_t compar);

	/* Return the absolute value of X.  */
	__forceinline __device__ int abs(int x) { return x >= 0 ? x : -x; }
	__forceinline __device__ long int labs(long int x) { return x >= 0 ? x : -x; }
#if defined(ULLONG_MAX)
	__forceinline __device__ long long int llabs(long long int x) { return x >= 0 ? x : -x; }
#endif

	/* Return the `div_t', `ldiv_t' or `lldiv_t' representation of the value of NUMER over DENOM. */
	extern __device__ div_t div(int numer, int denom);
	extern __device__ ldiv_t ldiv(long int numer, long int denom);
#if defined(ULLONG_MAX)
	extern __device__ lldiv_t lldiv(long long int numer, long long int denom);
#endif

	/* Return the length of the multibyte character in S, which is no longer than N.  */
	extern __device__ int mblen(const char *s, size_t n);
	/* Return the length of the given multibyte character, putting its `wchar_t' representation in *PWC.  */
	extern __device__ int mbtowc(wchar_t *__restrict __pwc, const char *__restrict s, size_t n);
	/* Put the multibyte character represented by WCHAR in S, returning its length.  */
	extern __device__ int wctomb (char *s, wchar_t wchar);

	/* Convert a multibyte string to a wide char string.  */
	extern __device__ size_t mbstowcs(wchar_t *__restrict  pwcs, const char *__restrict s, size_t n);
	/* Convert a wide char string to multibyte string.  */
	extern __device__ size_t wcstombs(char *__restrict s, const wchar_t *__restrict pwcs, size_t n);


	#define _CRT_ATOF_DEFINED
#ifdef __cplusplus
}
#endif

#endif  /* _INC_STDLIB */