/*
stdio.h - definitions/declarations for standard I/O routines
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

#ifndef __CUDA_ARCH__
#include <stdio.h>
#elif !defined(_INC_STDIOx)
#define _INC_STDIOx

#ifdef  __cplusplus
extern "C" {
#endif

	//#define _FSTDIO			// function stdio
#include <stddefcu.h>
#include <stdargcu.h>

	// <reent.h> defines __sFILE, _fpos_t.
	// They must be defined there because struct _reent needs them (and we don't want reent.h to include this file.
	struct __sFile {
		int unused;
	};
	//typedef struct __sFILE FILE;

#define _IOFBF 0	/* setvbuf should set fully buffered */
#define _IOLBF 1	/* setvbuf should set line buffered */
#define _IONBF 2	/* setvbuf should set unbuffered */

#ifndef NULL
#define NULL 0
#endif

#define BUFSIZ			1024

	/* End of file character. Some things throughout the library rely on this being -1.  */
#ifndef EOF
# define EOF (-1)
#endif

#define FOPEN_MAX       20
#define FILENAME_MAX    1024
#define L_tmpnam		1024
#ifndef __STRICT_ANSI__
#define P_tmpdir		"/tmp"
#endif

	/* The possibilities for the third argument to `fseek'. These values should not be changed.  */
#define SEEK_SET	0	/* Seek from beginning of file.  */
#define SEEK_CUR	1	/* Seek from current position.  */
#define SEEK_END	2	/* Seek from end of file.  */

#define TMP_MAX         26

	/* Standard streams.  */
	//__constant__ extern FILE *__iob_file[3];
	//#define stdin  (__iob_file[0]) /* Standard input stream.  */
	//#define stdout (__iob_file[1]) /* Standard output stream.  */
	//#define stderr (__iob_file[2]) /* Standard error output stream.  */

	/*
	* Functions defined in ANSI C standard.
	*/

#define __VALIST char*

	//int printf(const char *, ...);
	//int     _EXFUN(scanf, (const char *, ...));
	//int     _EXFUN(sscanf, (const char *, const char *, ...));
	//int     _EXFUN(vfprintf, (FILE *, const char *, __VALIST));
	//int     _EXFUN(vprintf, (const char *, __VALIST));
	//int     _EXFUN(vsprintf, (char *, const char *, __VALIST));
	//__device__ int vsnprintf(const char *buf, size_t bufLen, const char *fmt, va_list *va);
	//int     _EXFUN(fgetc, (FILE *));
	//char *  _EXFUN(fgets, (char *, int, FILE *));
	//int     _EXFUN(fputc, (int, FILE *));
	//int     _EXFUN(fputs, (const char *, FILE *));
	//int     _EXFUN(getc, (FILE *));
	//int     _EXFUN(getchar, (void));
	//char *  _EXFUN(gets, (char *));
	//int     _EXFUN(putc, (int, FILE *));
	//int     _EXFUN(putchar, (int));
	//int     _EXFUN(puts, (const char *));
	//int     _EXFUN(ungetc, (int, FILE *));
	//size_t  _EXFUN(fread, (void *, size_t _size, size_t _n, FILE *));
	//size_t  _EXFUN(fwrite, (const void *, size_t _size, size_t _n, FILE *));

#define sprintf(buf, fmt, ...) snprintf(buf, -1, fmt, __VA_ARGS__)
	//STDARG(int, __snprintf, vsnprintf(buf, bufLen, fmt, &va), const char *buf, size_t bufLen, const char *fmt) 

#ifdef __cplusplus
}
#endif

#endif  /* _INC_STDIO */