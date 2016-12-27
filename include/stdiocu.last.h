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

#if !__CUDACC__
#include <stdio.h>
#elif !defined(_INC_STDIO)
#define _INC_STDIO
#include <features.h>
#include <crtdefscu.h>
#include <stdargcu.h>
//#include <stddefcu.h>


#ifdef  __cplusplus
extern "C" {
#endif

#ifndef __USE_FILE_OFFSET64
	typedef __STDIO_fpos_t fpos_t;
#else
	typedef __STDIO_fpos64_t fpos_t;
#endif
#ifdef __USE_LARGEFILE64
	typedef __STDIO_fpos64_t fpos64_t;
#endif

	/* The possibilities for the third argument to `setvbuf'.  */
#define _IOFBF __STDIO_IOFBF		/* Fully buffered.  */
#define _IOLBF __STDIO_IOLBF		/* Line buffered.  */
#define _IONBF __STDIO_IONBF		/* No buffering.  */

	/* Default buffer size.  */
#ifndef BUFSIZ
#define BUFSIZ __STDIO_BUFSIZ
#endif


	/* End of file character.
	Some things throughout the library rely on this being -1.  */
#ifndef EOF
#define EOF (-1)
#endif

	/* The possibilities for the third argument to `fseek'.
   These values should not be changed.  */
#define SEEK_SET	0	/* Seek from beginning of file.  */
#define SEEK_CUR	1	/* Seek from current position.  */
#define SEEK_END	2	/* Seek from end of file.  */





	//#define _FSTDIO			// function stdio

	// <reent.h> defines __sFILE, _fpos_t.
	// They must be defined there because struct _reent needs them (and we don't want reent.h to include this file.
	struct __sFile {
		int unused;
	};
	typedef struct __sFILE FILE;

#define _IOFBF 0	/* setvbuf should set fully buffered */
#define _IOLBF 1	/* setvbuf should set line buffered */
#define _IONBF 2	/* setvbuf should set unbuffered */

#ifndef NULL
#define NULL 0
#endif

#define BUFSIZ			1024

	/* End of file character. Some things throughout the library rely on this being -1.  */
#ifndef EOF
#define EOF (-1)
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
	__constant__ extern FILE *__iob_file[3];
#define stdin  (__iob_file[0]) /* Standard input stream.  */
#define stdout (__iob_file[1]) /* Standard output stream.  */
#define stderr (__iob_file[2]) /* Standard error output stream.  */

	/* Functions defined in ANSI C standard. */
	_Check_return_opt_ _CRTIMP int __cdecl printf(_In_z_ _Printf_format_string_ const char * _Format, ...);
#define sprintf(buf, fmt, ...) snprintf(buf, -1, fmt, __VA_ARGS__)
#if 1
	__device__ int vscanf(const char *str, va_list *va);
	__device__ int vsscanf(const char *str, const char *fmt, va_list *va);
	__device__ int vfprintf(FILE *f, const char *fmt, va_list *va, bool wait = true);
	__device__ int vprintf(const char *fmt, va_list *va);
	__device__ int vsprintf(char *buf, const char *fmt, va_list *va);
	__device__ int vsnprintf(const char *buf, size_t bufLen, const char *fmt, va_list *va);
	__device__ int fgetc(FILE *f);
	__device__ char *fgets(char *s, int n, FILE *f);
	__device__ int fputc(int c, FILE *f, bool wait = true);
	__device__ int fputs(const char *s, FILE *f, bool wait = true);
	__device__ int getc(FILE *f);
	__device__ int getchar(void);
	__device__ char *gets(char *s);
	__device__ int putc(int c, FILE *f, bool wait = true);
	__device__ int putchar(int c);
	__device__ int puts(const char *s, bool wait = true);
	__device__ int ungetc(int c, FILE *f, bool wait = true);
	__device__ size_t fread(void *p, size_t size, size_t n, FILE *f, bool wait = true);
	__device__ size_t fwrite(const void *p, size_t size, size_t n, FILE *f, bool wait = true);
#else
#define fprintf(f, ...) printf(__VA_ARGS__)
#define fgetc(f) (int)0
#define fgets(s, n, f) (int)0
#define fputc(c, f) printf("%c", c)
#define fputs(s, f) printf(s)
#define puts(s) printf("%s\n", s)
#define fread(p, s, n, f) (size_t)0
#define fwrite(p, s, n, f) (size_t)0
#endif


	//#define fileno(f) (int)(f == _stdin ? 0 : f == _stdout ? 1 : f == _stderr ? 2 : -1)
	//#define setvbuf(f, b, m, s) (int)0
	//#define fopen(f, m) (FILE *)0
	//#define fflush(f) (int)0
	//#define fclose(f) (int)0
	//#define fseek(f, o, s) (int)0
	//#define ftell(f) (int)0
	//#define feof(f) (int)0
	//#define ferror(f) (int)0
	//#define clearerr(f) (void)0
	//#define rename(a, b) (int)0
	//#define unlink(a) (int)0
	//#define close(a) (int)0
	//#define system(c) (int)0









#ifdef __cplusplus
}
#endif

STDARG(int, scanf, vscanf(str, &va), const char *str);
STDARG(int, sscanf, vsscanf(str, fmt, &va), const char *str, const char *fmt);
STDARG(int, snprintf, vsnprintf(buf, bufLen, fmt, &va), const char *buf, size_t bufLen, const char *fmt);

#endif  /* _INC_STDIO */