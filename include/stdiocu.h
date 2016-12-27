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
#include <featurescu.h>
#include <crtdefscu.h>
#include <stdargcu.h>

__BEGIN_DECLS;

#include <sys/types.h>

__BEGIN_NAMESPACE_STD;
/* The opaque type of streams.  This is the definition used elsewhere.  */
typedef struct __STDIO_FILE_STRUCT FILE;
__END_NAMESPACE_STD;

#include <bits/libcu_stdio.h>

__BEGIN_NAMESPACE_STD;
#ifndef __USE_FILE_OFFSET64
typedef __STDIO_fpos_t fpos_t;
#else
typedef __STDIO_fpos64_t fpos_t;
#endif
__END_NAMESPACE_STD;
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

/* End of file character. Some things throughout the library rely on this being -1.  */
#ifndef EOF
#define EOF (-1)
#endif

/* The possibilities for the third argument to `fseek'. These values should not be changed.  */
#define SEEK_SET	0	/* Seek from beginning of file.  */
#define SEEK_CUR	1	/* Seek from current position.  */
#define SEEK_END	2	/* Seek from end of file.  */

/* Default path prefix for `mkstemp'.  */
#define P_tmpdir "/tmp"

/* Standard streams.  */
extern __device__ FILE *stdin;	       /* Standard input stream.  */
extern __device__ FILE *stdout;        /* Standard output stream.  */
extern __device__ FILE *stderr;        /* Standard error output stream.  */
/* C89/C99 say they're macros.  Make them happy.  */
#define stdin stdin
#define stdout stdout
#define stderr stderr

__BEGIN_NAMESPACE_STD;
/* Remove file FILENAME.  */
extern __device__ int remove(const char *filename);
/* Rename file OLD to NEW.  */
extern __device__ int rename(const char *old, const char *new_);
/* Remove file FILENAME.  */
extern __device__ int _unlink(const char *filename);
__END_NAMESPACE_STD;

__BEGIN_NAMESPACE_STD;
/* Create a temporary file and open it read/write. */
#ifndef __USE_FILE_OFFSET64
extern __device__ FILE *tmpfile(void);
#else
#define tmpfile tmpfile64
#endif
__END_NAMESPACE_STD;

__BEGIN_NAMESPACE_STD;
/* Close STREAM. */
extern __device__ int fclose(FILE *stream, bool wait = true);
/* Flush STREAM, or all streams if STREAM is NULL. */
extern __device__ int fflush(FILE *stream);
__END_NAMESPACE_STD;

__BEGIN_NAMESPACE_STD;
#ifndef __USE_FILE_OFFSET64
/* Open a file and create a new stream for it. */
extern __device__ FILE *fopen(const char *__restrict filename, const char *__restrict modes);
/* Open a file, replacing an existing stream with it. */
extern __device__ FILE *freopen(const char *__restrict filename, const char *__restrict modes, FILE *__restrict stream);
#else
#define fopen fopen64
#define freopen freopen64
#endif
__END_NAMESPACE_STD;
#ifdef __USE_LARGEFILE64
extern __device__ FILE *fopen64(const char *__restrict filename, const char *__restrict modes);
extern __device__ FILE *freopen64(const char *__restrict filename, const char *__restrict modes, FILE *__restrict stream);
#endif

__BEGIN_NAMESPACE_STD;
/* If BUF is NULL, make STREAM unbuffered. Else make it use buffer BUF, of size BUFSIZ.  */
extern __device__ void setbuf(FILE *__restrict stream, char *__restrict buf);
/* Make STREAM use buffering mode MODE. If BUF is not NULL, use N bytes of it for buffering; else allocate an internal buffer N bytes long.  */
extern __device__ int setvbuf(FILE *__restrict stream, char *__restrict buf, int modes, size_t n);
__END_NAMESPACE_STD;

__BEGIN_NAMESPACE_STD;
/* Write formatted output to STREAM. */
//moved: extern __device__ int fprintf(FILE *__restrict stream, const char *__restrict format, ...);
/* Write formatted output to stdout. */
//moved: extern __device__ int printf(const char *__restrict format, ...);
/* Write formatted output to S.  */
//moved: extern __device__ int sprintf(char *__restrict s, const char *__restrict format, ...);

/* Write formatted output to S from argument list ARG. */
extern __device__ int vfprintf(FILE *__restrict s, const char *__restrict format, va_list arg);
/* Write formatted output to stdout from argument list ARG. */
extern __device__ int vprintf(const char *__restrict format, va_list arg);
/* Write formatted output to S from argument list ARG.  */
extern __device__ int vsprintf(char *__restrict s, const char *__restrict format, va_list arg);
__END_NAMESPACE_STD;

__BEGIN_NAMESPACE_C99;
/* Maximum chars of output to write in MAXLEN.  */
//moved: extern __device__ int snprintf(char *__restrict s, size_t maxlen, const char *__restrict format, ...);
extern __device__ int vsnprintf(char *__restrict s, size_t maxlen, const char *__restrict format, va_list arg);
__END_NAMESPACE_C99;

__BEGIN_NAMESPACE_STD;
/* Read formatted input from STREAM.  */
//moved: extern __device__ int fscanf(FILE *__restrict stream, const char *__restrict format, ...);
/* Read formatted input from stdin.  */
//moved: extern __device__ int scanf(const char *__restrict format, ...);
/* Read formatted input from S.  */
//moved: extern __device__ int sscanf(const char *__restrict s, const char *__restrict format, ...);
__END_NAMESPACE_STD;

__BEGIN_NAMESPACE_C99;
/* Read formatted input from S into argument list ARG.  */
extern __device__ int vfscanf(FILE *__restrict s, const char *__restrict format, va_list arg);
/* Read formatted input from stdin into argument list ARG. */
extern __device__ int vscanf(const char *__restrict format, va_list arg);
/* Read formatted input from S into argument list ARG.  */
extern __device__ int vsscanf(const char *__restrict s, const char *__restrict format, va_list arg);
__END_NAMESPACE_C99;

__BEGIN_NAMESPACE_STD;
/* Read a character from STREAM.  */
extern __device__ int fgetc(FILE *stream);
extern __device__ int getc(FILE *stream);

/* Read a character from stdin.  */
extern __device__ int getchar(void);
__END_NAMESPACE_STD;

/* The C standard explicitly says this is a macro, so we always do the optimization for it.  */
//sky: #define getc(fp) __GETC(fp)

__BEGIN_NAMESPACE_STD;
/* Write a character to STREAM.  */
extern __device__ int fputc(int c, FILE *stream, bool wait = true);
extern __device__ int putc(int c, FILE *stream);

/* Write a character to stdout.  */
extern __device__ int putchar(int c);
__END_NAMESPACE_STD;

/* The C standard explicitly says this can be a macro, so we always do the optimization for it.  */
//sky: #define putc(ch, fp) __PUTC(ch, fp)

__BEGIN_NAMESPACE_STD;
/* Get a newline-terminated string of finite length from STREAM.  */
extern __device__ char *fgets(char *__restrict s, int n, FILE *__restrict stream);
__END_NAMESPACE_STD;

__BEGIN_NAMESPACE_STD;
/* Write a string to STREAM.  */
extern __device__ int fputs(const char *__restrict s, FILE *__restrict stream, bool wait = true);

/* Write a string, followed by a newline, to stdout.  */
extern __device__ int puts(const char *s);

/* Push a character back onto the input buffer of STREAM.  */
extern __device__ int ungetc(int c, FILE *stream);

/* Read chunks of generic data from STREAM.  */
extern __device__ size_t fread(void *__restrict ptr, size_t size, size_t n, FILE *__restrict stream, bool wait = true);
/* Write chunks of generic data to STREAM.  */
extern __device__ size_t fwrite(const void *__restrict ptr, size_t size, size_t n, FILE *__restrict s, bool wait = true);
__END_NAMESPACE_STD;

__BEGIN_NAMESPACE_STD;
/* Seek to a certain position on STREAM.  */
extern __device__ int fseek(FILE *stream, long int off, int whence);
/* Return the current position of STREAM.  */
extern __device__ long int ftell(FILE *stream);
/* Rewind to the beginning of STREAM.  */
extern __device__ void rewind(FILE *stream);
__END_NAMESPACE_STD;

/* The Single Unix Specification, Version 2, specifies an alternative,
more adequate interface for the two functions above which deal with
file offset.  `long int' is not the right type.  These definitions
are originally defined in the Large File Support API.  */
#if defined(__USE_LARGEFILE)
#ifndef __USE_FILE_OFFSET64
/* Seek to a certain position on STREAM.   */
extern __device__ int fseeko(FILE *stream, __off_t off, int whence);
/* Return the current position of STREAM.  */
extern __device__ __off_t ftello(FILE *stream);
#else
#define fseeko fseeko64
#define ftello ftello64
#endif
#endif

__BEGIN_NAMESPACE_STD;
#ifndef __USE_FILE_OFFSET64
/* Get STREAM's position.  */
extern __device__ int fgetpos(FILE *__restrict stream, fpos_t *__restrict pos);
/* Set STREAM's position.  */
extern __device__ int fsetpos(FILE *stream, const fpos_t *pos);
#else
#define fgetpos fgetpos64
#define fsetpos fsetpos64
#endif
__END_NAMESPACE_STD;

#ifdef __USE_LARGEFILE64
extern __device__ int fseeko64(FILE *stream, __off64_t off, int whence);
extern __device__ __off64_t ftello64(FILE *stream);
extern __device__ int fgetpos64(FILE *__restrict stream, fpos64_t *__restrict pos);
extern __device__ int fsetpos64(FILE *stream, const fpos64_t *pos);
#endif

__BEGIN_NAMESPACE_STD;
/* Clear the error and EOF indicators for STREAM.  */
extern __device__ void clearerr(FILE *stream);
/* Return the EOF indicator for STREAM.  */
extern __device__ int feof(FILE *stream);
/* Return the error indicator for STREAM.  */
extern __device__ int ferror(FILE *stream);
__END_NAMESPACE_STD;

__BEGIN_NAMESPACE_STD;
/* Print a message describing the meaning of the value of errno.  */
extern __device__ void perror(const char *s);
__END_NAMESPACE_STD;

/* Return the system file descriptor for STREAM.  */
extern __device__ int _fileno(FILE *stream);

/* If we are compiling with optimizing read this file.  It contains
several optimizing inline functions and macros.  */
#ifdef __LIBCU__
#define fgetc(fp)                   __FGETC(fp)
#define fputc(ch, fp)				__FPUTC(ch, fp)
#define getchar()                   __GETC(__stdin)
#define putchar(ch)                 __PUTC((ch), __stdout)
/* Clear the error and EOF indicators for STREAM.  */
#define clearerr(fp)                __CLEARERR(fp)
#define feof(fp)                    __FEOF(fp)
#define ferror(fp)                  __FERROR(fp)
#endif

_Check_return_opt_ _CRTIMP int __cdecl printf(_In_z_ _Printf_format_string_ const char *_Format, ...);

__END_DECLS;

__BEGIN_NAMESPACE_STD;
/* Write formatted output to STREAM. */
STDARG(int, fprintf, vfprintf(stream, format, va), FILE *__restrict stream, const char *__restrict format);
/* Write formatted output to stdout. */
//builtin: STDARG(int, printf, vprintf(format, va), const char *__restrict format);
/* Write formatted output to S.  */
STDARG(int, sprintf, vsprintf(s, format, va), char *__restrict s, const char *__restrict format);
//macro: #define sprintf(s, format, ...) snprintf(s, -1, format, __VA_ARGS__)
__END_NAMESPACE_STD;

__BEGIN_NAMESPACE_C99;
/* Maximum chars of output to write in MAXLEN.  */
STDARG(int, snprintf, vsnprintf(s, maxlen, format, va), char *__restrict s, size_t maxlen, const char *__restrict format);
__END_NAMESPACE_C99;

__BEGIN_NAMESPACE_STD;
/* Read formatted input from STREAM.  */
STDARG(int, fscanf, vfscanf(stream, format, va), FILE *__restrict stream, const char *__restrict format);
/* Read formatted input from stdin.  */
STDARG(int, scanf, vscanf(format, va), const char *__restrict format);
/* Read formatted input from S.  */
STDARG(int, sscanf, vsscanf(s, format, va), const char *__restrict s, const char *__restrict format);
__END_NAMESPACE_STD;


#endif  /* _INC_STDIO */