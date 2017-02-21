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

//#pragma once
#ifndef _STDIOCU_H
#define _STDIOCU_H
#include <featurescu.h>

#ifndef CORE_MAXFILESTREAM
#define CORE_MAXFILESTREAM 10
#endif

#include <stdio.h>
#if defined(__CUDA_ARCH__) || defined(LIBCUFORCE)
#include <stdargcu.h>

//typedef struct __STDIO_FILE_STRUCT FILE;

__BEGIN_DECLS;

//#include <sys/types.h>
//__BEGIN_NAMESPACE_STD;
///* The opaque type of streams.  This is the definition used elsewhere.  */
////typedef struct __STDIO_FILE_STRUCT FILE;
//__END_NAMESPACE_STD;
//
//#include <bits/libcu_stdio.h>
//
///* The type of the second argument to `fgetpos' and `fsetpos'.  */
//__BEGIN_NAMESPACE_STD;
//#ifndef __USE_FILE_OFFSET64
//typedef __STDIO_fpos_t fpos_t;
//#else
//typedef __STDIO_fpos64_t fpos_t;
//#endif
//__END_NAMESPACE_STD;
//#ifdef __USE_LARGEFILE64
//typedef __STDIO_fpos64_t fpos64_t;
//#endif
//
///* The possibilities for the third argument to `setvbuf'.  */
//#define _IOFBF __STDIO_IOFBF		/* Fully buffered.  */
//#define _IOLBF __STDIO_IOLBF		/* Line buffered.  */
//#define _IONBF __STDIO_IONBF		/* No buffering.  */
//
///* Default buffer size.  */
//#ifndef BUFSIZ
//#define BUFSIZ __STDIO_BUFSIZ
//#endif
//
///* End of file character. Some things throughout the library rely on this being -1.  */
//#ifndef EOF
//#define EOF (-1)
//#endif
//
///* The possibilities for the third argument to `fseek'. These values should not be changed.  */
//#define SEEK_SET	0	/* Seek from beginning of file.  */
//#define SEEK_CUR	1	/* Seek from current position.  */
//#define SEEK_END	2	/* Seek from end of file.  */
//
///* Default path prefix for `mkstemp'.  */
//#define P_tmpdir "/tmp"

#define ISDEVICEFILE(stream) (stream >= __iob_file && stream <= __iob_file + CORE_MAXFILESTREAM)
extern __constant__ FILE __iob_file[CORE_MAXFILESTREAM];
#undef stdin
#undef stdout
#undef stderr
#define stdin  (&__iob_file[0]) /* Standard input stream.  */
#define stdout (&__iob_file[1]) /* Standard output stream.  */
#define stderr (&__iob_file[2]) /* Standard error output stream.  */

__END_DECLS;
#include <sentinel-stdiomsg.h>
__BEGIN_DECLS;

__BEGIN_NAMESPACE_STD;
/* Remove file FILENAME.  */
extern __device__ int remove_(const char *filename);
#define remove remove_
/* Rename file OLD to NEW.  */
extern  __device__ int rename_(const char *old, const char *new_);
#define rename rename_
/* Remove file FILENAME.  */
extern __device__ int _unlink_(const char *filename);
#define _unlink _unlink_
__END_NAMESPACE_STD;

__BEGIN_NAMESPACE_STD;
// mktemp
/* Create a temporary file and open it read/write. */
#ifndef __USE_FILE_OFFSET64
extern __device__ FILE *tmpfile_(void);
#define tmpfile tmpfile_
#else
#define tmpfile tmpfile64
#endif
__END_NAMESPACE_STD;

__BEGIN_NAMESPACE_STD;
/* Close STREAM. */
extern __device__ int fclose_device(FILE *stream);
__forceinline __device__ int fclose_(FILE *stream, bool wait = true) { if (ISDEVICEFILE(stream)) return fclose_device(stream); stdio_fclose msg(wait, stream); return msg.RC; }
#define fclose fclose_
/* Flush STREAM, or all streams if STREAM is NULL. */
extern __device__ int fflush_device(FILE *stream);
__forceinline __device__ int fflush_(FILE *stream) { if (ISDEVICEFILE(stream)) return fflush_device(stream); stdio_fflush msg(false, stream); return msg.RC; }
#define fflush fflush_
__END_NAMESPACE_STD;

__BEGIN_NAMESPACE_STD;
#ifndef __USE_FILE_OFFSET64
/* Open a file, replacing an existing stream with it. */
extern __device__ FILE *freopen_(const char *__restrict filename, const char *__restrict modes, FILE *__restrict stream);
#define freopen freopen_
/* Open a file and create a new stream for it. */
__forceinline __device__ FILE *fopen_(const char *__restrict filename, const char *__restrict modes) { return freopen_(filename, modes, nullptr); }
#define fopen fopen_
#else
#define fopen fopen64
#define freopen freopen64
#endif
__END_NAMESPACE_STD;
#ifdef __USE_LARGEFILE64
/* Open a file, replacing an existing stream with it. */
extern __device__ FILE *freopen64_(const char *__restrict filename, const char *__restrict modes, FILE *__restrict stream);
#define freopen64 freopen64_
/* Open a file and create a new stream for it. */
__forceinline __device__ FILE *fopen64_(const char *__restrict filename, const char *__restrict modes) { return freopen64_(filename, modes, nullptr); }
#define fopen64 fopen64_
#endif

__BEGIN_NAMESPACE_STD;
/* Make STREAM use buffering mode MODE. If BUF is not NULL, use N bytes of it for buffering; else allocate an internal buffer N bytes long.  */
extern __device__ int setvbuf_device(FILE *__restrict stream, char *__restrict buf, int modes, size_t n);
__forceinline __device__ int setvbuf_(FILE *__restrict stream, char *__restrict buf, int modes, size_t n) { if (ISDEVICEFILE(stream)) return setvbuf_device(stream, buf, modes, n); stdio_setvbuf msg(stream, buf, modes, n); return msg.RC; }
#define setvbuf setvbuf_
/* If BUF is NULL, make STREAM unbuffered. Else make it use buffer BUF, of size BUFSIZ.  */
__forceinline __device__ void setbuf_(FILE *__restrict stream, char *__restrict buf) { if (ISDEVICEFILE(stream)) setvbuf_device(stream, buf, buf ? _IOFBF : _IONBF, BUFSIZ); else stdio_setvbuf msg(stream, buf, -1, 0); }
#define setbuf setbuf_
__END_NAMESPACE_STD;

__BEGIN_NAMESPACE_C99;
/* Maximum chars of output to write in MAXLEN.  */
//moved: extern __device__ int snprintf_(char *__restrict s, size_t maxlen, const char *__restrict format, ...);
extern __device__ int vsnprintf_(char *__restrict s, size_t maxlen, const char *__restrict format, va_list va);
#define vsnprintf vsnprintf_
__END_NAMESPACE_C99;

__BEGIN_NAMESPACE_STD;
/* Write formatted output to STREAM. */
//moved: extern __device__ int fprintf(FILE *__restrict stream, const char *__restrict format, ...);
/* Write formatted output to stdout. */
//moved: extern __device__ int printf(const char *__restrict format, ...);
/* Write formatted output to S.  */
//moved: extern __device__ int sprintf(char *__restrict s, const char *__restrict format, ...);

/* Write formatted output to S from argument list ARG. */
extern __device__ int vfprintf_(FILE *__restrict s, const char *__restrict format, va_list va, bool wait = true);
#define vfprintf vfprintf_
/* Write formatted output to stdout from argument list ARG. */
//builtin: __forceinline __device__ int vprintf_(const char *__restrict format, va_list va) { return vfprintf(stdout, format, va, true); };
//#define vprintf vprintf_
/* Write formatted output to S from argument list ARG.  */
__forceinline __device__ int vsprintf_(char *__restrict s, const char *__restrict format, va_list va) { return vsnprintf(s, 0xffffffff, format, va); }
#define vsprintf vsprintf_
__END_NAMESPACE_STD;

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
extern __device__ int vfscanf_(FILE *__restrict s, const char *__restrict format, va_list va, bool wait = true);
#define vfscanf vfscanf_
/* Read formatted input from stdin into argument list ARG. */
__forceinline __device__ int vscanf_(const char *__restrict format, va_list va) { return vfscanf(stdin, format, va, true); }
#define vscanf vscanf_
/* Read formatted input from S into argument list ARG.  */
extern __device__ int vsscanf_(const char *__restrict s, const char *__restrict format, va_list va);
#define vsscanf vsscanf_
__END_NAMESPACE_C99;

__BEGIN_NAMESPACE_STD;
/* Read a character from STREAM.  */
extern __device__ int fgetc_device(FILE *stream);
__forceinline __device__ int fgetc_(FILE *stream) { if (ISDEVICEFILE(stream)) return fgetc_device(stream); stdio_fgetc msg(stream); return msg.RC; }
#define fgetc fgetc_
#define getc(stream) fgetc(stream)
/* Read a character from stdin.  */
__forceinline __device__ int getchar_(void) { return fgetc(stdin); }
#define getchar getchar_
__END_NAMESPACE_STD;

/* The C standard explicitly says this is a macro, so we always do the optimization for it.  */
//sky: #define getc(fp) __GETC(fp)

__BEGIN_NAMESPACE_STD;
/* Write a character to STREAM.  */
extern __device__ int fputc_device(int c, FILE *stream);
__forceinline __device__ int fputc_(int c, FILE *stream, bool wait = true) { if (ISDEVICEFILE(stream)) return fputc_device(c, stream); stdio_fputc msg(wait, c, stream); return msg.RC; }
#define fputc fputc_
#define putc(c, stream) fputc(c, stream)
/* Write a character to stdout.  */
__forceinline __device__ int putchar_(int c) { return fputc(c, stdout); }
#define putchar putchar_
__END_NAMESPACE_STD;

/* The C standard explicitly says this can be a macro, so we always do the optimization for it.  */
//sky: #define putc(ch, fp) __PUTC(ch, fp)

__BEGIN_NAMESPACE_STD;
/* Get a newline-terminated string of finite length from STREAM.  */
extern __device__ char *fgets_device(char *__restrict s, int n, FILE *__restrict stream);
__forceinline __device__ char *fgets_(char *__restrict s, int n, FILE *__restrict stream) { if (ISDEVICEFILE(stream)) return fgets_device(s, n, stream); stdio_fgets msg(s, n, stream); return msg.RC; }
#define fgets fgets_
__END_NAMESPACE_STD;

__BEGIN_NAMESPACE_STD;
/* Write a string to STREAM.  */
extern __device__ int fputs_device(const char *__restrict s, FILE *__restrict stream);
__forceinline __device__ int fputs_(const char *__restrict s, FILE *__restrict stream, bool wait = true) { if (ISDEVICEFILE(stream)) return fputs_device(s, stream); stdio_fputs msg(wait, s, stream); return msg.RC; }
#define fputs fputs_

/* Write a string, followed by a newline, to stdout.  */
//extern __device__ int puts(const char *s);
__forceinline __device__ int puts_(const char *s) { fputs(s, stdout); return fputs("\n", stdout); }
#define puts puts_

/* Push a character back onto the input buffer of STREAM.  */
extern __device__ int ungetc_device(int c, FILE *stream);
__forceinline __device__ int ungetc_(int c, FILE *stream, bool wait = true) { if (ISDEVICEFILE(stream)) return ungetc_device(c, stream); stdio_ungetc msg(wait, c, stream); return msg.RC; }
#define ungetc ungetc_

/* Read chunks of generic data from STREAM.  */
extern __device__ size_t fread_device(void *__restrict ptr, size_t size, size_t n, FILE *__restrict stream);
__forceinline __device__ size_t fread_(void *__restrict ptr, size_t size, size_t n, FILE *__restrict stream, bool wait = true) { if (ISDEVICEFILE(stream)) return fread_device(ptr, size, n, stream); stdio_fread msg(wait, size, n, stream); memcpy(ptr, msg.Ptr, msg.RC); return msg.RC; }
#define fread fread_
/* Write chunks of generic data to STREAM.  */
extern __device__ size_t fwrite_device(const void *__restrict ptr, size_t size, size_t n, FILE *__restrict s);
__forceinline __device__ size_t fwrite_(const void *__restrict ptr, size_t size, size_t n, FILE *__restrict s, bool wait = true) { if (ISDEVICEFILE(s)) return fwrite_device(ptr, size, n, s); stdio_fwrite msg(wait, ptr, size, n, s); return msg.RC; }
#define fwrite fwrite_
__END_NAMESPACE_STD;

__BEGIN_NAMESPACE_STD;
/* Seek to a certain position on STREAM.  */
extern __device__ int fseek_device(FILE *stream, long int off, int whence);
__forceinline __device__ int fseek_(FILE *stream, long int off, int whence) { if (ISDEVICEFILE(stream)) return fseek_device(stream, off, whence); stdio_fseek msg(true, stream, off, whence); return msg.RC; }
#define fseek fseek_
/* Return the current position of STREAM.  */
extern __device__ long int ftell_device(FILE *stream);
__forceinline __device__ long int ftell_(FILE *stream) { if (ISDEVICEFILE(stream)) return ftell_device(stream); stdio_ftell msg(stream); return msg.RC; }
#define ftell ftell_
/* Rewind to the beginning of STREAM.  */
extern __device__ void rewind_device(FILE *stream);
__forceinline __device__ void rewind_(FILE *stream) { if (ISDEVICEFILE(stream)) rewind_device(stream); else stdio_rewind msg(stream); }
#define rewind rewind_
__END_NAMESPACE_STD;

/* The Single Unix Specification, Version 2, specifies an alternative,
more adequate interface for the two functions above which deal with
file offset.  `long int' is not the right type.  These definitions
are originally defined in the Large File Support API.  */
#if defined(__USE_LARGEFILE)
#ifndef __USE_FILE_OFFSET64
/* Seek to a certain position on STREAM.   */
extern __device__ int fseeko_(FILE *stream, __off_t off, int whence);
#define fseeko fseeko_
/* Return the current position of STREAM.  */
extern __device__ __off_t ftello_(FILE *stream);
#define ftello ftello_
#else
#define fseeko fseeko64
#define ftello ftello64
#endif
#endif

__BEGIN_NAMESPACE_STD;
#ifndef __USE_FILE_OFFSET64
/* Get STREAM's position.  */
extern __device__ int fgetpos_device(FILE *__restrict stream, fpos_t *__restrict pos);
__forceinline __device__ int fgetpos_(FILE *__restrict stream, fpos_t *__restrict pos) { if (ISDEVICEFILE(stream)) return fgetpos_device(stream, pos); stdio_fgetpos msg(stream, pos); return msg.RC; }
#define fgetpos fgetpos_
/* Set STREAM's position.  */
extern __device__ int fsetpos_device(FILE *stream, const fpos_t *pos);
__forceinline __device__ int fsetpos_(FILE *stream, const fpos_t *pos) { if (ISDEVICEFILE(stream)) return fsetpos_device(stream, pos); stdio_fsetpos msg(stream, pos); return msg.RC; }
#define fsetpos fsetpos_
#else
#define fgetpos fgetpos64
#define fsetpos fsetpos64
#endif
__END_NAMESPACE_STD;

#ifdef __USE_LARGEFILE64
extern __device__ int fseeko64_(FILE *stream, __off64_t off, int whence);
#define fseeko64 fseeko64_
extern __device__ __off64_t ftello64_(FILE *stream);
#define ftello64 ftello64_
extern __device__ int fgetpos64_(FILE *__restrict stream, fpos64_t *__restrict pos);
#define fgetpos64 fgetpos64_
extern __device__ int fsetpos64_(FILE *stream, const fpos64_t *pos);
#define fsetpos64 fsetpos64_
#endif

__BEGIN_NAMESPACE_STD;
/* Clear the error and EOF indicators for STREAM.  */
extern __device__ void clearerr_device(FILE *stream);
__forceinline __device__ void clearerr_(FILE *stream) { if (ISDEVICEFILE(stream)) clearerr_device(stream); else stdio_clearerr msg(stream); }
#define clearerr clearerr_
/* Return the EOF indicator for STREAM.  */
extern __device__ int feof_device(FILE *stream);
__forceinline __device__ int feof_(FILE *stream) { if (ISDEVICEFILE(stream)) return feof_device(stream); stdio_feof msg(stream); return msg.RC; }
#define feof feof_
/* Return the error indicator for STREAM.  */
extern __device__ int ferror_device(FILE *stream);
__forceinline __device__ int ferror_(FILE *stream) { if (ISDEVICEFILE(stream)) return ferror_device(stream); stdio_ferror msg(stream); return msg.RC; }
#define ferror ferror_
__END_NAMESPACE_STD;

__BEGIN_NAMESPACE_STD;
/* Print a message describing the meaning of the value of errno.  */
//extern __device__ void perror_(const char *s); #define perror perror_
__END_NAMESPACE_STD;

/* Return the system file descriptor for STREAM.  */
extern __device__ int fileno_device(FILE *stream);
__forceinline __device__ int fileno_(FILE *stream) { if (ISDEVICEFILE(stream)) return fileno_device(stream); stdio_fileno msg(stream); return msg.RC; }
#define fileno fileno_

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

__device__ char *vmtagprintf_(void *tag, const char *format, va_list va);
#define vmtagprintf vmtagprintf_
__device__ char *vmprintf_(const char *format, va_list va);
#define vmprintf vmprintf_
__device__ char *vmnprintf_(char *__restrict s, size_t maxlen, const char *format, va_list va);
#define vmnprintf vmnprintf_

__END_DECLS;

__BEGIN_NAMESPACE_STD;
/* Write formatted output to STREAM. */
STDARG(int, fprintf_, vfprintf_(stream, format, va), FILE *__restrict stream, const char *__restrict format);
#define fprintf_ fprintf
//#if !defined(__CUDACC_RTC__)
//#define fprintf(stream, format, ...) fprintf_(stream, format, __VA_ARGS__)
//#endif
/* Write formatted output to stdout. */
//builtin: STDARG(int, printf_, vprintf_(format, va), const char *__restrict format);
//#define printf printf_
/* Write formatted output to S.  */
//STDARG(int, sprintf_, vsprintf_(s, format, va), char *__restrict s, const char *__restrict format);
//STDARG(int, sprintf_, vsprintf_(s, format, va), const char *__restrict s, const char *__restrict format);
#define sprintf(s, format, ...) snprintf_(s, 0xffffffff, format, __VA_ARGS__)
__END_NAMESPACE_STD;

__BEGIN_NAMESPACE_C99;
/* Maximum chars of output to write in MAXLEN.  */
STDARG(int, snprintf_, vsnprintf_(s, maxlen, format, va), char *__restrict s, size_t maxlen, const char *__restrict format);
STDARG(int, snprintf_, vsnprintf_((char *)s, maxlen, format, va), const char *__restrict s, size_t maxlen, const char *__restrict format);
#define snprintf snprintf_
__END_NAMESPACE_C99;

__BEGIN_NAMESPACE_STD;
/* Read formatted input from STREAM.  */
STDARG(int, fscanf_, vfscanf_(stream, format, va), FILE *__restrict stream, const char *__restrict format);
#define fscanf fscanf_
/* Read formatted input from stdin.  */
STDARG(int, scanf_, vscanf_(format, va), const char *__restrict format);
#define scanf scanf_
/* Read formatted input from S.  */
STDARG(int, sscanf_, vsscanf_(s, format, va), const char *__restrict s, const char *__restrict format);
STDARG2(int, sscanf_, vsscanf_(s, format, va), const char *__restrict s, const char *__restrict format);
STDARG3(int, sscanf_, vsscanf_(s, format, va), const char *__restrict s, const char *__restrict format);
#define sscanf sscanf_
__END_NAMESPACE_STD;

#else
#define snprintf _snprintf
#define fprintf_ fprintf
#define vmtagprintf(tag, format, va) format
#define vmprintf(format, va) format
#define vmnprintf(s, maxlen, format, va) nullptr
#endif  /* __CUDA_ARCH__ */

#endif  /* _STDIOCU_H */

