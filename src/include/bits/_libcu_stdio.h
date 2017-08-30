/*
libcu_stdio.h - define file helpers
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

#ifndef _LIBCU_STDIO_H
#define _LIBCU_STDIO_H

//#define __STDIO_BUFFERS
/* ANSI/ISO mandate at least 256. */
#if defined(HAS_STDIO_BUFSIZ_NONE__)
/* Fake this because some apps use stdio.h BUFSIZ. */
#define __STDIO_BUFSIZ			256
#undef __STDIO_BUFFERS
#elif defined(HAS_STDIO_BUFSIZ_256__)
#define __STDIO_BUFSIZ			256
#elif defined(HAS_STDIO_BUFSIZ_512__)
#define __STDIO_BUFSIZ			512
#elif defined(HAS_STDIO_BUFSIZ_1024__)
#define __STDIO_BUFSIZ		   1024
#elif defined(HAS_STDIO_BUFSIZ_2048__)
#define __STDIO_BUFSIZ		   2048
#elif defined(HAS_STDIO_BUFSIZ_4096__)
#define __STDIO_BUFSIZ		   4096
#elif defined(HAS_STDIO_BUFSIZ_8192__)
#define __STDIO_BUFSIZ		   8192
#else
#error config seems to be out of sync regarding bufsiz options
#endif

#ifdef HAS_STDIO_BUFSIZ_NONE__
#define __STDIO_BUILTIN_BUF_SIZE		0
#else
#if defined(HAS_STDIO_BUILTIN_BUFFER_NONE__)
#define __STDIO_BUILTIN_BUF_SIZE		0
#elif defined(HAS_STDIO_BUILTIN_BUFFER_4__)
#define __STDIO_BUILTIN_BUF_SIZE		4
#elif defined(HAS_STDIO_BUILTIN_BUFFER_8__)
#define __STDIO_BUILTIN_BUF_SIZE		8
#else
#error config seems to be out of sync regarding builtin buffer size
#endif
#endif

#ifdef HAS_STDIO_GETC_MACRO__
#define __STDIO_GETC_MACRO
#endif

#ifdef HAS_STDIO_PUTC_MACRO__
#define __STDIO_PUTC_MACRO
#endif

#define __STDIO_IOFBF 0		/* Fully buffered.  */
#define __STDIO_IOLBF 1		/* Line buffered.  */
#define __STDIO_IONBF 2		/* No buffering.  */

typedef struct {
	_off_t __pos;
#ifdef __STDIO_MBSTATE
	__mbstate_t __mbstate;
#endif
#ifdef HAS_WCHAR__
	int __mblen_pending;
#endif
} __STDIO_fpos_t;

#ifdef HAS_LFS__
typedef struct {
	__off64_t __pos;
#ifdef __STDIO_MBSTATE
	__mbstate_t __mbstate;
#endif
#ifdef HAS_WCHAR__
	int __mblen_pending;
#endif
} __STDIO_fpos64_t;
#endif

#ifdef HAS_LFS__
typedef _off64_t __offmax_t;
#else
typedef _off_t __offmax_t;
#endif

struct __STDIO_FILE_STRUCT {
	unsigned short __modeflags;
#ifdef HAS_WCHAR__
	unsigned char __ungot_width[2]; /* 0: current (building) char; 1: scanf */
#else
	unsigned char __ungot[2];
#endif
	int __filedes;
#ifdef __STDIO_BUFFERS
	unsigned char *__bufstart;	/* pointer to buffer */
	unsigned char *__bufend;	/* pointer to 1 past end of buffer */
	unsigned char *__bufpos;
	unsigned char *__bufread;	/* pointer to 1 past last buffered read char */
#ifdef __STDIO_GETC_MACRO
	unsigned char *__bufgetc_u;	/* 1 past last readable by getc_unlocked */
#endif
#ifdef __STDIO_PUTC_MACRO
	unsigned char *__bufputc_u;	/* 1 past last writeable by putc_unlocked */
#endif
#endif

#ifdef HAS_WCHAR__
	wchar_t __ungot[2];
#endif
#ifdef __STDIO_MBSTATE
	__mbstate_t __state;
#endif
#if __STDIO_BUILTIN_BUF_SIZE > 0
	unsigned char __builtinbuf[__STDIO_BUILTIN_BUF_SIZE];
#endif
};

/* Having ungotten characters implies the stream is reading.
* The scheme used here treats the least significant 2 bits of the stream's modeflags member as follows:
*   0 0   Not currently reading.
*   0 1   Reading, but no ungetc() or scanf() push back chars.
*   1 0   Reading with one ungetc() char (ungot[1] is 1)
*         or one scanf() pushed back char (ungot[1] is 0).
*   1 1   Reading with both an ungetc() char and a scanf()
*         pushed back char.  Note that this must be the result
*         of a scanf() push back (in ungot[0]) _followed_ by
*         an ungetc() call (in ungot[1]).
*
* Notes:
*   scanf() can NOT use ungetc() to push back characters.
*     (See section 7.19.6.2 of the C9X rationale -- WG14/N897.)
*/

#define __MASK_READING		0x0003U /* (0x0001 | 0x0002) */
#define __FLAG_READING		0x0001U
#define __FLAG_UNGOT		0x0002U
#define __FLAG_EOF			0x0004U
#define __FLAG_ERROR		0x0008U
#define __FLAG_WRITEONLY	0x0010U
#define __FLAG_READONLY		0x0020U /* (__FLAG_WRITEONLY << 1) */
#define __FLAG_WRITING		0x0040U
#define __FLAG_NARROW		0x0080U

#define __FLAG_FBF			0x0000U /* must be 0 */
#define __FLAG_LBF			0x0100U
#define __FLAG_NBF			0x0200U /* (__FLAG_LBF << 1) */
#define __MASK_BUFMODE		0x0300U /* (__FLAG_LBF|__FLAG_NBF) */
#define __FLAG_APPEND		0x0400U /* fixed! == O_APPEND for linux */
#define __FLAG_WIDE			0x0800U
/* available slot			0x1000U */
#define __FLAG_FREEFILE		0x2000U
#define __FLAG_FREEBUF		0x4000U
#define __FLAG_LARGEFILE	0x8000U /* fixed! == 0_LARGEFILE for linux */
#define __FLAG_FAILED_FREOPEN	__FLAG_LARGEFILE

/* Note: In no-buffer mode, it would be possible to pack the necessary flags into one byte.  Since we wouldn't be buffering and there would
* be no support for wchar, the only flags we would need would be:
*   2 bits : ungot count
*   2 bits : eof + error
*   2 bits : readonly + writeonly
*   1 bit  : freefile
*   1 bit  : appending
* So, for a very small system (< 128 files) we might have a 4-byte FILE struct of:
*   unsigned char flags;
*   signed char filedes;
*   unsigned char ungot[2];
*/

#define __CLEARERR_UNLOCKED(stream) \
	((void)((stream)->__modeflags &= ~(__FLAG_EOF|__FLAG_ERROR)))
#define __FEOF_UNLOCKED(stream) ((stream)->__modeflags & __FLAG_EOF)
#define __FERROR_UNLOCKED(stream) ((stream)->__modeflags & __FLAG_ERROR)

#define __CLEARERR(stream) __CLEARERR_UNLOCKED(stream)
#define __FERROR(stream) __FERROR_UNLOCKED(stream)
#define __FEOF(stream) __FEOF_UNLOCKED(stream)

extern __device__ int __fgetc_unlocked(FILE *stream);
extern __device__ int __fputc_unlocked(int c, FILE *stream);

/* First define the default definitions. They are overridden below as necessary. */
#define __FGETC_UNLOCKED(stream) (__fgetc_unlocked)((stream))
#define __FGETC(stream) (fgetc)((stream))
#define __GETC_UNLOCKED_MACRO(stream) (__fgetc_unlocked)((stream))
#define __GETC_UNLOCKED(stream) (__fgetc_unlocked)((stream))
#define __GETC(stream) (fgetc)((stream))

#define __FPUTC_UNLOCKED(c, stream) (__fputc_unlocked)((c), (stream))
#define __FPUTC(c, stream) (fputc)((c), (stream))
#define __PUTC_UNLOCKED_MACRO(c, stream) (__fputc_unlocked)((c), (stream))
#define __PUTC_UNLOCKED(c, stream) (__fputc_unlocked)((c), (stream))
#define __PUTC(c, stream) (fputc)((c), (stream))

#ifdef __STDIO_GETC_MACRO
extern FILE *__stdin; /* For getchar() macro. */
#undef  __GETC_UNLOCKED_MACRO
#define __GETC_UNLOCKED_MACRO(stream) \
	(((stream)->__bufpos < (stream)->__bufgetc_u) ? (*(stream)->__bufpos++) : __fgetc_unlocked(stream))
#if 0
/* Classic macro approach.  getc{_unlocked} can have side effects. */
#undef  __GETC_UNLOCKED
#define __GETC_UNLOCKED(stream) GETC_UNLOCKED_MACRO((__stream))
#else
/* Using gcc extension for safety and additional inlining. */
#undef  __FGETC_UNLOCKED
#define __FGETC_UNLOCKED(stream) \
	(__extension__ ({ FILE *__S = (stream); __GETC_UNLOCKED_MACRO(__S);}) )
#undef  __GETC_UNLOCKED
#define __GETC_UNLOCKED(stream) __FGETC_UNLOCKED((stream))
#undef  __FGETC
#define __FGETC(stream) __FGETC_UNLOCKED((stream))
#undef  __GETC
#define __GETC(stream) __FGETC_UNLOCKED((stream))
#endif
#else
#define __stdin stdin
#endif /* __STDIO_GETC_MACRO */

#ifdef __STDIO_PUTC_MACRO
extern FILE *__stdout; /* For putchar() macro. */
#undef  __PUTC_UNLOCKED_MACRO
#define __PUTC_UNLOCKED_MACRO(c, stream) \
	(((stream)->__bufpos < (stream)->__bufputc_u) ? (*(stream)->__bufpos++) = (c) : __fputc_unlocked((c), (stream)))
#if 0
/* Classic macro approach.  putc{_unlocked} can have side effects.*/
#undef  __PUTC_UNLOCKED
#define __PUTC_UNLOCKED(c, stream) __PUTC_UNLOCKED_MACRO((c), (stream))
#else
/* Using gcc extension for safety and additional inlining. */
#undef  __FPUTC_UNLOCKED
#define __FPUTC_UNLOCKED(c, stream) \
	(__extension__ ({ FILE *__S = (stream); __PUTC_UNLOCKED_MACRO((c), __S); }))
#undef  __PUTC_UNLOCKED
#define __PUTC_UNLOCKED(c, stream) __FPUTC_UNLOCKED((c), (stream))
#undef  __FPUTC
#define __FPUTC(c, stream) __FPUTC_UNLOCKED((c), (stream))
#undef  __PUTC
#define __PUTC(c, stream) __FPUTC_UNLOCKED((c), (stream))
#endif
#else
#define __stdout stdout
#endif /* __STDIO_PUTC_MACRO */

#endif /* _LIBCU_STDIO_H */