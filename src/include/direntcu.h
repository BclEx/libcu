/*
dirent.h - Directory Entities
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
#ifndef _DIRENTCU_H
#define	_DIRENTCU_H
#define	_DIRENT_H
#include <crtdefscu.h>
#include <featurescu.h>
#include <sys/types.h>

__BEGIN_DECLS;

/* This file defines `struct dirent'.

It defines the macro `_DIRENT_HAVE_D_NAMLEN' iff there is a `d_namlen'
member that gives the length of `d_name'.

It defines the macro `_DIRENT_HAVE_D_RECLEN' iff there is a `d_reclen'
member that gives the size of the entire directory entry.

It defines the macro `_DIRENT_HAVE_D_OFF' iff there is a `d_off'
member that gives the file offset of the next directory entry.

It defines the macro `_DIRENT_HAVE_D_TYPE' iff there is a `d_type'
member that gives the type of the file.
*/
#include <bits/libcu_dirent.h>

/* These macros extract size information from a `struct dirent *'.
They may evaluate their argument multiple times, so it must not
have side effects.  Each of these may involve a relatively costly
call to `strlen' on some systems, so these values should be cached.

_D_EXACT_NAMLEN (DP)	returns the length of DP->d_name, not including
its terminating null character.

_D_ALLOC_NAMLEN (DP)	returns a size at least (_D_EXACT_NAMLEN (DP) + 1);
that is, the allocation size needed to hold the DP->d_name string.
Use this macro when you don't need the exact length, just an upper bound.
This macro is less likely to require calling `strlen' than _D_EXACT_NAMLEN.
*/

#ifdef _DIRENT_HAVE_D_NAMLEN
#define _D_EXACT_NAMLEN(d) ((d)->d_namlen)
#define _D_ALLOC_NAMLEN(d) (_D_EXACT_NAMLEN(d)+1)
#else
#define _D_EXACT_NAMLEN(d) (strlen((d)->d_name))
#ifdef _DIRENT_HAVE_D_RECLEN
# define _D_ALLOC_NAMLEN(d) (((char *)(d) + (d)->d_reclen) - &(d)->d_name[0])
#else
# define _D_ALLOC_NAMLEN(d) (sizeof((d)->d_name) > 1 ? sizeof((d)->d_name) : _D_EXACT_NAMLEN(d)+1)
#endif
#endif

/* This is the data type of directory stream objects. The actual structure is opaque to users.  */
typedef struct __dirstream DIR;

/* Open a directory stream on NAME. Return a DIR stream on the directory, or NULL if it could not be opened. */
extern __device__ DIR *opendir(const char *name);

/* Close the directory stream DIRP. Return 0 if successful, -1 if not.  */
extern __device__ int closedir(DIR *dirp);

/* Read a directory entry from DIRP.  Return a pointer to a `struct dirent' describing the entry, or NULL for EOF or error.  The
storage returned may be overwritten by a later readdir call on the same DIR stream.

If the Large File Support API is selected we have to use the appropriate interface.  */
#ifndef __USE_FILE_OFFSET64
extern __device__ struct dirent *readdir(DIR *dirp);
#else
#define readdir readdir64
#endif
#ifdef __USE_LARGEFILE64
extern __device__ struct dirent64 *readdir64(DIR *dirp);
#endif

/* Rewind DIRP to the beginning of the directory.  */
extern __device__ void rewinddir(DIR *dirp);

__END_DECLS;

#endif  /* _DIRENTCU_H */
#else
//typedef struct __dirstream DIR;
//#include <_dirent.h>
#endif