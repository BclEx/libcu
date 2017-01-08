/*
libcu_dirent.h - dirent helpers
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

#ifndef _LIBCU_DIRENT_H
#define _LIBCU_DIRENT_H

struct dirent
{
#ifndef __USE_FILE_OFFSET64
	_ino_t d_ino;
	_off_t d_off;
#else
	__ino64_t d_ino;
	__off64_t d_off;
#endif
	unsigned short int d_reclen;
	unsigned char d_type;
	char d_name[256];		/* We must not include limits.h! */
};

#ifdef __USE_LARGEFILE64
struct dirent64
{
	__ino64_t d_ino;
	__off64_t d_off;
	unsigned short int d_reclen;
	unsigned char d_type;
	char d_name[256];		/* We must not include limits.h! */
};
#endif

#undef  _DIRENT_HAVE_D_NAMLEN
#define _DIRENT_HAVE_D_RECLEN
#define _DIRENT_HAVE_D_OFF
#define _DIRENT_HAVE_D_TYPE

#endif /* _LIBCU_DIRENT_H */