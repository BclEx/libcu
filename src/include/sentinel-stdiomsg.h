/*
sentinel-stdiomsg.h - messages for sentinel
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

#if !defined(_INC_SENTINEL_STDIOMSG)
#define _INC_SENTINEL_STDIOMSG
#include <sentinel.h>
#include <stringcu.h>

enum {
	STDIO_REMOVE = 1,
	STDIO_RENAME,
	STDIO_UNLINK,
	STDIO_FCLOSE,
	STDIO_FFLUSH,
	STDIO_FREOPEN,
	STDIO_SETVBUF,
	STDIO_FGETC,
	STDIO_FPUTC,
	STDIO_FGETS,
	STDIO_FPUTS,
	STDIO_UNGETC,
	STDIO_FREAD,
	STDIO_FWRITE,
	STDIO_FSEEK,
	STDIO_FTELL,
	STDIO_REWIND,
	STDIO_FGETPOS,
	STDIO_FSETPOS,
	STDIO_CLEARERR,
	STDIO_FEOF,
	STDIO_FERROR,
	STDIO_FILENO,
};

struct stdio_remove
{
	static __forceinline __device__ char *Prepare(stdio_remove *t, char *data, char *dataEnd)
	{
		int strLength = (t->Str ? (int)strlen(t->Str) + 1 : 0);
		char *str = (char *)(data += _ROUND8(sizeof(*t)));
		char *end = (char *)(data += strLength);
		if (end > dataEnd) return nullptr;
		memcpy(str, t->Str, strLength);
		t->Str = str;
		return end;
	}
	sentinelMessage Base;
	const char *Str;
	__device__ stdio_remove(const char *str)
		: Base(true, STDIO_UNLINK, 1024, SENTINELPREPARE(Prepare)), Str(str) { sentinelSend(this, sizeof(stdio_remove)); }
	int RC;
};

struct stdio_rename
{
	static __forceinline __device__ char *Prepare(stdio_rename *t, char *data, char *dataEnd)
	{
		int oldnameLength = (t->Oldname ? (int)strlen(t->Oldname) + 1 : 0);
		int newnameLength = (t->Newname ? (int)strlen(t->Newname) + 1 : 0);
		char *oldname = (char *)(data += _ROUND8(sizeof(*t)));
		char *newname = (char *)(data += oldnameLength);
		char *end = (char *)(data += newnameLength);
		if (end > dataEnd) return nullptr;
		memcpy(oldname, t->Oldname, oldnameLength);
		memcpy(newname, t->Newname, newnameLength);
		t->Oldname = oldname;
		t->Newname = newname;
		return end;
	}
	sentinelMessage Base;
	const char *Oldname; const char *Newname;
	__device__ stdio_rename(const char *oldname, const char *newname)
		: Base(true, STDIO_RENAME, 1024, SENTINELPREPARE(Prepare)), Oldname(oldname), Newname(newname) { sentinelSend(this, sizeof(stdio_rename)); }
	int RC;
};

struct stdio_unlink
{
	static __forceinline __device__ char *Prepare(stdio_unlink *t, char *data, char *dataEnd)
	{
		int strLength = (t->Str ? (int)strlen(t->Str) + 1 : 0);
		char *str = (char *)(data += _ROUND8(sizeof(*t)));
		char *end = (char *)(data += strLength);
		if (end > dataEnd) return nullptr;
		memcpy(str, t->Str, strLength);
		t->Str = str;
		return end;
	}
	sentinelMessage Base;
	const char *Str;
	__device__ stdio_unlink(const char *str)
		: Base(true, STDIO_UNLINK, 1024, SENTINELPREPARE(Prepare)), Str(str) { sentinelSend(this, sizeof(stdio_unlink)); }
	int RC;
};

struct stdio_fclose
{
	sentinelMessage Base;
	FILE *File;
	__device__ stdio_fclose(bool wait, FILE *file)
		: Base(wait, STDIO_FCLOSE, 0, nullptr), File(file) { sentinelSend(this, sizeof(stdio_fclose)); }
	int RC;
};

struct stdio_fflush
{
	sentinelMessage Base;
	FILE *File;
	__device__ stdio_fflush(bool wait, FILE *file)
		: Base(wait, STDIO_FFLUSH, 0, nullptr), File(file) { sentinelSend(this, sizeof(stdio_fflush)); }
	int RC;
};

struct stdio_freopen
{
	static __forceinline __device__ char *Prepare(stdio_freopen *t, char *data, char *dataEnd)
	{
		int filenameLength = (t->Filename ? (int)strlen(t->Filename) + 1 : 0);
		int modeLength = (t->Mode ? (int)strlen(t->Mode) + 1 : 0);
		char *filename = (char *)(data += _ROUND8(sizeof(*t)));
		char *mode = (char *)(data += filenameLength);
		char *end = (char *)(data += modeLength);
		if (end > dataEnd) return nullptr;
		memcpy(filename, t->Filename, filenameLength);
		memcpy(mode, t->Mode, modeLength);
		t->Filename = filename;
		t->Mode = mode;
		return end;
	}
	sentinelMessage Base;
	const char *Filename; const char *Mode; FILE *Stream;
	__device__ stdio_freopen(const char *filename, const char *mode, FILE *stream)
		: Base(true, STDIO_FREOPEN, 1024, SENTINELPREPARE(Prepare)), Filename(filename), Mode(mode), Stream(stream) { sentinelSend(this, sizeof(stdio_freopen)); }
	FILE *RC;
};

struct stdio_setvbuf
{
	static __forceinline __device__ char *Prepare(stdio_setvbuf *t, char *data, char *dataEnd)
	{
		int bufferLength = (t->Buffer ? (int)strlen(t->Buffer) + 1 : 0);
		char *buffer = (char *)(data += _ROUND8(sizeof(*t)));
		char *end = (char *)(data += bufferLength);
		if (end > dataEnd) return nullptr;
		memcpy(buffer, t->Buffer, bufferLength);
		t->Buffer = buffer;
		return end;
	}
	sentinelMessage Base;
	FILE *File; char *Buffer; int Mode; size_t Size;
	__device__ stdio_setvbuf(FILE *file, char *buffer, int mode, size_t size)
		: Base(true, STDIO_SETVBUF, 1024, SENTINELPREPARE(Prepare)), File(file), Buffer(buffer), Mode(mode), Size(size) { sentinelSend(this, sizeof(stdio_setvbuf)); }
	int RC;
};

struct stdio_fgetc
{
	sentinelMessage Base;
	FILE *File;
	__device__ stdio_fgetc(FILE *file)
		: Base(true, STDIO_FGETC, 0, nullptr), File(file) { sentinelSend(this, sizeof(stdio_fgetc)); }
	int RC;
};

struct stdio_fputc
{
	sentinelMessage Base;
	int Ch; FILE *File;
	__device__ stdio_fputc(bool wait, int ch, FILE *file)
		: Base(wait, STDIO_FPUTC, 0, nullptr), Ch(ch), File(file) { sentinelSend(this, sizeof(stdio_fputc)); }
	int RC;
};

struct stdio_fgets
{
	static __forceinline __device__ char *Prepare(stdio_fgets *t, char *data, char *dataEnd)
	{
		t->Str = (char *)(data += _ROUND8(sizeof(*t)));
		char *end = (char *)(data += 1024);
		if (end > dataEnd) return nullptr;
		return end;
	}
	sentinelMessage Base;
	int Num; FILE *File;
	__device__ stdio_fgets(char *str, int num, FILE *file)
		: Base(true, STDIO_FGETS, 1024, SENTINELPREPARE(Prepare)), Str(str), Num(num), File(file) { sentinelSend(this, sizeof(stdio_fgets)); }
	char *Str; 
	char *RC;
};

struct stdio_fputs
{
	static __forceinline __device__ char *Prepare(stdio_fputs *t, char *data, char *dataEnd)
	{
		int strLength = (t->Str ? (int)strlen(t->Str) + 1 : 0);
		char *str = (char *)(data += _ROUND8(sizeof(*t)));
		char *end = (char *)(data += strLength);
		if (end > dataEnd) return nullptr;
		memcpy(str, t->Str, strLength);
		t->Str = str;
		return end;
	}
	sentinelMessage Base;
	const char *Str; FILE *File;
	__device__ stdio_fputs(bool wait, const char *str, FILE *file)
		: Base(wait, STDIO_FPUTS, 1024, SENTINELPREPARE(Prepare)), Str(str), File(file) { sentinelSend(this, sizeof(stdio_fputs)); }
	int RC;
};

struct stdio_ungetc
{
	sentinelMessage Base;
	int Ch; FILE *File;
	__device__ stdio_ungetc(bool wait, int ch, FILE *file)
		: Base(wait, STDIO_UNGETC, 0, nullptr), Ch(ch), File(file) { sentinelSend(this, sizeof(stdio_ungetc)); }
	int RC;
};

struct stdio_fread
{
	static __forceinline __device__ char *Prepare(stdio_fread *t, char *data, char *dataEnd)
	{
		t->Ptr = (char *)(data += _ROUND8(sizeof(*t)));
		char *end = (char *)(data += 1024);
		if (end > dataEnd) return nullptr;
		return end;
	}
	sentinelMessage Base;
	size_t Size; size_t Num; FILE *File;
	__device__ stdio_fread(bool wait, size_t size, size_t num, FILE *file)
		: Base(wait, STDIO_FREAD, 1024, SENTINELPREPARE(Prepare)), Size(size), Num(num), File(file) { sentinelSend(this, sizeof(stdio_fread)); }
	size_t RC;
	void *Ptr;
};

struct stdio_fwrite
{
	static __forceinline __device__ char *Prepare(stdio_fwrite *t, char *data, char *dataEnd)
	{
		size_t size = t->Size * t->Num;
		char *ptr = (char *)(data += _ROUND8(sizeof(*t)));
		char *end = (char *)(data += size);
		if (end > dataEnd) return nullptr;
		memcpy(ptr, t->Ptr, size);
		t->Ptr = ptr;
		return end;
	}
	sentinelMessage Base;
	const void *Ptr; size_t Size; size_t Num; FILE *File;
	__device__ stdio_fwrite(bool wait, const void *ptr, size_t size, size_t num, FILE *file)
		: Base(wait, STDIO_FWRITE, 1024, SENTINELPREPARE(Prepare)), Ptr(ptr), Size(size), Num(num), File(file) { sentinelSend(this, sizeof(stdio_fwrite)); }
	size_t RC;
};

struct stdio_fseek
{
	sentinelMessage Base;
	FILE *File; long int Offset; int Origin;
	__device__ stdio_fseek(bool wait, FILE *file, long int offset, int origin)
		: Base(wait, STDIO_FSEEK, 0, nullptr), File(file), Offset(offset), Origin(origin) { sentinelSend(this, sizeof(stdio_fseek)); }
	int RC;
};

struct stdio_ftell
{
	sentinelMessage Base;
	FILE *File;
	__device__ stdio_ftell(FILE *file)
		: Base(true, STDIO_FTELL, 0, nullptr), File(file) { sentinelSend(this, sizeof(stdio_ftell)); }
	int RC;
};

struct stdio_rewind
{
	sentinelMessage Base;
	FILE *File;
	__device__ stdio_rewind(FILE *file)
		: Base(true, STDIO_REWIND, 0, nullptr), File(file) { sentinelSend(this, sizeof(stdio_rewind)); }
};

struct stdio_fgetpos
{
	sentinelMessage Base;
	FILE *File;
	fpos_t *Pos;
	__device__ stdio_fgetpos(FILE *__restrict file, fpos_t *__restrict pos)
		: Base(true, STDIO_FGETPOS, 0, nullptr), File(file), Pos(pos) { sentinelSend(this, sizeof(stdio_fgetpos)); }
	int RC;
};

struct stdio_fsetpos
{
	sentinelMessage Base;
	FILE *File;
	const fpos_t *Pos;
	__device__ stdio_fsetpos(FILE *__restrict file, const fpos_t *pos)
		: Base(true, STDIO_FSETPOS, 0, nullptr), File(file), Pos(pos) { sentinelSend(this, sizeof(stdio_fsetpos)); }
	int RC;
};

struct stdio_clearerr
{
	sentinelMessage Base;
	FILE *File;
	__device__ stdio_clearerr(FILE *file)
		: Base(false, STDIO_CLEARERR, 0, nullptr), File(file) { sentinelSend(this, sizeof(stdio_clearerr)); }
};

struct stdio_feof
{
	sentinelMessage Base;
	FILE *File;
	__device__ stdio_feof(FILE *file)
		: Base(true, STDIO_FEOF, 0, nullptr), File(file) { sentinelSend(this, sizeof(stdio_feof)); }
	int RC;
};

struct stdio_ferror
{
	sentinelMessage Base;
	FILE *File;
	__device__ stdio_ferror(FILE *file)
		: Base(true, STDIO_FERROR, 0, nullptr), File(file) { sentinelSend(this, sizeof(stdio_ferror)); }
	int RC;
};

struct stdio_fileno
{
	sentinelMessage Base;
	FILE *File;
	__device__ stdio_fileno(FILE *file)
		: Base(false, STDIO_FILENO, 0, nullptr), File(file) { sentinelSend(this, sizeof(stdio_fileno)); }
	int RC;
};


#endif  /* _INC_SENTINEL_STDIOMSG */