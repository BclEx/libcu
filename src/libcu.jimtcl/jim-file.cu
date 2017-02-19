/*
* Implements the file command for jim
*
* (c) 2008 Steve Bennett <steveb@workware.net.au>
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions
* are met:
*
* 1. Redistributions of source code must retain the above copyright
*    notice, this list of conditions and the following disclaimer.
* 2. Redistributions in binary form must reproduce the above
*    copyright notice, this list of conditions and the following
*    disclaimer in the documentation and/or other materials
*    provided with the distribution.
*
* THIS SOFTWARE IS PROVIDED BY THE JIM TCL PROJECT ``AS IS'' AND ANY
* EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
* THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
* PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
* JIM TCL PROJECT OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
* INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
* (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
* OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
* HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
* STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
* ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
* ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*
* The views and conclusions contained in the software and documentation
* are those of the authors and should not be interpreted as representing
* official policies, either expressed or implied, of the Jim Tcl Project.
*
* Based on code originally from Tcl 6.7:
*
* Copyright 1987-1991 Regents of the University of California
* Permission to use, copy, modify, and distribute this
* software and its documentation for any purpose and without
* fee is hereby granted, provided that the above copyright
* notice appear in all copies.  The University of California
* makes no representations about the suitability of this
* software for any purpose.  It is provided "as is" without
* express or implied warranty.
*/

//#include <limitscu.h>
//#include <stdlibcu.h>
//#include <stringcu.h>
//#include <stdiocu.h>
//#include <errnocu.h>
#include <sys/statcu.h>
#include "jimautoconf.h"
#include "jim-subcmd.h"
#ifdef HAVE_UTIMES
#include <sys/time.h>
#endif
#ifdef HAVE_UNISTD_H
#include <unistdcu.h>
#elif defined(_MSC_VER)
#include <direct.h>
#define F_OK 0
#define W_OK 2
#define R_OK 4
#define S_ISREG(m) (((m) & S_IFMT) == S_IFREG)
#define S_ISDIR(m) (((m) & S_IFMT) == S_IFDIR)
#endif

#ifndef MAXPATHLEN
#define MAXPATHLEN JIM_PATH_LEN
#endif

#if defined(__MINGW32__) || defined(_MSC_VER)
#define ISWINDOWS 1
#define HAVE_MKDIR_ONE_ARG
#else
#define ISWINDOWS 0
#endif

// JimGetFileType --
//  Given a mode word, returns a string identifying the type of a file.
// Results:
//  A static text string giving the file type from mode.
// Side effects:
//  None.
//
static __device__ const char *JimGetFileType(int mode)
{
	if (S_ISREG(mode)) return "file";
	else if (S_ISDIR(mode)) return "directory";
#ifdef S_ISCHR
	else if (S_ISCHR(mode)) return "characterSpecial";
#endif
#ifdef S_ISBLK
	else if (S_ISBLK(mode)) return "blockSpecial";
#endif
#ifdef S_ISFIFO
	else if (S_ISFIFO(mode)) return "fifo";
#endif
#ifdef S_ISLNK
	else if (S_ISLNK(mode)) return "link";
#endif
#ifdef S_ISSOCK
	else if (S_ISSOCK(mode)) return "socket";
#endif
	return "unknown";
}

// StoreStatData --
//  This is a utility procedure that breaks out the fields of a "stat" structure and stores them in textual form into the
//  elements of an associative array.
//
// Results:
//  Returns a standard Tcl return value.  If an error occurs then a message is left in interp->result.
//
// Side effects:
//  Elements of the associative array given by "varName" are modified.
static __device__ void AppendStatElement(Jim_Interp *interp, Jim_Obj *listObj, const char *key, jim_wide value)
{
	Jim_ListAppendElement(interp, listObj, Jim_NewStringObj(interp, key, -1));
	Jim_ListAppendElement(interp, listObj, Jim_NewIntObj(interp, value));
}

static __device__ int StoreStatData(Jim_Interp *interp, Jim_Obj *varName, const struct stat *sb)
{
	// Just use a list to store the data
	Jim_Obj *listObj = Jim_NewListObj(interp, NULL, 0);
	AppendStatElement(interp, listObj, "dev", sb->st_dev);
	AppendStatElement(interp, listObj, "ino", sb->st_ino);
	AppendStatElement(interp, listObj, "mode", sb->st_mode);
	AppendStatElement(interp, listObj, "nlink", sb->st_nlink);
	AppendStatElement(interp, listObj, "uid", sb->st_uid);
	AppendStatElement(interp, listObj, "gid", sb->st_gid);
	AppendStatElement(interp, listObj, "size", sb->st_size);
	AppendStatElement(interp, listObj, "atime", sb->st_atime);
	AppendStatElement(interp, listObj, "mtime", sb->st_mtime);
	AppendStatElement(interp, listObj, "ctime", sb->st_ctime);
	Jim_ListAppendElement(interp, listObj, Jim_NewStringObj(interp, "type", -1));
	Jim_ListAppendElement(interp, listObj, Jim_NewStringObj(interp, JimGetFileType((int)sb->st_mode), -1));
	// Was a variable specified?
	if (varName) {
		Jim_Obj *objPtr = Jim_GetVariable(interp, varName, JIM_NONE);
		if (objPtr) {
			if (Jim_DictSize(interp, objPtr) < 0) {
				// This message matches the one from Tcl
				Jim_SetResultFormatted(interp, "can't set \"%#s(dev)\": variable isn't array", varName);
				Jim_FreeNewObj(interp, listObj);
				return JIM_ERROR;
			}
			if (Jim_IsShared(objPtr))
				objPtr = Jim_DuplicateObj(interp, objPtr);
			// Just cheat here and append as a list and convert to a dict
			Jim_ListAppendList(interp, objPtr, listObj);
			Jim_DictSize(interp, objPtr);
			Jim_InvalidateStringRep(objPtr);
			Jim_FreeNewObj(interp, listObj);
			listObj = objPtr;
		}
		Jim_SetVariable(interp, varName, listObj);
	}
	// And also return the value
	Jim_SetResult(interp, listObj);
	return JIM_OK;
}

static __device__ int file_cmd_dirname(Jim_Interp *interp, int argc, Jim_Obj *const *argv)
{
	const char *path = Jim_String(argv[0]);
	const char *p = strrchr((char *)path, '/');
	if (!p && path[0] == '.' && path[1] == '.' && path[2] == '\0') Jim_SetResultString(interp, "..", -1);
	else if (!p) Jim_SetResultString(interp, ".", -1);
	else if (p == path) Jim_SetResultString(interp, "/", -1);
	else if (ISWINDOWS && p[-1] == ':')  Jim_SetResultString(interp, path, (int)(p - path) + 1); // z:/dir => z:/
	else Jim_SetResultString(interp, path, (int)(p - path));
	return JIM_OK;
}

static __device__ int file_cmd_rootname(Jim_Interp *interp, int argc, Jim_Obj *const *argv)
{
	const char *path = Jim_String(argv[0]);
	const char *lastSlash = strrchr((char *)path, '/');
	const char *p = strrchr((char *)path, '.');
	if (p == NULL || (lastSlash != NULL && lastSlash > p))
		Jim_SetResult(interp, argv[0]);
	else
		Jim_SetResultString(interp, path, (int)(p - path));
	return JIM_OK;
}

static __device__ int file_cmd_extension(Jim_Interp *interp, int argc, Jim_Obj *const *argv)
{
	const char *path = Jim_String(argv[0]);
	const char *lastSlash = strrchr((char *)path, '/');
	const char *p = strrchr((char *)path, '.');
	if (p == NULL || (lastSlash != NULL && lastSlash >= p))
		p = "";
	Jim_SetResultString(interp, p, -1);
	return JIM_OK;
}

static __device__ int file_cmd_tail(Jim_Interp *interp, int argc, Jim_Obj *const *argv)
{
	const char *path = Jim_String(argv[0]);
	const char *lastSlash = strrchr((char *)path, '/');
	if (lastSlash)
		Jim_SetResultString(interp, lastSlash + 1, -1);
	else
		Jim_SetResult(interp, argv[0]);
	return JIM_OK;
}

static __device__ int file_cmd_normalize(Jim_Interp *interp, int argc, Jim_Obj *const *argv)
{
#ifdef HAVE_REALPATH
	const char *path = Jim_String(argv[0]);
	char *newname = Jim_Alloc(MAXPATHLEN + 1);
	if (realpath(path, newname)) {
		Jim_SetResult(interp, Jim_NewStringObjNoAlloc(interp, newname, -1));
		return JIM_OK;
	}
	else {
		Jim_Free(newname);
		Jim_SetResultFormatted(interp, "can't normalize \"%#s\": %s", argv[0], strerror(errno));
		return JIM_ERROR;
	}
#else
	Jim_SetResultString(interp, "Not implemented", -1);
	return JIM_ERROR;
#endif
}

static __device__ int file_cmd_join(Jim_Interp *interp, int argc, Jim_Obj *const *argv)
{
	char *newname = (char *)Jim_Alloc(MAXPATHLEN + 1);
	char *last = newname;
	*newname = 0;
	// Simple implementation for now
	for (int i = 0; i < argc; i++) {
		int len;
		const char *part = Jim_GetString(argv[i], &len);
		// Absolute component, so go back to the start
		if (*part == '/')
			last = newname;
		// Absolute component on mingw, so go back to the start
		else if (ISWINDOWS && strchr(part, ':'))
			last = newname;
		else if (part[0] == '.') {
			if (part[1] == '/') { part += 2; len -= 2; }
			// Adding '.' to an existing path does nothing
			else if (part[1] == 0 && last != newname) continue;
		}
		// Add a slash if needed
		if (last != newname && last[-1] != '/')
			*last++ = '/';
		if (len) {
			if (last + len - newname >= MAXPATHLEN) {
				Jim_Free(newname);
				Jim_SetResultString(interp, "Path too long", -1);
				return JIM_ERROR;
			}
			memcpy(last, part, len);
			last += len;
		}
		// Remove a slash if needed
		if (last > newname + 1 && last[-1] == '/')
			if (!ISWINDOWS || !(last > newname + 2 && last[-2] == ':')) // but on on Windows, leave the trailing slash on "c:/ "
				*--last = 0;
	}
	*last = 0;
	// Probably need to handle some special cases ...
	Jim_SetResult(interp, Jim_NewStringObjNoAlloc(interp, newname, (int)(last - newname)));
	return JIM_OK;
}

static __device__ int file_access(Jim_Interp *interp, Jim_Obj *filename, int mode)
{
	Jim_SetResultBool(interp, access(Jim_String(filename), mode) != -1);
	return JIM_OK;
}

static __device__ int file_cmd_readable(Jim_Interp *interp, int argc, Jim_Obj *const *argv)
{
	return file_access(interp, argv[0], R_OK);
}

static __device__ int file_cmd_writable(Jim_Interp *interp, int argc, Jim_Obj *const *argv)
{
	return file_access(interp, argv[0], W_OK);
}

static __device__ int file_cmd_executable(Jim_Interp *interp, int argc, Jim_Obj *const *argv)
{
#ifdef X_OK
	return file_access(interp, argv[0], X_OK);
#else
	// If no X_OK, just assume true
	Jim_SetResultBool(interp, 1);
	return JIM_OK;
#endif
}

static __device__ int file_cmd_exists(Jim_Interp *interp, int argc, Jim_Obj *const *argv)
{
	return file_access(interp, argv[0], F_OK);
}

static __device__ int file_cmd_delete(Jim_Interp *interp, int argc, Jim_Obj *const *argv)
{
	int force = Jim_CompareStringImmediate(interp, argv[0], "-force");
	if (force || Jim_CompareStringImmediate(interp, argv[0], "--")) {
		argc++;
		argv--;
	}
	while (argc--) {
		const char *path = Jim_String(argv[0]);
		if (_unlink(path) == -1 && errno != ENOENT) {
			if (_rmdir(path) == -1)
				if (!force || Jim_EvalPrefix(interp, "file delete force", 1, argv) != JIM_OK) { // Maybe try using the script helper
					Jim_SetResultFormatted(interp, "couldn't delete file \"%s\": %s", path, strerror(errno));
					return JIM_ERROR;
				}
		}
		argv++;
	}
	return JIM_OK;
}

#if __CUDACC__
#define MKDIR_DEFAULT(PATHNAME) __mkdir(PATHNAME)
#else
#ifdef HAVE_MKDIR_ONE_ARG
#define MKDIR_DEFAULT(PATHNAME) mkdir(PATHNAME)
#else
#define MKDIR_DEFAULT(PATHNAME) mkdir(PATHNAME, 0755)
#endif
#endif

// Create directory, creating all intermediate paths if necessary.
// Returns 0 if OK or -1 on failure (and sets errno)
// Note: The path may be modified.
static __device__ int mkdir_all(char *path)
{
	int ok = 1;
	// First time just try to make the dir
	goto first;
	while (ok--) {
		// Must have failed the first time, so recursively make the parent and try again
		{
			char *slash = strrchr(path, '/');
			if (slash && slash != path) {
				*slash = 0;
				if (mkdir_all(path) != 0)
					return -1;
				*slash = '/';
			}
		}
first:
		if (MKDIR_DEFAULT(path) == 0)
			return 0;
		if (errno == ENOENT)
			continue; // Create the parent and try again
		// Maybe it already exists as a directory
		if (errno == EEXIST) {
			struct stat sb;
			if (_stat(path, &sb) == 0 && S_ISDIR(sb.st_mode))
				return 0;
			// Restore errno
			errno = EEXIST;
		}
		// Failed
		break;
	}
	return -1;
}

static __device__ int file_cmd_mkdir(Jim_Interp *interp, int argc, Jim_Obj *const *argv)
{
	while (argc--) {
		char *path = Jim_StrDup(Jim_String(argv[0]));
		int rc = mkdir_all(path);
		Jim_Free(path);
		if (rc != 0) {
			Jim_SetResultFormatted(interp, "can't create directory \"%#s\": %s", argv[0], strerror(errno));
			return JIM_ERROR;
		}
		argv++;
	}
	return JIM_OK;
}

static __device__ int file_cmd_tempfile(Jim_Interp *interp, int argc, Jim_Obj *const *argv)
{
	int fd = Jim_MakeTempFile(interp, (argc >= 1) ? Jim_String(argv[0]) : NULL);
	if (fd < 0)
		return JIM_ERROR;
	__close(fd);
	return JIM_OK;
}

static __device__ int file_cmd_rename(Jim_Interp *interp, int argc, Jim_Obj *const *argv)
{
	int force = 0;
	if (argc == 3) {
		if (!Jim_CompareStringImmediate(interp, argv[0], "-force"))
			return -1;
		force++;
		argv++;
		argc--;
	}
	const char *source = Jim_String(argv[0]);
	const char *dest = Jim_String(argv[1]);
	if (!force && access(dest, F_OK) == 0) {
		Jim_SetResultFormatted(interp, "error renaming \"%#s\" to \"%#s\": target exists", argv[0], argv[1]);
		return JIM_ERROR;
	}
	if (rename(source, dest) != 0) {
		Jim_SetResultFormatted(interp, "error renaming \"%#s\" to \"%#s\": %s", argv[0], argv[1], strerror(errno));
		return JIM_ERROR;
	}
	return JIM_OK;
}

#if defined(HAVE_LINK) && defined(HAVE_SYMLINK)
static const char * const _link_options[] = { "-hard", "-symbolic", NULL };
static __device__ int file_cmd_link(Jim_Interp *interp, int argc, Jim_Obj *const *argv)
{
	enum { OPT_HARD, OPT_SYMBOLIC, };
	int option = OPT_HARD;
	if (argc == 3) {
		if (Jim_GetEnum(interp, argv[0], _link_options, &option, NULL, JIM_ENUM_ABBREV | JIM_ERRMSG) != JIM_OK)
			return JIM_ERROR;
		argv++;
		argc--;
	}
	const char *dest = Jim_String(argv[0]);
	const char *source = Jim_String(argv[1]);
	int ret = (option == OPT_HARD ? link(source, dest) : symlink(source, dest));
	if (ret != 0) {
		Jim_SetResultFormatted(interp, "error linking \"%#s\" to \"%#s\": %s", argv[0], argv[1], strerror(errno));
		return JIM_ERROR;
	}
	return JIM_OK;
}
#endif

static __device__ int file_stat(Jim_Interp *interp, Jim_Obj *filename, struct stat *sb)
{
	const char *path = Jim_String(filename);
	if (_stat(path, sb) == -1) {
		Jim_SetResultFormatted(interp, "could not read \"%#s\": %s", filename, strerror(errno));
		return JIM_ERROR;
	}
	return JIM_OK;
}

#ifdef HAVE_LSTAT
static __device__ int file_lstat(Jim_Interp *interp, Jim_Obj *filename, struct _stat *sb)
{
	const char *path = Jim_String(filename);
	if (_lstat(path, sb) == -1) {
		Jim_SetResultFormatted(interp, "could not read \"%#s\": %s", filename, strerror(errno));
		return JIM_ERROR;
	}
	return JIM_OK;
}
#else
#define file_lstat file_stat
#endif

static __device__ int file_cmd_atime(Jim_Interp *interp, int argc, Jim_Obj *const *argv)
{
	struct stat sb;
	if (file_stat(interp, argv[0], &sb) != JIM_OK)
		return JIM_ERROR;
	Jim_SetResultInt(interp, sb.st_atime);
	return JIM_OK;
}

static __device__ int file_cmd_mtime(Jim_Interp *interp, int argc, Jim_Obj *const *argv)
{
	if (argc == 2) {
#ifdef HAVE_UTIMES
		jim_wide newtime;
		struct timeval times[2];
		if (Jim_GetWide(interp, argv[1], &newtime) != JIM_OK)
			return JIM_ERROR;
		times[1].tv_sec = times[0].tv_sec = newtime;
		times[1].tv_usec = times[0].tv_usec = 0;
		if (utimes(Jim_String(argv[0]), times) != 0) {
			Jim_SetResultFormatted(interp, "can't set time on \"%#s\": %s", argv[0], strerror(errno));
			return JIM_ERROR;
		}
#else
		Jim_SetResultString(interp, "Not implemented", -1);
		return JIM_ERROR;
#endif
	}
	struct stat sb;
	if (file_stat(interp, argv[0], &sb) != JIM_OK)
		return JIM_ERROR;
	Jim_SetResultInt(interp, sb.st_mtime);
	return JIM_OK;
}

static __device__ int file_cmd_copy(Jim_Interp *interp, int argc, Jim_Obj *const *argv)
{
	return Jim_EvalPrefix(interp, "file copy", argc, argv);
}

static __device__ int file_cmd_size(Jim_Interp *interp, int argc, Jim_Obj *const *argv)
{
	struct stat sb;
	if (file_stat(interp, argv[0], &sb) != JIM_OK)
		return JIM_ERROR;
	Jim_SetResultInt(interp, sb.st_size);
	return JIM_OK;
}

static __device__ int file_cmd_isdirectory(Jim_Interp *interp, int argc, Jim_Obj *const *argv)
{
	struct stat sb;
	int ret = 0;
	if (file_stat(interp, argv[0], &sb) == JIM_OK)
		ret = S_ISDIR(sb.st_mode);
	Jim_SetResultInt(interp, ret);
	return JIM_OK;
}

static __device__ int file_cmd_isfile(Jim_Interp *interp, int argc, Jim_Obj *const *argv)
{
	struct stat sb;
	int ret = 0;
	if (file_stat(interp, argv[0], &sb) == JIM_OK)
		ret = S_ISREG(sb.st_mode);
	Jim_SetResultInt(interp, ret);
	return JIM_OK;
}

#ifdef HAVE_GETEUID
static __device__ int file_cmd_owned(Jim_Interp *interp, int argc, Jim_Obj *const *argv)
{
	struct _stat sb;
	int ret = 0;
	if (file_stat(interp, argv[0], &sb) == JIM_OK)
		ret = (geteuid() == sb.st_uid);
	Jim_SetResultInt(interp, ret);
	return JIM_OK;
}
#endif

#if defined(HAVE_READLINK)
static __device__ int file_cmd_readlink(Jim_Interp *interp, int argc, Jim_Obj *const *argv)
{
	const char *path = Jim_String(argv[0]);
	char *linkValue = Jim_Alloc(MAXPATHLEN + 1);
	int linkLength = readlink(path, linkValue, MAXPATHLEN);
	if (linkLength == -1) {
		Jim_Free(linkValue);
		Jim_SetResultFormatted(interp, "couldn't readlink \"%#s\": %s", argv[0], strerror(errno));
		return JIM_ERROR;
	}
	linkValue[linkLength] = 0;
	Jim_SetResult(interp, Jim_NewStringObjNoAlloc(interp, linkValue, linkLength));
	return JIM_OK;
}
#endif

static __device__ int file_cmd_type(Jim_Interp *interp, int argc, Jim_Obj *const *argv)
{
	struct stat sb;
	if (file_lstat(interp, argv[0], &sb) != JIM_OK)
		return JIM_ERROR;
	Jim_SetResultString(interp, JimGetFileType((int)sb.st_mode), -1);
	return JIM_OK;
}

#ifdef HAVE_LSTAT
static __device__ int file_cmd_lstat(Jim_Interp *interp, int argc, Jim_Obj *const *argv)
{
	struct _stat sb;
	if (file_lstat(interp, argv[0], &sb) != JIM_OK)
		return JIM_ERROR;
	return StoreStatData(interp, argc == 2 ? argv[1] : NULL, &sb);
}
#else
#define file_cmd_lstat file_cmd_stat
#endif

static __device__ int file_cmd_stat(Jim_Interp *interp, int argc, Jim_Obj *const *argv)
{
	struct stat sb;
	if (file_stat(interp, argv[0], &sb) != JIM_OK)
		return JIM_ERROR;
	return StoreStatData(interp, argc == 2 ? argv[1] : NULL, &sb);
}

__constant__ static const jim_subcmd_type _file_command_table[] = {
	{ "atime", "name", file_cmd_atime, 1, 1 }, // Description: Last access time
	{ "mtime", "name ?time?", file_cmd_mtime, 1, 2 }, // Description: Get or set last modification time
	{ "copy", "?-force? source dest", file_cmd_copy, 2, 3 }, // Description: Copy source file to destination file
	{ "dirname", "name", file_cmd_dirname, 1, 1, }, // Description: Directory part of the name
	{ "rootname", "name", file_cmd_rootname, 1, 1 }, // Description: Name without any extension
	{ "extension", "name", file_cmd_extension, 1, 1, }, // Description: Last extension including the dot
	{ "tail", "name", file_cmd_tail, 1, 1 }, // Description: Last component of the name
	{ "normalize", "name", file_cmd_normalize, 1, 1 }, // Description: Normalized path of name
	{ "join", "name ?name ...?", file_cmd_join, 1, -1 }, // Description: Join multiple path components
	{ "readable", "name", file_cmd_readable, 1, 1 }, // Description: Is file readable
	{ "writable", "name", file_cmd_writable, 1, 1 }, // Description: Is file writable
	{ "executable", "name", file_cmd_executable, 1, 1 }, // Description: Is file executable
	{ "exists", "name", file_cmd_exists, 1, 1 }, // Description: Does file exist
	{ "delete", "?-force|--? name ...", file_cmd_delete, 1, -1 }, // Description: Deletes the files or directories (must be empty unless -force)
	{ "mkdir", "dir ...", file_cmd_mkdir, 1, -1 }, // Description: Creates the directories
	{ "tempfile", "?template?", file_cmd_tempfile, 0, 1 }, // Description: Creates a temporary filename
	{ "rename", "?-force? source dest", file_cmd_rename, 2, 3 }, // Description: Renames a file
#if defined(HAVE_LINK) && defined(HAVE_SYMLINK)
	{ "link", "?-symbolic|-hard? newname target", file_cmd_link, 2, 3 }, // Description: Creates a hard or soft link
#endif
#if defined(HAVE_READLINK)
	{ "readlink", "name", file_cmd_readlink, 1, 1 }, // Description: Value of the symbolic link
#endif
	{ "size", "name", file_cmd_size, 1, 1 }, // Description: Size of file
	{ "stat", "name ?var?", file_cmd_stat, 1, 2 }, // Description: Returns results of stat, and may store in var array
	{ "lstat", "name ?var?", file_cmd_lstat, 1, 2 }, // Description: Returns results of lstat, and may store in var array
	{ "type", "name", file_cmd_type, 1, 1 }, // Description: Returns type of the file
#ifdef HAVE_GETEUID
	{ "owned", "name", file_cmd_owned, 1, 1 }, // Description: Returns 1 if owned by the current owner
#endif
	{ "isdirectory", "name", file_cmd_isdirectory, 1, 1 }, // Description: Returns 1 if name is a directory
	{ "isfile", "name", file_cmd_isfile, 1, 1 }, // Description: Returns 1 if name is a file
	{ NULL }
};

static __device__ int Jim_CdCmd(ClientData dummy, Jim_Interp *interp, int argc, Jim_Obj *const *argv)
{
	if (argc != 2) {
		Jim_WrongNumArgs(interp, 1, argv, "dirname");
		return JIM_ERROR;
	}
	const char *path = Jim_String(argv[1]);
	if (chdir(path) != 0) {
		Jim_SetResultFormatted(interp, "couldn't change working directory to \"%s\": %s", path, strerror(errno));
		return JIM_ERROR;
	}
	return JIM_OK;
}

static __device__ int Jim_PwdCmd(ClientData dummy, Jim_Interp *interp, int argc, Jim_Obj *const *argv)
{
	char *cwd = (char *)Jim_Alloc(MAXPATHLEN);
	if (getcwd(cwd, MAXPATHLEN) == NULL) {
		Jim_SetResultString(interp, "Failed to get pwd", -1);
		Jim_Free(cwd);
		return JIM_ERROR;
	}
	else if (ISWINDOWS) {
		// Try to keep backslashes out of paths
		char *p = cwd;
		while ((p = (char *)strchr(p, '\\')) != NULL)
			*p++ = '/';
	}
	Jim_SetResultString(interp, cwd, -1);
	Jim_Free(cwd);
	return JIM_OK;
}

__device__ int Jim_fileInit(Jim_Interp *interp)
{
	if (Jim_PackageProvide(interp, "file", "1.0", JIM_ERRMSG))
		return JIM_ERROR;
	Jim_CreateCommand(interp, "file", Jim_SubCmdProc, (void *)_file_command_table, NULL);
	Jim_CreateCommand(interp, "pwd", Jim_PwdCmd, NULL, NULL);
	Jim_CreateCommand(interp, "cd", Jim_CdCmd, NULL, NULL);
	return JIM_OK;
}
