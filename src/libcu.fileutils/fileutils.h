#ifndef _FILEUTILS_H
#define _FILEUTILS_H

#define	PATHLEN 256

//#define	CMDLEN		512	
//#define	MAXARGS		500	
//#define	ALIASALLOC	20
//#define	STDIN		0
//#define	STDOUT		1
//#define	MAXSOURCE	10

#undef S_ISLNK
#ifdef S_ISLNK
#define	LSTAT lstat
#else
#define	LSTAT stat
#endif

//#ifndef isblank
//#define isblank(ch)	((ch) == ' ' || (ch) == '\t')
//#endif
//#define isquote(ch)	((ch) == '"' || (ch) == '\'')
//#define isdecimal(ch) ((ch) >= '0' && (ch) <= '9')
//#define isoctal(ch)	((ch) >= '0' && (ch) <= '7')

extern __device__ bool copyFile(char *srcName, char *destName, bool setModes);

struct group { short gr_gid; };
__device__ struct group *getgrnam(char *name) { return nullptr; }
struct passwd { short pw_uid; };
__device__ struct passwd *getpwnam(char *name) { return nullptr; }

#endif  /* _FILEUTILS_H */