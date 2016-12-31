#include <stdiocu.h>
#include <stringcu.h>
#include <cuda_runtimecu.h>
#include <sentinel-stdiomsg.h>

//__device__ int _close(int a) { io_close msg(a); return msg.RC; }

__BEGIN_DECLS;

__constant__ FILE *__iob_file[3] = { (FILE *)1, (FILE *)2, (FILE *)3 };
//__device__ FILE *stdin;
//__device__ FILE *stdout;
//__device__ FILE *stderr;

#pragma region printf

/*
__device__ char *vmtagprintf(void *tag, const char *format, va_list args)
{
	//if (!RuntimeInitialize()) return nullptr;
	assert(tag != nullptr);
	char base[PRINT_BUF_SIZE];
	strbld_t b;
	strbldInit(&b, base, sizeof(base), 1000000000); //? tag->Limit[LIMIT_LENGTH]);
	b.Tag = tag;
	strbldAppendFormat(b, true, format, args);
	char *z = strbldToString(b);
	// if (b->AllocFailed) tagallocfailed(tag);
	return z;
}

__device__ char *vmprintf(const char *format, va_list args)
{
	//if (!RuntimeInitialize()) return nullptr;
	char base[PRINT_BUF_SIZE];
	strbld_t b;
	strbldInit(&b, base, sizeof(base), CORE_MAX_LENGTH);
	b.AllocType = 2;
	strbldAppendFormat(b, false, format, args);
	return strbldToString(b);
}
*/

/* Write formatted output to S from argument list ARG. */
//__device__ int fprintf(FILE *f, const char *v, bool wait) { stdio_fprintf msg(wait, f, v); _free((void *)v); return msg.RC; }
__device__ int vfprintf(FILE *__restrict s, const char *__restrict format, va_list args)
{
	//strbld_t b;
	//strbldInit(&b, (char *)s, (int)maxlen, 0);
	//b.AllocType = 0;
	//strbldAppendFormat(b, false, format, args);
	//strbldToString(b);
	//return b->index;
	return -1;
}

/* Write formatted output to stdout from argument list ARG. */
//built-in: __device__ int vprintf(const char *__restrict format, va_list arg) { return -1; }
/* Write formatted output to S from argument list ARG.  */
__device__ int vsnprintf(char *__restrict s, size_t maxlen, const char *__restrict format, va_list args)
{
	if (maxlen <= 0) return -1;
	strbld_t b;
	strbldInit(&b, (char *)s, (int)maxlen, 0); b.allocType = 0;
	strbldAppendFormat(&b, false, format, args);
	strbldToString(&b);
	return b.index;
}

__device__ char *vmnprintf(char *__restrict s, size_t maxlen, const char *format, va_list args)
{
	if (maxlen <= 0) return (char *)s;
	strbld_t b;
	strbldInit(&b, (char *)s, (int)maxlen, 0); b.allocType = 0;
	strbldAppendFormat(&b, false, format, args);
	return strbldToString(&b);
}

#pragma endregion

__END_DECLS;


/* Read formatted input from S into argument list ARG.  */
__device__ int vfscanf(FILE *__restrict s, const char *__restrict format, va_list arg) { return -1; }
/* Read formatted input from stdin into argument list ARG. */
__device__ int vscanf(const char *__restrict format, va_list arg) { return -1; }
/* Read formatted input from S into argument list ARG.  */
__device__ int vsscanf(const char *__restrict s, const char *__restrict format, va_list arg) { return -1; }

/* Read a character from STREAM.  */
//__device__ int getc(FILE *stream) { return -1; }

/* Read a character from stdin.  */
__device__ int getchar(void) { return -1; }

/* Write a character to STREAM.  */
//__device__ int putc(int c, FILE *stream) { return -1; }

/* Write a character to stdout.  */
__device__ int putchar(int c) { return -1; }

/* Write a string, followed by a newline, to stdout.  */
__device__ int puts(const char *s) { printf("%s\n", s); return -1; }

/* Push a character back onto the input buffer of STREAM.  */
__device__ int ungetc(int c, FILE *stream) { return -1; }

/* Print a message describing the meaning of the value of errno.  */
//__device__ void perror(const char *s) { }
