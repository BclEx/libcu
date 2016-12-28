#include <stdiocu.h>
#include <cuda_runtimecu.h>
#include <sentinel-stdiomsg.h>

//__device__ int _close(int a) { io_close msg(a); return msg.RC; }

__BEGIN_DECLS;

__constant__ FILE *__iob_file[3] = { (FILE *)1, (FILE *)2, (FILE *)3 };
//__device__ FILE *stdin;
//__device__ FILE *stdout;
//__device__ FILE *stderr;

__END_DECLS;

/* Remove file FILENAME.  */
//__device__ int remove(const char *filename) { return -1; }
/* Rename file OLD to NEW.  */
//__device__ int rename(const char *old, const char *new_) { stdio_rename msg(old, new_); return msg.RC; }
/* Remove file FILENAME.  */
//__device__ int _unlink(const char *filename) { stdio_unlink msg(filename); return msg.RC; }

/* Close STREAM. */
//__device__ int fclose(FILE *stream, bool wait) { if (stream == stdout || stream == stderr) return 0; stdio_fclose msg(wait, stream); return msg.RC; }
/* Flush STREAM, or all streams if STREAM is NULL. */
//__device__ int fflush(FILE *stream) { if (stream == stdout || stream == stderr) return 0; stdio_fflush msg(false, stream); return msg.RC; }

/* Open a file and create a new stream for it. */
//__device__ FILE *fopen(const char *__restrict filename, const char *__restrict modes) { stdio_fopen msg(filename, modes); return msg.RC; }

/* Open a file, replacing an existing stream with it. */
//__device__ FILE *freopen(const char *__restrict filename, const char *__restrict modes, FILE *__restrict stream) { return nullptr; }

/* If BUF is NULL, make STREAM unbuffered. Else make it use buffer BUF, of size BUFSIZ.  */
//__device__ void setbuf(FILE *__restrict stream, char *__restrict buf) { }
/* Make STREAM use buffering mode MODE. If BUF is not NULL, use N bytes of it for buffering; else allocate an internal buffer N bytes long.  */
//__device__ int setvbuf(FILE *__restrict stream, char *__restrict buf, int modes, size_t n) { stdio_setvbuf msg(stream, buf, modes, n); return msg.RC; }

/* Write formatted output to S from argument list ARG. */
__device__ int vfprintf(FILE *__restrict s, const char *__restrict format, va_list arg) { return -1; }
//__device__ int fprintf(FILE *f, const char *v, bool wait) { stdio_fprintf msg(wait, f, v); _free((void *)v); return msg.RC; }

/* Write formatted output to stdout from argument list ARG. */
__device__ int vprintf(const char *__restrict format, va_list arg) { return -1; }
/* Write formatted output to S from argument list ARG.  */
__device__ int vsprintf(char *__restrict s, const char *__restrict format, va_list arg) { return -1; }

__device__ int vsnprintf(char *__restrict s, size_t maxlen, const char *__restrict format, va_list arg) { return -1; }

/* Read formatted input from S into argument list ARG.  */
__device__ int vfscanf(FILE *__restrict s, const char *__restrict format, va_list arg) { return -1; }
/* Read formatted input from stdin into argument list ARG. */
__device__ int vscanf(const char *__restrict format, va_list arg) { return -1; }
/* Read formatted input from S into argument list ARG.  */
__device__ int vsscanf(const char *__restrict s, const char *__restrict format, va_list arg) { return -1; }

/* Read a character from STREAM.  */
//__device__ int fgetc(FILE *stream) { stdio_fgetc msg(stream); return msg.RC; }
//__device__ int getc(FILE *stream) { return -1; }

/* Read a character from stdin.  */
__device__ int getchar(void) { return -1; }

/* Write a character to STREAM.  */
//__device__ int fputc(int c, FILE *stream, bool wait) { if (stream == stdout || stream == stderr) { printf("%c", c); return 0; } stdio_fputc msg(wait, c, stream); return msg.RC; }
//__device__ int putc(int c, FILE *stream) { return -1; }

/* Write a character to stdout.  */
__device__ int putchar(int c) { return -1; }

/* Get a newline-terminated string of finite length from STREAM.  */
//__device__ char *fgets(char *__restrict s, int n, FILE *__restrict stream) { stdio_fgets msg(s, n, stream); return msg.RC; }

/* Write a string to STREAM.  */
//__device__ int fputs(const char *__restrict s, FILE *__restrict stream, bool wait) { if (stream == stdout || stream == stderr) { printf(s); return 0; } stdio_fputs msg(wait, s, stream); return msg.RC; }

/* Write a string, followed by a newline, to stdout.  */
__device__ int puts(const char *s) { printf("%s\n", s); return -1; }

/* Push a character back onto the input buffer of STREAM.  */
__device__ int ungetc(int c, FILE *stream) { return -1; }

/* Read chunks of generic data from STREAM.  */
//__device__ size_t fread(void *__restrict ptr, size_t size, size_t n, FILE *__restrict stream, bool wait) { stdio_fread msg(wait, size, n, stream); memcpy(ptr, msg.Ptr, msg.RC); return msg.RC; }
/* Write chunks of generic data to STREAM.  */
//__device__ size_t fwrite(const void *__restrict ptr, size_t size, size_t n, FILE *__restrict s, bool wait) { stdio_fwrite msg(wait, ptr, size, n, s); return msg.RC; }

/* Seek to a certain position on STREAM.  */
//__device__ int fseek(FILE *stream, long int off, int whence) { stdio_fseek msg(true, stream, off, whence); return msg.RC; }
/* Return the current position of STREAM.  */
//__device__ long int ftell(FILE *stream) { stdio_ftell msg(stream); return msg.RC; }
/* Rewind to the beginning of STREAM.  */
//__device__ void rewind(FILE *stream) { }

/* Clear the error and EOF indicators for STREAM.  */
//__device__ void clearerr(FILE *stream) { stdio_clearerr msg(stream); }
/* Return the EOF indicator for STREAM.  */
//__device__ int feof(FILE *stream) { stdio_feof msg(stream); return msg.RC; }
/* Return the error indicator for STREAM.  */
//__device__ int ferror(FILE *stream) { if (stream == stdout || stream == stderr) return 0; stdio_ferror msg(stream); return msg.RC; }

/* Print a message describing the meaning of the value of errno.  */
//__device__ void perror(const char *s) { }

/* Return the system file descriptor for STREAM.  */
//__device__ int _fileno(FILE *stream) { return (stream == stdin ? 0 : stream == stdout ? 1 : stream == stderr ? 2 : -1); }