#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <io.h>
#include <sentinel-iomsg.h>
#include <sentinel-stdiomsg.h>
#include <sentinel-stdlibmsg.h>
#include <math.h>

//#define panic(fmt, ...) { printf(fmt, __VA_ARGS__); exit(1); }

bool sentinelDefaultExecutor(void *tag, sentinelMessage *data, int length)
{
	switch (data->OP) {
	case STDIO_REMOVE: { stdio_remove *msg = (stdio_remove *)data; msg->RC = remove(msg->Str); return true; }
	case STDIO_RENAME: { stdio_rename *msg = (stdio_rename *)data; msg->RC = rename(msg->Oldname, msg->Newname); return true; }
	case STDIO_UNLINK: { stdio_unlink *msg = (stdio_unlink *)data; msg->RC = _unlink(msg->Str); return true; }
	case STDIO_FCLOSE: { stdio_fclose *msg = (stdio_fclose *)data; msg->RC = fclose(msg->File); return true; }
	case STDIO_FFLUSH: { stdio_fflush *msg = (stdio_fflush *)data; msg->RC = fflush(msg->File); return true; }
	case STDIO_FREOPEN: { stdio_freopen *msg = (stdio_freopen *)data; 
					   //FILE *f = (!msg->Stream ? fopen(msg->Filename, msg->Mode) : freopen(msg->Filename, msg->Mode, msg->Stream)); msg->RC = f;
					   return true; }
	case STDIO_SETVBUF: { stdio_setvbuf *msg = (stdio_setvbuf *)data; if (msg->Mode != -1) msg->RC = setvbuf(msg->File, msg->Buffer, msg->Mode, msg->Size); else setbuf(msg->File, msg->Buffer); return true; }
	case STDIO_FGETC: { stdio_fgetc *msg = (stdio_fgetc *)data; msg->RC = fgetc(msg->File); return true; }
	case STDIO_FGETS: { stdio_fgets *msg = (stdio_fgets *)data; msg->RC = fgets(msg->Str, msg->Num, msg->File); return true; }
	case STDIO_FPUTC: { stdio_fputc *msg = (stdio_fputc *)data; msg->RC = fputc(msg->Ch, msg->File); return true; }
	case STDIO_FPUTS: { stdio_fputs *msg = (stdio_fputs *)data; msg->RC = fputs(msg->Str, msg->File); return true; }
	case STDIO_UNGETC: { stdio_ungetc *msg = (stdio_ungetc *)data; msg->RC = ungetc(msg->Ch, msg->File); return true; }
	case STDIO_FREAD: { stdio_fread *msg = (stdio_fread *)data; msg->RC = fread(msg->Ptr, msg->Size, msg->Num, msg->File); return true; }
	case STDIO_FWRITE: { stdio_fwrite *msg = (stdio_fwrite *)data; msg->RC = fwrite(msg->Ptr, msg->Size, msg->Num, msg->File); return true; }
	case STDIO_FSEEK: { stdio_fseek *msg = (stdio_fseek *)data; msg->RC = fseek(msg->File, msg->Offset, msg->Origin); return true; }
	case STDIO_FTELL: { stdio_ftell *msg = (stdio_ftell *)data; msg->RC = ftell(msg->File); return true; }
	case STDIO_REWIND: { stdio_rewind *msg = (stdio_rewind *)data; rewind(msg->File); return true; }
	case STDIO_FGETPOS: { stdio_fgetpos *msg = (stdio_fgetpos *)data; msg->RC = fgetpos(msg->File, msg->Pos); return true; }
	case STDIO_FSETPOS: { stdio_fsetpos *msg = (stdio_fsetpos *)data; msg->RC = fsetpos(msg->File, msg->Pos); return true; }
	case STDIO_CLEARERR: { stdio_clearerr *msg = (stdio_clearerr *)data; clearerr(msg->File); return true; }
	case STDIO_FEOF: { stdio_feof *msg = (stdio_feof *)data; msg->RC = feof(msg->File); return true; }
	case STDIO_FERROR: { stdio_ferror *msg = (stdio_ferror *)data; msg->RC = ferror(msg->File); return true; }
	case STDIO_FILENO: { stdio_fileno *msg = (stdio_fileno *)data; msg->RC = _fileno(msg->File); return true; }
	case IO_CLOSE: { io_close *msg = (io_close *)data; msg->RC = _close(msg->Handle); return true; }
	case STDLIB_SYSTEM: { stdlib_system *msg = (stdlib_system *)data; msg->RC = system(msg->Str); return true; }
	case STDLIB_EXIT: { stdlib_exit *msg = (stdlib_exit *)data; if (msg->Std) exit(msg->Status); else _exit(msg->Status); return true; }
	}
	return false;
}