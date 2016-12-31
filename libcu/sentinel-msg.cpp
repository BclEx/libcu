#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <io.h>
#include <sentinel-iomsg.h>
#include <sentinel-stdiomsg.h>
#include <sentinel-stdlibmsg.h>
#include <math.h>

bool sentinelDefaultExecutor(void *tag, sentinelMessage *data, int length)
{
	printf("%d", data->OP);
	switch (data->OP)
	{
	case 0: {
		stdio_fprintf *msg = (stdio_fprintf *)data;
		msg->RC = fprintf(msg->File, msg->Format);
		return true; }
	case 1: {
		stdio_setvbuf *msg = (stdio_setvbuf *)data;
		msg->RC = setvbuf(msg->File, msg->Buffer, msg->Mode, msg->Size);
		return true; }
	case 2: {
		stdio_fopen *msg = (stdio_fopen *)data;
		msg->RC = fopen(msg->Filename, msg->Mode);
		return true; }
	case 3: {
		stdio_fflush *msg = (stdio_fflush *)data;
		msg->RC = fflush(msg->File);
		return true; }
	case 4: {
		stdio_fclose *msg = (stdio_fclose *)data;
		msg->RC = fclose(msg->File);
		return true; }
	case 5: {
		stdio_fgetc *msg = (stdio_fgetc *)data;
		msg->RC = fgetc(msg->File);
		return true; }
	case 6: {
		stdio_fgets *msg = (stdio_fgets *)data;
		msg->RC = fgets(msg->Str, msg->Num, msg->File);
		return true; }
	case 7: {
		stdio_fputc *msg = (stdio_fputc *)data;
		msg->RC = fputc(msg->Ch, msg->File);
		return true; }
	case 8: {
		stdio_fputs *msg = (stdio_fputs *)data;
		msg->RC = fputs(msg->Str, msg->File);
		return true; }
	case 9: {
		stdio_fread *msg = (stdio_fread *)data;
		msg->RC = fread(msg->Ptr, msg->Size, msg->Num, msg->File);
		return true; }
	case 10: {
		stdio_fwrite *msg = (stdio_fwrite *)data;
		msg->RC = fwrite(msg->Ptr, msg->Size, msg->Num, msg->File);
		return true; }
	case 11: {
		stdio_fseek *msg = (stdio_fseek *)data;
		msg->RC = fseek(msg->File, msg->Offset, msg->Origin);
		return true; }
	case 12: {
		stdio_ftell *msg = (stdio_ftell *)data;
		msg->RC = ftell(msg->File);
		return true; }
	case 13: {
		stdio_feof *msg = (stdio_feof *)data;
		msg->RC = feof(msg->File);
		return true; }
	case 14: {
		stdio_ferror *msg = (stdio_ferror *)data;
		msg->RC = ferror(msg->File);
		return true; }
	case 15: {
		stdio_clearerr *msg = (stdio_clearerr *)data;
		clearerr(msg->File);
		return true; }
	case 16: {
		stdio_rename *msg = (stdio_rename *)data;
		msg->RC = rename(msg->Oldname, msg->Newname);
		return true; }
	case 17: {
		stdio_unlink *msg = (stdio_unlink *)data;
		msg->RC = _unlink(msg->Str);
		return true; }
	case 18: {
		io_close *msg = (io_close *)data;
		msg->RC = _close(msg->Handle);
		return true; }
	case 19: {
		stdlib_system *msg = (stdlib_system *)data;
		msg->RC = system(msg->Str);
		return true; }
	}
	return false;
}