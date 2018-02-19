#include <ext/global.h>
#include <ext/convert.h>
#include <assert.h>
#include "libcu.sqlite.h"

#define sqlite3Isalpha(c) isalpha(c)
#define sqlite3_os_init runtime_os_init
#define sqlite3_os_end runtime_os_end
#define sqlite3_win32_write_debug runtime_win32_write_debug
#define sqlite3_win32_sleep runtime_win32_sleep
#define sqlite3_win32_is_nt runtime_win32_is_nt
#define sqlite3_win32_utf8_to_unicode runtime_win32_utf8_to_unicode
#define sqlite3_win32_unicode_to_utf8 runtime_win32_unicode_to_utf8
#define sqlite3_win32_mbcs_to_utf8 runtime_win32_mbcs_to_utf8
#define sqlite3_win32_mbcs_to_utf8_v2 runtime_win32_mbcs_to_utf8_v2
#define sqlite3_win32_utf8_to_mbcs runtime_win32_utf8_to_mbcs
#define sqlite3_win32_utf8_to_mbcs_v2 runtime_win32_utf8_to_mbcs_v2
#define sqlite3_win32_set_directory runtime_win32_set_directory

#define SQLITEINT_H
#include "sqlite3\os_win.c"