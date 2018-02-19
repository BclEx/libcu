#include <ext/global.h>
#include <ext/convert.h>
#include <assert.h>
#include "libcu.sqlite.h"

#define sqlite3Isalpha(c) isalpha(c)
#define sqlite3_os_init runtime_os_init
#define sqlite3_os_end runtime_os_end

#define SQLITEINT_H
#include "sqlite3\os_win.c"