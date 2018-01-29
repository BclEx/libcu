#include <ext/global.h>
#include <ext/convert.h>
#include <assert.h>
#include "libcu.sqlite.h"

#define sqlite3Isalpha(c) isalpha(c)

#define SQLITEINT_H
#include "sqlite3\os_win.c"