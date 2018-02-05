#include <stringcu.h> //: wal.c
#include <ext/global.h>
#include <assert.h>
#include "libcu.sqlite.h"

#define PgHdr PgHdr
#	define pDirty dirty
#	define pData data

#define SQLITEINT_H
#include "sqlite3\sqliteLimit.h"
#include "sqlite3\wal.c"