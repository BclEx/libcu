#include <stringcu.h> //: pager.c
#include <ext/global.h>
#include <ext/bitvec.h>
#include <assert.h>
#include "libcu.sqlite.h"

#define PgHdr PgHdr
#	define pPager pager
#	define pDirty dirty
#	define pData data

#define sqlite3_backup backupbase_t

#define SQLITEINT_H
//#include "sqlite3\sqliteLimit.h"

//#include "sqlite3\pager.h"
//#include "sqlite3\pager.c"