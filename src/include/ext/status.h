/*
status.h - xxx
The MIT License

Copyright (c) 2016 Sky Morey

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <ext\global.h>
#ifndef _EXT_STATUS_H
#define _EXT_STATUS_H
__BEGIN_DECLS;

// CAPI3REF: Status Parameters
#define STATUS int
#define STATUS_MEMORY_USED			0
#define STATUS_PAGECACHE_USED       1
#define STATUS_PAGECACHE_OVERFLOW   2
#define STATUS_SCRATCH_USED         3
#define STATUS_SCRATCH_OVERFLOW     4
#define STATUS_MALLOC_SIZE          5
#define STATUS_PARSER_STACK         6
#define STATUS_PAGECACHE_SIZE       7
#define STATUS_SCRATCH_SIZE         8
#define STATUS_MALLOC_COUNT         9

// CAPI3REF: Libcu Runtime Status
/* Query status information. */
extern __host_device__ RC status(STATUS op, int *current, int *highwater, bool resetFlag);
/* Query status information. */
extern __host_device__ RC status64(STATUS op, int64_t *current, int64_t *highwater, bool resetFlag);

// CAPI3REF: Status Parameters for tag objects
#define TAGSTATUS_LOOKASIDE_USED       0
//#define TAGSTATUS_CACHE_USED           1
//#define TAGSTATUS_SCHEMA_USED          2
//#define TAGSTATUS_STMT_USED            3
#define TAGSTATUS_LOOKASIDE_HIT        4
#define TAGSTATUS_LOOKASIDE_MISS_SIZE  5
#define TAGSTATUS_LOOKASIDE_MISS_FULL  6
//#define TAGSTATUS_CACHE_HIT            7
//#define TAGSTATUS_CACHE_MISS           8
//#define TAGSTATUS_CACHE_WRITE          9
//#define TAGSTATUS_DEFERRED_FKS        10
//#define TAGSTATUS_CACHE_USED_SHARED   11
#define TAGSTATUS_MAX                 11   // Largest defined TAGSTATUS

// CAPI3REF: Database Connection Status
extern __host_device__ RC tagstatus(tagbase_t *tag, STATUS op, int *current, int *highwater, bool resetFlag);

/* Return the current value of a status parameter. */
extern __host_device__ int64_t status_now(STATUS op);
/* Add N to the value of a status record. */
extern __host_device__ void status_inc(STATUS op, int n);
/* Dec N to the value of a status record. */
extern __host_device__ void status_dec(STATUS op, int n);
/* Adjust the highwater mark if necessary. */
extern __host_device__ void status_max(STATUS op, int x);

__END_DECLS;
#endif	/* _EXT_STATUS_H */