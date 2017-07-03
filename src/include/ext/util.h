/*
util.h - xxx
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
#ifndef _EXT_UTIL_H
#define _EXT_UTIL_H
#ifdef  __cplusplus
extern "C" {
#endif

#ifdef LIBCU_UNTESTABLE
#define sqlite3FaultSim(X) RC_OK
#else
	__host_device__ RC sqlite3FaultSim(int);
#endif

	//__host_device__ void tagErrorWithMsg(tagbase_t *tag, int errCode, const char *format, ...)
	__host_device__ void tagError(tagbase_t *tag, int errCode);
	__host_device__ void tagSystemError(tagbase_t *tag, RC rc);
	__host_device__ bool tagSafetyCheckOk(tagbase_t *tag);
	__host_device__ bool tagSafetyCheckSickOrOk(tagbase_t *tag);

#ifdef  __cplusplus
}
#endif
#endif	/* _EXT_UTIL_H */