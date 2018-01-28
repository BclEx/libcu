#include <ext/math.h> //: util.c
#include <assert.h>

#ifndef OMIT_FLOATING_POINT
/* Return true if the floating point value is Not a Number (NaN).
**
** Use the math library isnan() function if compiled with SQLITE_HAVE_ISNAN. Otherwise, we have our own implementation that works on most systems.
*/
__host_device__ int math_isnan(double x) //: sqlite3IsNaN
{
	int rc;
#if !LIBCU_HAVE_ISNAN && !HAVE_ISNAN
#ifdef __FAST_MATH__
# error LIBCU will not work correctly with the -ffast-math option of GCC.
#endif
	volatile double y = x;
	volatile double z = y;
	rc = (y != z);
#else  /* if HAVE_ISNAN */
	rc = isnan(x);
#endif /* HAVE_ISNAN */
	ASSERTCOVERAGE(rc);
	return rc;
}
#endif /* SQLITE_OMIT_FLOATING_POINT */


/* Attempt to add, substract, or multiply the 64-bit signed value b against the other 64-bit signed integer at *ao and store the result in *ao.
** Return 0 on success.  Or if the operation would have resulted in an overflow, leave *ao unchanged and return 1.
*/
__host_device__ bool math_add64(int64_t *ao, int64_t b) //: sqlite3AddInt64
{
#if GCC_VERSION>=5004000 && !defined(__INTEL_COMPILER)
	return __builtin_add_overflow(*a, b, a);
#else
	int64_t a = *ao;
	ASSERTCOVERAGE(a == 0); ASSERTCOVERAGE(a == 1);
	ASSERTCOVERAGE(b == -1); ASSERTCOVERAGE(b == 0);
	if (b >= 0) {
		ASSERTCOVERAGE(a > 0 && INT64_MAX - a == b);
		ASSERTCOVERAGE(a > 0 && INT64_MAX - a == b - 1);
		if (a > 0 && INT64_MAX - a < b) return true;
	}
	else {
		ASSERTCOVERAGE(a < 0 && -(a + INT64_MAX) == b + 1);
		ASSERTCOVERAGE(a < 0 && -(a + INT64_MAX) == b + 2);
		if (a < 0 && -(a + INT64_MAX) > b + 1) return true;
	}
	*ao += b;
	return false; 
#endif
}

__host_device__ bool math_sub64(int64_t *ao, int64_t b) //: sqlite3SubInt64
{
#if GCC_VERSION>=5004000 && !defined(__INTEL_COMPILER)
	return __builtin_sub_overflow(*ao, b, ao);
#else
	ASSERTCOVERAGE(b == INT64_MIN + 1);
	if (b == INT64_MIN) {
		ASSERTCOVERAGE(*ao == -1); ASSERTCOVERAGE(*ao == 0);
		if (*ao >= 0) return true;
		*ao -= b;
		return false;
	}
	else return math_add64(ao, -b);
#endif
}

__host_device__ bool math_mul64(int64_t *ao, int64_t b) //: sqlite3MulInt64
{
#if GCC_VERSION>=5004000 && !defined(__INTEL_COMPILER)
	return __builtin_mul_overflow(*ao, b, ao);
#else
	int64_t a = *ao;
	if (b > 0) {
		if (a > INT64_MAX / b) return true;
		if (a < INT64_MIN / b) return true;
	}
	else if (b < 0) {
		if (a > 0) {
			if (b < INT64_MIN / a) return true;
		}
		else if (a < 0) {
			if (b == INT64_MIN) return true;
			if (a == INT64_MIN) return true;
			if (-a > INT64_MAX / -b) return true;
		}
	}
	*ao = a*b;
	return false;
#endif
}

/* Compute the absolute value of a 32-bit signed integer, of possible.  Or if the integer has a value of -2147483648, return +2147483647 */
__host_device__ int math_abs32(int x) //: sqlite3AbsInt32
{
	if (x >= 0) return x;
	if (x == (int)0x80000000) return 0x7fffffff;
	return -x;
}

/* Find (an approximate) sum of two logest_t values.  This computation is not a simple "+" operator because logest_t is stored as a logarithmic value. */
static __host_constant__ const unsigned char __math_logEstAdd[] = {
	10, 10,                        // 0,1
	9, 9,                          // 2,3
	8, 8,                          // 4,5
	7, 7, 7,                       // 6,7,8
	6, 6, 6,                       // 9,10,11
	5, 5, 5,                       // 12-14
	4, 4, 4, 4,                    // 15-18
	3, 3, 3, 3, 3, 3,              // 19-24
	2, 2, 2, 2, 2, 2, 2,           // 25-31
};
__host_device__ logest_t math_addLogest(logest_t a, logest_t b) //: sqlite3LogEstAdd
{
	if (a >= b) {
		if (a > b + 49) return a;
		if (a > b + 31) return a + 1;
		return a + __math_logEstAdd[a - b];
	}
	else {
		if (b > a + 49) return b;
		if (b > a + 31) return b + 1;
		return b + __math_logEstAdd[b - a];
	}
}

/* Convert an integer into a logest_t.  In other words, compute an approximation for 10*log2(x). */
static __host_constant__ logest_t __math_logest[] = { 0, 2, 3, 5, 6, 7, 8, 9 };
__host_device__ logest_t math_logest(uint64_t x) //: sqlite3LogEst
{
	logest_t y = 40;
	if (x < 8) {
		if (x < 2) return 0;
		while (x < 8) { y -= 10; x <<= 1; }
	}
	else {
#if GCC_VERSION>=5004000
		int i = 60 - __builtin_clzll(x);
		y += i * 10;
		x >>= i;
#else
		while (x > 255) { y += 40; x >>= 4; }  /*OPTIMIZATION-IF-TRUE*/
		while (x > 15) { y += 10; x >>= 1; }
#endif
	}
	return __math_logest[x & 7] + y - 10;
}

/* Convert a double into a logest_t. In other words, compute an approximation for 10*log2(x). */
__host_device__ logest_t math_logestFromDouble(double x) //: sqlite3LogEstFromDouble
{
	uint64_t a;
	logest_t e;
	assert(sizeof(x) == 8 && sizeof(a) == 8);
	if (x <= 1) return 0;
	if (x <= 2000000000) return math_logest((uint64_t)x);
	memcpy(&a, &x, 8);
	e = (a >> 52) - 1022;
	return e * 10;
}

#if defined(SQLITE_ENABLE_STMT_SCANSTATUS) || \
	defined(SQLITE_ENABLE_STAT3_OR_STAT4) || \
	defined(SQLITE_EXPLAIN_ESTIMATED_ROWS)
/*
** Convert a logest_t into an integer.
**
** Note that this routine is only used when one or more of various non-standard compile-time options is enabled.
*/
uint64_t sqlite3LogEstToInt(logest_t x) //: sqlite3LogEstToInt
{
	uint64_t n = x % 10;
	x /= 10;
	if (n >= 5) n -= 2;
	else if (n >= 1) n -= 1;
#if defined(SQLITE_ENABLE_STMT_SCANSTATUS) || defined(SQLITE_EXPLAIN_ESTIMATED_ROWS)
	if (x > 60) return (uint64_t)INT64_MAX;
#else
	// If only SQLITE_ENABLE_STAT3_OR_STAT4 is on, then the largest input possible to this routine is 310, resulting in a maximum x of 31
	assert(x <= 60);
#endif
	return x >= 3 ? (n + 8) << (x - 3) : (n + 8) >> (3 - x);
}
#endif