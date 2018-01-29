#include <ext/global.h> //: pragma.c
#include <assert.h>

/* Interpret the given string as a safety level.  Return 0 for OFF, 1 for ON or NORMAL, 2 for FULL, and 3 for EXTRA.  Return 1 for an empty or 
** unrecognized string argument.  The FULL and EXTRA option is disallowed if the omitFull parameter it 1.
**
** Note that the values returned are one less that the values that should be passed into sqlite3BtreeSetSafetyLevel().  The is done
** to support legacy SQL code.  The safety level used to be boolean and older scripts may have used numbers 0 for OFF and 1 for ON.
*/
static __host_device__ uint8_t getSafetyLevel(const char *z, int omitFull, uint8_t dflt)
{
	// 123456789 123456789 123
	static const char texts[] = "onoffalseyestruextrafull";
	static const uint8_t offsets[] = {0, 1, 2,  4,    9,  12,  15,   20};
	static const uint8_t lengths[] = {2, 2, 3,  5,    3,   4,   5,    4};
	static const uint8_t values[] =  {1, 0, 0,  0,    1,   1,   3,    2};
	// on no off false yes true extra full
	if (isdigit(*z))
		return (uint8_t)convert_atoi(z);
	int n = (int)strlen(z);
	for (int i = 0; i < _LENGTHOF(lengths); i++)
		if (lengths[i] == n && !strnicmp(&texts[offsets[i]], z , n) && (!omitFull || values[i] <= 1)) return values[i];
	return dflt;
}

/* Interpret the given string as a boolean value. */
uint8_t util_getBoolean(const char *z, uint8_t dflt) //: sqlite3GetBoolean
{
	return getSafetyLevel(z, 1, dflt) != 0;
}