#include <ext/global.h> //: random.c
#include <assert.h>

/* All threads share a single random number generator. This structure is the current state of the generator. */
static __hostb_device__ _WSD struct PrngGlobal {
	bool isInit;  // True if initialized
	unsigned char i, j;         // State variables
	unsigned char s[256];       // State variables
} _prng;
#define prng _GLOBAL(struct PrngGlobal, _prng)

/* Return N random bytes. */
__host_device__ void randomness_(int n, void *p) //: sqlite3_randomness
{
#ifndef OMIT_AUTOINIT
	if (runtimeInitialize()) return;
#endif
#if LIBCU_THREADSAFE
	mutex *mutex = mutex_alloc(MUTEX_STATIC_PRNG);
#endif
	mutex_enter(mutex);
	if (n <= 0 || !p) {
		prng.isInit = false;
		mutex_leave(mutex);
		return;
	}

	// Initialize the state of the random number generator once, the first time this routine is called.  The seed value does
	// not need to contain a lot of randomness since we are not trying to do secure encryption or anything like that...
	//
	// Nothing in this file or anywhere else in SQLite does any kind of encryption.  The RC4 algorithm is being used as a PRNG (pseudo-random
	// number generator) not as an encryption device.
	unsigned char t;
	unsigned char *z = (unsigned char *)p;
	if (!prng.isInit) {
		char k[256];
		prng.j = 0;
		prng.i = 0;
		vsys_randomness(vsystemFind(nullptr), 256, k);
		int i;
		for (i = 0; i < 256; i++)
			prng.s[i] = (uint8_t)i;
		for (i = 0; i < 256; i++) {
			prng.j += prng.s[i] + k[i];
			t = prng.s[prng.j];
			prng.s[prng.j] = prng.s[i];
			prng.s[i] = t;
		}
		prng.isInit = true;
	}
	assert(n > 0);
	do {
		prng.i++;
		t = prng.s[prng.i];
		prng.j += t;
		prng.s[prng.i] = prng.s[prng.j];
		prng.s[prng.j] = t;
		t += prng.s[prng.i];
		*(z++) = prng.s[t];
	} while (--n);
	mutex_leave(mutex);
}

#ifndef LIBCU_UNTESTABLE
/* For testing purposes, we sometimes want to preserve the state of PRNG and restore the PRNG to its saved state at a later time, or
** to reset the PRNG to its initial state.  These routines accomplish those tasks.
**
** The sqlite3_test_control() interface calls these routines to control the PRNG.
*/
static __hostb_device__ _WSD struct PrngGlobal _prngSaved;
__host_device__ void randomness_save() //: sqlite3PrngSaveState
{
	memcpy(&_GLOBAL(struct PrngGlobal, _prngSaved), &_GLOBAL(struct PrngGlobal, _prng), sizeof(_prng));
}

__host_device__ void randomness_restore() //: sqlite3PrngRestoreState
{
	memcpy(&_GLOBAL(struct PrngGlobal, _prng), &_GLOBAL(struct PrngGlobal, _prngSaved), sizeof(_prng));
}
#endif /* LIBCU_UNTESTABLE */
