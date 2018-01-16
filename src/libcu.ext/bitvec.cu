#include <stdlibcu.h>
#include <ext/global.h>
#include <ext/bitvec.h>
#include <assert.h>

/* Size of the Bitvec structure in bytes. */
#define BITVEC_SZ        512
/* Round the union size down to the nearest pointer boundary, since that's how  it will be aligned within the Bitvec struct. */
#define BITVEC_USIZE	(((BITVEC_SZ - (3 * sizeof(uint32_t))) / sizeof(bitvec_t *)) * sizeof(bitvec_t *))
/* Type of the array "element" for the bitmap representation. Should be a power of 2, and ideally, evenly divide into BITVEC_USIZE. 
** Setting this to the "natural word" size of your CPU may improve performance. */
#define BITVEC_TELEM     uint8_t
/* Size, in bits, of the bitmap element. */
#define BITVEC_SZELEM    8
/* Number of elements in a bitmap array. */
#define BITVEC_NELEM     (BITVEC_USIZE / sizeof(BITVEC_TELEM))
/* Number of bits in the bitmap array. */
#define BITVEC_NBIT      (BITVEC_NELEM * BITVEC_SZELEM)
/* Number of u32 values in hash table. */
#define BITVEC_NINT      (BITVEC_USIZE / sizeof(uint32_t))
/* Maximum number of entries in hash table before  sub-dividing and re-hashing. */
#define BITVEC_MXHASH    (BITVEC_NINT/2)
/* Hashing function for the aHash representation. Empirical testing showed that the *37 multiplier 
** (an arbitrary prime)in the hash function provided no fewer collisions than the no-op *1. */
#define BITVEC_HASH(X)   (((X)*1)%BITVEC_NINT)
#define BITVEC_NPTR      (BITVEC_USIZE / sizeof(bitvec_t *))

/*
** A bitmap is an instance of the following structure.
**
** This bitmap records the existence of zero or more bits with values between 1 and iSize, inclusive.
**
** There are three possible representations of the bitmap. If iSize<=BITVEC_NBIT, then Bitvec.u.aBitmap[] is a straight
** bitmap.  The least significant bit is bit 1.
**
** If iSize>BITVEC_NBIT and iDivisor==0 then Bitvec.u.aHash[] is a hash table that will hold up to BITVEC_MXHASH distinct values.
**
** Otherwise, the value i is redirected into one of BITVEC_NPTR sub-bitmaps pointed to by Bitvec.u.apSub[].  Each subbitmap
** handles up to iDivisor separate values of i.  apSub[0] holds values between 1 and iDivisor.  apSub[1] holds values between
** iDivisor+1 and 2*iDivisor.  apSub[N] holds values between N*iDivisor+1 and (N+1)*iDivisor.  Each subbitmap is normalized
** to hold deal with values between 1 and iDivisor.
*/
struct bitvec_t {
	uint32_t size;      // Maximum bit index.  Max iSize is 4,294,967,296.
	uint32_t set;       // Number of bits that are set - only valid for aHash element.  Max is BITVEC_NINT.  For BITVEC_SZ of 512, this would be 125.
	uint32_t divisor;   // Number of bits handled by each apSub[] entry. Should >=0 for apSub element. Max iDivisor is max(u32) / BITVEC_NPTR + 1. For a BITVEC_SZ of 512, this would be 34,359,739.
	union {
		BITVEC_TELEM bitmap[BITVEC_NELEM];	// Bitmap representation
		uint32_t hash[BITVEC_NINT];			// Hash table representation
		bitvec_t *sub[BITVEC_NPTR];			// Recursive representation
	} u;
};

/*
** Create a new bitmap object able to handle bits between 0 and "size", inclusive.  Return a pointer to the new object.  Return NULL if 
** malloc fails.
*/
__device__ bitvec_t *bitvecNew(uint32_t size) //: sqlite3BitvecCreate
{
	assert(sizeof(bitvec_t) == BITVEC_SZ);
	bitvec_t *p = (bitvec_t *)allocZero(sizeof(bitvec_t));
	if (p)
		p->size = size;
	return p;
}

/*
** Check to see if the i-th bit is set.  Return true or false. If p is NULL (if the bitmap has not been created) or if
** i is out of range, then return false.
*/
__device__ bool bitvecGet(bitvec_t *p, uint32_t index) //: sqlite3BitvecTestNotNull
{
	assert(p);
	index--;
	if (index >= p->size)
		return false;
	while (p->divisor) {
		uint32_t bin = index / p->divisor;
		index %= p->divisor;
		p = p->u.sub[bin];
		if (!p)
			return false;
	}
	if (p->size <= BITVEC_NBIT)
		return (p->u.bitmap[index / BITVEC_SZELEM] & (1 << (index & (BITVEC_SZELEM - 1)))) != 0;
	uint32_t h = BITVEC_HASH(index++);
	while (p->u.hash[h]) {
		if (p->u.hash[h] == index)
			return true;
		h = (h + 1) % BITVEC_NINT;
	}
	return false;
}

/*
** Set the i-th bit.  Return 0 on success and an error code if anything goes wrong.
**
** This routine might cause sub-bitmaps to be allocated.  Failing to get the memory needed to hold the sub-bitmap is the only
** that can go wrong with an insert, assuming p and i are valid.
**
** The calling function must ensure that p is a valid Bitvec object and that the value for "i" is within range of the Bitvec object.
** Otherwise the behavior is undefined.
*/
__device__ bool bitvecSet(bitvec_t *p, uint32_t index) //: sqlite3BitvecSet
{
	if (!p)
		return true;
	assert(index > 0);
	assert(index <= p->size);
	index--;
	while ((p->size > BITVEC_NBIT) && p->divisor) {
		uint32_t bin = index / p->divisor;
		index %= p->divisor;
		if (!p->u.sub[bin] && !(p->u.sub[bin] = bitvecNew(p->divisor)))
			return RC_NOMEM_BKPT;
		p = p->u.sub[bin];
	}
	if (p->size <= BITVEC_NBIT) {
		p->u.bitmap[index / BITVEC_SZELEM] |= (1 << (index & (BITVEC_SZELEM - 1)));
		return true;
	}
	uint32_t h = BITVEC_HASH(index++);
	// if there wasn't a hash collision, and this doesn't completely fill the hash, then just add it without worring about sub-dividing and re-hashing.
	if (!p->u.hash[h])
		if (p->set < (BITVEC_NINT - 1)) goto bitvec_set_end;
		else goto bitvec_set_rehash;
		// there was a collision, check to see if it's already in hash, if not, try to find a spot for it
		do {
			if (p->u.hash[h] == index)
				return true;
			h++;
			if (h >= BITVEC_NINT) h = 0;
		} while (p->u.hash[h]);
		// we didn't find it in the hash.  h points to the first available free spot. check to see if this is going to make our hash too "full".
bitvec_set_rehash:
		if (p->set >= BITVEC_MXHASH) {
			uint32_t *values = (uint32_t *)tagstackAllocRaw(nullptr, sizeof(p->u.hash));
			if (!values)
				return RC_NOMEM_BKPT;
			memcpy(values, p->u.hash, sizeof(p->u.hash));
			memset(p->u.sub, 0, sizeof(p->u.sub));
			p->divisor = ((p->size + BITVEC_NPTR - 1) / BITVEC_NPTR);
			bool rc = bitvecSet(p, index);
			for (unsigned int j = 0; j < BITVEC_NINT; j++)
				if (values[j]) rc |= bitvecSet(p, values[j]);
			tagstackFree(nullptr, values);
			return rc;
		}
bitvec_set_end:
		p->set++;
		p->u.hash[h] = index;
		return true;
}

/*
** Clear the i-th bit.
**
** pBuf must be a pointer to at least BITVEC_SZ bytes of temporary storage
** that BitvecClear can use to rebuilt its hash table.
*/
__device__ void bitvecClear(bitvec_t *p, uint32_t index, void *buffer) //: sqlite3BitvecClear
{
	if (!p)
		return;
	assert(index > 0);
	index--;
	while (p->divisor) {
		uint32_t bin = index / p->divisor;
		index %= p->divisor;
		p = p->u.sub[bin];
		if (!p)
			return;
	}
	if (p->size <= BITVEC_NBIT)
		p->u.bitmap[index / BITVEC_SZELEM] &= ~(1 << (index & (BITVEC_SZELEM - 1)));
	else {
		uint32_t *values = (uint32_t *)buffer;
		memcpy(values, p->u.hash, sizeof(p->u.hash));
		memset(p->u.hash, 0, sizeof(p->u.hash));
		p->set = 0;
		for (unsigned int j = 0; j < BITVEC_NINT; j++)
			if (values[j] && values[j] != (index + 1)) {
				uint32_t h = BITVEC_HASH(values[j] - 1);
				p->set++;
				while (p->u.hash[h]) {
					h++;
					if (h >= BITVEC_NINT) h = 0;
				}
				p->u.hash[h] = values[j];
			}
	}
}

/*
** Destroy a bitmap object.  Reclaim all memory used.
*/
__device__ void bitvecDestroy(bitvec_t *p) //: sqlite3BitvecDestroy
{
	if (!p)
		return;
	if (p->divisor)
		for (unsigned int i = 0; i < BITVEC_NPTR; i++)
			bitvecDestroy(p->u.sub[i]);
	mfree(p);
}

/*
** Return the value of the iSize parameter specified when Bitvec *p was created.
*/
__device__ uint32_t bitvecSize(bitvec_t *p) //: sqlite3BitvecSize
{
	return p->size;
}

#ifndef LIBCU_UNTESTABLE

/*
** Let V[] be an array of unsigned characters sufficient to hold up to N bits.  Let I be an integer between 0 and N.  0<=I<N.
** Then the following macros can be used to set, clear, or test individual bits within V.
*/
#define SETBIT(V,I)      V[I>>3] |= (1<<(I&7))
#define CLEARBIT(V,I)    V[I>>3] &= ~(1<<(I&7))
#define TESTBIT(V,I)     (V[I>>3]&(1<<(I&7)))!=0

/*
** This routine runs an extensive test of the Bitvec code.
**
** The input is an array of integers that acts as a program
** to test the Bitvec.  The integers are opcodes followed
** by 0, 1, or 3 operands, depending on the opcode.  Another
** opcode follows immediately after the last operand.
**
** There are 6 opcodes numbered from 0 through 5.  0 is the
** "halt" opcode and causes the test to end.
**
**    0          Halt and return the number of errors
**    1 N S X    Set N bits beginning with S and incrementing by X
**    2 N S X    Clear N bits beginning with S and incrementing by X
**    3 N        Set N randomly chosen bits
**    4 N        Clear N randomly chosen bits
**    5 N S X    Set N bits from S increment X in array only, not in bitvec
**
** The opcodes 1 through 4 perform set and clear operations are performed
** on both a Bitvec object and on a linear array of bits obtained from malloc.
** Opcode 5 works on the linear array only, not on the Bitvec.
** Opcode 5 is used to deliberately induce a fault in order to
** confirm that error detection works.
**
** At the conclusion of the test the linear array is compared
** against the Bitvec object.  If there are any differences,
** an error is returned.  If they are the same, zero is returned.
**
** If a memory allocation error occurs, return -1.
*/
__device__ int bitvecBuiltinTest(int size, int *ops) //: sqlite3BitvecBuiltinTest
{
	int i, rc = -1;

	// Allocate the Bitvec to be tested and a linear array of bits to act as the reference
	bitvec_t *bitvec = bitvecNew(size);
	unsigned char *v = (unsigned char *)allocZero((size+7)/8+1);
	void *tmpSpace = alloc64((size_t)BITVEC_SZ);
	if (!bitvec || !v || !tmpSpace) goto bitvec_end;

	// NULL pBitvec tests
	bitvecSet(0, 1);
	bitvecClear(0, 1, tmpSpace);

	// Run the program
	int nx, op, pc = 0;
	while (op = ops[pc]) {
		switch (op) {
		case 1:
		case 2:
		case 5: {
			nx = 4;
			i = ops[pc+2] - 1;
			ops[pc+2] += ops[pc+3];
			break; }
		case 3:
		case 4: 
		default: {
			nx = 2;
			randomness_(sizeof(i), &i);
			break; }
		}
		if ((--ops[pc+1]) > 0) nx = 0;
		pc += nx;
		i = (i & 0x7fffffff) % size;
		if ((op & 1) != 0) {
			SETBIT(v, (i+1));
			if (op != 5)
				if (bitvecSet(bitvec, i+1)) goto bitvec_end;
		}
		else {
			CLEARBIT(v, (i+1));
			bitvecClear(bitvec, i+1, tmpSpace);
		}
	}

	// Test to make sure the linear array exactly matches the Bitvec object.  Start with the assumption that they do
	// match (rc==0).  Change rc to non-zero if a discrepancy is found.
	rc = bitvecGet(nullptr, 0) + bitvecGet(bitvec, size+1) + bitvecGet(bitvec, 0) + (bitvecSize(bitvec) - size);
	for (i = 1; i <= size; i++) {
		if (TESTBIT(v, i) != bitvecGet(bitvec, i)) {
			rc = i;
			break;
		}
	}

	/* Free allocated structure */
bitvec_end:
	mfree(tmpSpace);
	mfree(v);
	bitvecDestroy(bitvec);
	return rc;
}

#endif /* LIBCU_UNTESTABLE */