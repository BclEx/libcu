#include <sys/bitveccu.h>

__device__ bitvec_t *bitvecNew(uint32_t size)
{
	assert(sizeof(bitvec_t) == BITVEC_SZ);
	bitvec_t *p = (bitvec_t *)_allocZero(sizeof(bitvec_t));
	if (p)
		p->_size = size;
	return p;
}

__device__ bool bitvecGet(bitvec_t *p, uint32_t index)
{
	if (index > _size || index == 0)
		return false;
	index--;
	while (p->_divisor)
	{
		uint32_t bin = index / p->_divisor;
		index %= p->_divisor;
		p = p->u.sub[bin];
		if (!p) return false;
	}
	if (p->_size <= BITVEC_NBIT)
		return ((p->u.bitmap[index / BITVEC_SZELEM] & (1 << (index & (BITVEC_SZELEM - 1)))) != 0);
	uint32_t h = BITVEC_HASH(index++);
	while (p->u.hash[h]) {
		if (p->u.hash[h] == index) return true;
		h = (h + 1) % BITVEC_NINT;
	}
	return false;
}

__device__ bool bitvecSet(bitvec_t *p, uint32_t index)
{
	assert(index > 0);
	assert(index <= _size);
	index--;
	while ((p->_size > BITVEC_NBIT) && p->_divisor) {
		uint32_t bin = index / p->_divisor;
		index %= p->_divisor;
		if (!p->u.sub[bin] && !(p->u.sub[bin] = bitvecNew(p->_divisor))) return false;
		p = p->u.sub[bin];
	}
	if (p->_size <= BITVEC_NBIT) {
		p->u.bitmap[index / BITVEC_SZELEM] |= (1 << (index & (BITVEC_SZELEM - 1)));
		return true;
	}
	uint32_t h = BITVEC_HASH(index++);
	// if there wasn't a hash collision, and this doesn't completely fill the hash, then just add it without worring about sub-dividing and re-hashing.
	if (!p->u.hash[h])
		if (p->_set < (BITVEC_NINT - 1))
			goto bitvec_set_end;
		else
			goto bitvec_set_rehash;
	// there was a collision, check to see if it's already in hash, if not, try to find a spot for it
	do {
		if (p->u.hash[h] == index) return true;
		h++;
		if (h >= BITVEC_NINT) h = 0;
	} while (p->u.hash[h]);
	// we didn't find it in the hash.  h points to the first available free spot. check to see if this is going to make our hash too "full".
bitvec_set_rehash:
	if (p->_set >= BITVEC_MXHASH) {
		uint32_t *values = (uint32_t *)_stackalloc(nullptr, sizeof(p->u.hash));
		if (!values) return false;
		memcpy(values, p->u.hash, sizeof(p->u.hash));
		memset(p->u.sub, 0, sizeof(p->u.sub));
		p->_divisor = ((p->_size + BITVEC_NPTR - 1) / BITVEC_NPTR);
		bool rc = bitvecSet(p, index);
		for (unsigned int j = 0; j < BITVEC_NINT; j++)
			if (values[j]) rc |= p->Set(values[j]);
		_stackfree(nullptr, values);
		return rc;
	}
bitvec_set_end:
	p->_set++;
	p->u.hash[h] = index;
	return true;
}

__device__ void bitvecClear(bitvec_t *p, uint32_t index, void *buffer)
{
	assert(index > 0);
	index--;
	while (p->_divisor) {
		uint32_t bin = index / p->_divisor;
		index %= p->_divisor;
		p = p->u.sub[bin];
		if (!p) return;
	}
	if (p->_size <= BITVEC_NBIT)
		p->u.bitmap[index / BITVEC_SZELEM] &= ~(1 << (index & (BITVEC_SZELEM - 1)));
	else {
		uint32_t *values = (uint32_t *)buffer;
		memcpy(values, p->u.hash, sizeof(p->u.hash));
		memset(p->u.hash, 0, sizeof(p->u.hash));
		p->_set = 0;
		for (unsigned int j = 0; j < BITVEC_NINT; j++)
			if (values[j] && values[j] != (index + 1)) {
				uint32_t h = BITVEC_HASH(values[j] - 1);
				p->_set++;
				while (p->u.hash[h]) {
					h++;
					if (h >= BITVEC_NINT) h = 0;
				}
				p->u.hash[h] = values[j];
			}
	}
}