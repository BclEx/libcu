/*
hash.h - xxx
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

//#pragma once
#ifndef _EXT_HASH_H
#define _EXT_HASH_H
#ifdef  __cplusplus
extern "C" {
#endif

	struct hashElem_t
	{
		hashElem_t *next, *prev;		// Next and previous elements in the table
		void *data;						// Data associated with this element
		const char *key;				// Key associated with this element
	};

	struct hash_t
	{
		unsigned int tableSize;			// Number of buckets in the hash table
		unsigned int count;				// Number of entries in this table
		hashElem_t *first;				// The first element of the array
		struct htable_t
		{              
			int count;					// Number of entries with this hash
			hashElem_t *chain;			// Pointer to first entry with this hash
		} *table; // the hash table
	};

	extern __device__ void hashInit(hash_t *h);
	extern __device__ void *hashInsert(hash_t *h, const char *key, void *data);
	extern __device__ void *hashFind(hash_t *h, const char *key);
	extern __device__ void hashClear(hash_t *h);
#define hashFirst(h) ((h)->first)
#define hashNext(e) ((e)->next)
#define hashData(e) ((e)->data)

#define HASHINIT { 0, 0, nullptr, nullptr }

#ifdef  __cplusplus
}
#endif
#endif  /* _EXT_HASH_H */