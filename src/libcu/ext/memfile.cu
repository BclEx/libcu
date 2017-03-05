#include <ext\memfile.h>
#include <stdlibcu.h>
#include <stddefcu.h>
#include <assert.h>

#ifdef  __cplusplus
extern "C" {
#endif

#define JOURNAL_CHUNKSIZE ((int)(1024 - sizeof(fileChunk_t *)))

	typedef struct filePoint_t filePoint_t;
	typedef struct fileChunk_t fileChunk_t;

	struct fileChunk_t
	{
		fileChunk_t *next;			// Next chunk in the journal
		uint8_t chunk[JOURNAL_CHUNKSIZE]; // Content of this chunk
	};

	struct filePoint_t
	{
		int64_t offset;				// Offset from the beginning of the file
		fileChunk_t *chunk;			// Specific chunk into which cursor points
	};

	typedef struct memfile_t
	{
		bool opened;
		fileChunk_t *first;			// Head of in-memory chunk-list
		filePoint_t endpoint;		// Pointer to the end of the file
		filePoint_t readpoint;		// Pointer to the end of the last xRead()
	} memfile_t;

	__constant__ int __sizeofMemfile_t = _ROUND64(sizeof(memfile_t));

	__device__ void memfileOpen(memfile_t *f)
	{
		memset(f, 0, sizeof(memfile_t));
		f->opened = true;
	}

#define MIN(a, b) ((a) < (b) ? a : b)
	__device__ void memfileRead(memfile_t *f, void *buffer, int amount, int64_t offset)
	{
		// never try to read past the end of an in-memory file
		assert(offset + amount <= f->endpoint.offset);
		fileChunk_t *chunk;
		if (f->readpoint.offset != offset || offset == 0) {
			int64_t offset2 = 0;
			for (chunk = f->first; _ALWAYS(chunk) && (offset2 + JOURNAL_CHUNKSIZE) <= offset; chunk = chunk->next)
				offset2 += JOURNAL_CHUNKSIZE;
		}
		else
			chunk = f->readpoint.chunk;
		int chunkOffset = (int)(offset % JOURNAL_CHUNKSIZE);
		uint8_t *out = (uint8_t *)buffer;
		int read = amount;
		do {
			int space = JOURNAL_CHUNKSIZE - chunkOffset;
			int copy = MIN(read, (JOURNAL_CHUNKSIZE - chunkOffset));
			memcpy(out, &chunk->chunk[chunkOffset], copy);
			out += copy;
			read -= space;
			chunkOffset = 0;
		} while (read >= 0 && (chunk = chunk->next) && read > 0);
		f->readpoint.offset = offset + amount;
		f->readpoint.chunk = chunk;
	}

	__device__ bool memfileWrite(memfile_t *f, const void *buffer, int amount, int64_t offset)
	{
		// An in-memory file should only ever be appended to.
		assert(offset == f->endpoint.offset);
		uint8_t *b = (uint8_t *)buffer;
		while (amount > 0) {
			fileChunk_t *chunk = f->endpoint.chunk;
			int chunkOffset = (int)(f->endpoint.offset % JOURNAL_CHUNKSIZE);
			int space = MIN(amount, JOURNAL_CHUNKSIZE - chunkOffset);
			if (!chunkOffset) {
				// New chunk is required to extend the file
				fileChunk_t *newChunk = (fileChunk_t *)malloc(sizeof(fileChunk_t));
				if (!newChunk)
					return false;
				newChunk->next = nullptr;
				if (chunk) { assert(f->first); chunk->next = newChunk; }
				else { assert(!f->first); f->first = newChunk; }
				f->endpoint.chunk = newChunk;
			}
			memcpy(&f->endpoint.chunk->chunk[chunkOffset], b, space);
			b += space;
			amount -= space;
			f->endpoint.offset += space;
		}
		return true;
	}

	__device__ void memfileTruncate(memfile_t *f, int64_t size)
	{
		assert(!size);
		fileChunk_t *chunk = f->first;
		while (chunk)
		{
			fileChunk_t *tmp = chunk;
			chunk = chunk->next;
			free(tmp);
		}
		memfileOpen(f);
	}

	__device__ void memfileClose(memfile_t *f)
	{
		if (!f->opened)
			return;
		memfileTruncate(f, 0);
		f->opened = false;
	}

	__device__ int64_t memfileGetFileSize(memfile_t *f)
	{
		return (int64_t)f->endpoint.offset;
	}

#ifdef  __cplusplus
}
#endif