﻿#include <ext\vfile.h>
#include <stdint.h>
#include <new.h>

namespace ext
{
#define JOURNAL_CHUNKSIZE ((int)(1024 - sizeof(fileChunk_t *)))

	typedef struct filePoint_t filePoint_t;
	typedef struct fileChunk_t fileChunk_t;

	struct fileChunk_t
	{
		fileChunk_t *Next;				// Next chunk in the journal
		uint8_t Chunk[JOURNAL_CHUNKSIZE];	// Content of this chunk
	};

	struct filePoint_t
	{
		int64_t Offset;		// Offset from the beginning of the file
		fileChunk_t *Chunk;	// Specific chunk into which cursor points
	};

	class memoryVFile : public vfile
	{
	private:
		fileChunk_t *First;       // Head of in-memory chunk-list
		filePoint_t _endpoint;    // Pointer to the end of the file
		filePoint_t _readpoint;   // Pointer to the end of the last xRead()
	public:
		//: VFILE::Opened
		__device__ virtual RC Read(void *buffer, int amount, int64 offset);
		__device__ virtual RC Write(const void *buffer, int amount, int64 offset);
		__device__ virtual RC Truncate(int64 size);
		__device__ virtual RC Close_();
		__device__ virtual RC Sync(SYNC flags);
		__device__ virtual RC get_FileSize(int64 &size);
	};

	__device__ RC MemoryVFile::Read(void *buffer, int amount, int64 offset)
	{
		// SQLite never tries to read past the end of a rollback journal file
		_assert(offset + amount <= _endpoint.Offset);
		FileChunk *chunk;
		if (_readpoint.Offset != offset || offset == 0)
		{
			int64 offset2 = 0;
			for (chunk = First; _ALWAYS(chunk) && (offset2 + JOURNAL_CHUNKSIZE) <= offset; chunk = chunk->Next)
				offset2 += JOURNAL_CHUNKSIZE;
		}
		else
			chunk = _readpoint.Chunk;
		int chunkOffset = (int)(offset % JOURNAL_CHUNKSIZE);
		uint8 *out = (uint8 *)buffer;
		int read = amount;
		do
		{
			int space = JOURNAL_CHUNKSIZE - chunkOffset;
			int copy = MIN(read, (JOURNAL_CHUNKSIZE - chunkOffset));
			_memcpy(out, &chunk->Chunk[chunkOffset], copy);
			out += copy;
			read -= space;
			chunkOffset = 0;
		} while (read >= 0 && (chunk = chunk->Next) && read > 0);
		_readpoint.Offset = offset + amount;
		_readpoint.Chunk = chunk;
		return RC_OK;
	}

	__device__ RC MemoryVFile::Write(const void *buffer, int amount, int64 offset)
	{
		// An in-memory journal file should only ever be appended to. Random access writes are not required by sqlite.
		_assert(offset == _endpoint.Offset);
		uint8 *b = (uint8 *)buffer;
		while (amount > 0)
		{
			FileChunk *chunk = _endpoint.Chunk;
			int chunkOffset = (int)(_endpoint.Offset % JOURNAL_CHUNKSIZE);
			int space = MIN(amount, JOURNAL_CHUNKSIZE - chunkOffset);
			if (chunkOffset == 0)
			{
				// New chunk is required to extend the file
				FileChunk *newChunk = (FileChunk *)_alloc(sizeof(FileChunk));
				if (!newChunk)
					return RC_IOERR_NOMEM;
				newChunk->Next = nullptr;
				if (chunk) { _assert(First); chunk->Next = newChunk; }
				else { _assert(!First); First = newChunk; }
				_endpoint.Chunk = newChunk;
			}
			_memcpy(&_endpoint.Chunk->Chunk[chunkOffset], b, space);
			b += space;
			amount -= space;
			_endpoint.Offset += space;
		}
		return RC_OK;
	}

	__device__ RC MemoryVFile::Truncate(int64 size)
	{
		_assert(size == 0);
		FileChunk *chunk = First;
		while (chunk)
		{
			FileChunk *tmp = chunk;
			chunk = chunk->Next;
			_free(tmp);
		}
		MemoryVFileOpen(this);
		return RC_OK;
	}

	__device__ RC MemoryVFile::Close_()
	{
		Truncate(0);
		Opened = false;
		return RC_OK;
	}

	__device__ RC MemoryVFile::Sync(SYNC flags)
	{
		return RC_OK;
	}

	__device__ RC MemoryVFile::get_FileSize(int64 &size)
	{
		size = (int64)_endpoint.Offset;
		return RC_OK;
	}

	// extensions
	__device__ void VFile::MemoryVFileOpen(VFile *file)
	{
		_assert(_HASALIGNMENT8(file));
		_memset(file, 0, MemoryVFileSize());
		file = new (file) MemoryVFile();
		file->Type = 1;
		file->Opened = true;
	}

	__device__ bool VFile::HasMemoryVFile(VFile *file)
	{
		return (file->Type == 1);
	}

	__device__ int VFile::MemoryVFileSize() 
	{
		return sizeof(MemoryVFile);
	}
}
