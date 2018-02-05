#include <ext/memfile.h>
#include <stdlibcu.h>
#include <assert.h>

/* Forward references to internal structures */
typedef struct memfile_t memfile_t;
typedef struct filePoint_t filePoint_t;
typedef struct fileChunk_t fileChunk_t;

/* The rollback journal is composed of a linked list of these structures.
**
** The zChunk array is always at least 8 bytes in size - usually much more. Its actual size is stored in the MemJournal.nChunkSize variable.
*/
struct fileChunk_t {
	fileChunk_t *next;			// Next chunk in the journal
	uint8_t chunk[8];			// Content of this chunk
};

/* By default, allocate this many bytes of memory for each FileChunk object. */
#define MEMFILE_DFLT_FILECHUNKSIZE 1024

/* For chunk size nChunkSize, return the number of bytes that should be allocated for each FileChunk structure. */
#define fileChunkSize(chunkSize) (sizeof(fileChunk_t) + ((chunkSize) - 8))

/* An instance of this object serves as a cursor into the rollback journal. The cursor can be either for reading or writing. */
struct filePoint_t {
	int64_t offset;				// Offset from the beginning of the file
	fileChunk_t *chunk;			// Specific chunk into which cursor points
};

/* This structure is a subclass of sqlite3_file. Each open memory-journal is an instance of this class. */
typedef struct memfile_t {
	//bool opened;
	const void *method;			// Parent class. MUST BE FIRST
	int chunkSize;              // In-memory chunk-size
	int spill;                  // Bytes of data before flushing
	int size;                   // Bytes of data currently in memory
	fileChunk_t *first;			// Head of in-memory chunk-list
	filePoint_t endpoint;		// Pointer to the end of the file
	filePoint_t readpoint;		// Pointer to the end of the last xRead()
} memfile_t;

__constant__ const int __sizeofMemfile_t = sizeof(memfile_t);

__host_device__ void memfileOpen(memfile_t *f)
{
	memset(f, 0, sizeof(memfile_t));
	f->opened = true;
}

#define RC_IOERR 10 
#define RC_IOERR_SHORT_READ (RC_IOERR | (2<<8))
#define RC_IOERR_NOMEM_BKPT 10
#define MIN(a, b) ((a) < (b) ? a : b)

/* Read data from the in-memory journal file.  This is the implementation of the sqlite3_vfs.xRead method. */
__host_device__ int memfileRead(memfile_t *f, void *buf, int amount, int64_t offset)
{
#if defined(ENABLE_ATOMIC_WRITE) || defined(ENABLE_BATCH_ATOMIC_WRITE)
	if (amount + offset > p->endpoint.offset)
		return RC_IOERR_SHORT_READ;
#endif

	// never try to read past the end of an in-memory file
	fileChunk_t *chunk;
	assert(offset + amount <= f->endpoint.offset);
	assert(!f->readpoint.offset || f->readpoint.chunk);
	if (f->readpoint.offset != offset || !offset) { int64_t off = 0; for (chunk = f->first; _ALWAYS(chunk) && (off + f->chunkSize) <= offset; chunk = chunk->next) off += f->chunkSize; }
	else { chunk = f->readpoint.chunk; assert(chunk); }

	int chunkOffset = (int)(offset % f->chunkSize);
	uint8_t *out = (uint8_t *)buf;
	int read = amount;
	do {
		int space = f->chunkSize - chunkOffset;
		int copy = MIN(read, (f->chunkSize - chunkOffset));
		memcpy(out, &chunk->chunk[chunkOffset], copy);
		out += copy;
		read -= space;
		chunkOffset = 0;
	} while (read >= 0 && (chunk = chunk->next) && read > 0);
	f->readpoint.offset = chunk ? offset + amount : 0;
	f->readpoint.chunk = chunk;
	return 0;
}

/* Free the list of FileChunk structures headed at MemJournal.pFirst. */
static __host_device__ void memfileFreeChunks(memfile_t *f)
{
	fileChunk_t *next; for (fileChunk_t *p = f->first; p; p = next) {
		next = p->next;
		free(p);
	} 
	f->first = nullptr;
}

/* Flush the contents of memory to a real file on disk. */
__host_device__ int memfileCreateFile(memfile_t *f)
{
	//memfile_t copy = *p;
	//memset(p, 0, sizeof(memfile_t));
	//int rc = sqlite3OsOpen(copy.vfs, copy.journal, f, copy.flags, 0);
	//if (!rc) {
	//	int chunkSize = copy.chunkSize;
	//	int64_t off = 0;
	//	for (fileChunk_t *p = copy.first; p; p = p->next) {
	//		if (off + chunkSize > copy.endpoint.offset)
	//			chunkSize = copy.endpoint.offset - off;
	//		rc = sqlite3OsWrite(pReal, (uint8_t *)p->chunk, chunkSize, off);
	//		if (rc) break;
	//		off += chunkSize;
	//	}
	//	// No error has occurred. Free the in-memory buffers.
	//	if (!rc) memfileFreeChunks(&copy);
	//}
	//if (rc) {
	//	// If an error occurred while creating or writing to the file, restore the original before returning. This way, SQLite uses the in-memory
	//	// journal data to roll back changes made to the internal page-cache before this function was called.
	//	sqlite3OsClose(pReal);
	//	*p = copy;
	//}
	//return rc;
	return 0;
}

/* Write data to the file. */
__host_device__ int memfileWrite(memfile_t *f, const void *buf, int amount, int64_t offset)
{
	// If the file should be created now, create it and write the new data into the file on disk.
	if (f->spill > 0 && amount + offset > f->spill) {
		int rc = memfileCreateFile(f);
		if (!rc) rc = sqlite3OsWrite(f, buf, amount, offset);
		return rc;
	}
	// If the contents of this write should be stored in memory
	else {
		// An in-memory journal file should only ever be appended to. Random access writes are not required. The only exception to this is when
		// the in-memory journal is being used by a connection using the atomic-write optimization. In this case the first 28 bytes of the
		// journal file may be written as part of committing the transaction.
		assert(offset == f->endpoint.offset || !offset);
#if defined(ENABLE_ATOMIC_WRITE) || defined(ENABLE_BATCH_ATOMIC_WRITE)
		if (!offset && f->first){
			assert(f->chunkSize > amount);
			memcpy((uint8_t *)f->first->chunk, buf, amount);
		} else
#else
		assert(offset > 0 || !f->first);
#endif
		{
			int write = amount;
			uint8_t *b = (uint8_t *)buf;
			while (write > 0) {
				fileChunk_t *chunk = f->endpoint.chunk;
				int chunkOffset = (int)(f->endpoint.offset % f->chunkSize);
				int space = MIN(write, f->chunkSize - chunkOffset);

				if (!chunkOffset) {
					// New chunk is required to extend the file
					fileChunk_t *newChunk = (fileChunk_t *)malloc(fileChunkSize(f->chunkSize));
					if (!newChunk)
						return RC_IOERR_NOMEM_BKPT;
					newChunk->next = nullptr;
					if (chunk) { assert(f->first); chunk->next = newChunk; }
					else { assert(!f->first); f->first = newChunk; }
					f->endpoint.chunk = newChunk;
				}

				memcpy(&f->endpoint.chunk->chunk[chunkOffset], b, space);
				b += space;
				write -= space;
				f->endpoint.offset += space;
			}
			f->size = amount + offset;
		}
	}
	return 0;
}

/* Truncate the file.
**
** If the journal file is already on disk, truncate it there. Or, if it is still in main memory but is being truncated to zero bytes in size, ignore 
*/
__host_device__ int memfileTruncate(memfile_t *f, int64_t size)
{
	if (_ALWAYS(!size)) {
		memfileFreeChunks(f);
		f->size = 0;
		f->endpoint.chunk = nullptr;
		f->endpoint.offset = 0;
		f->readpoint.chunk = nullptr;
		f->readpoint.offset = 0;
	}
	return 0;
}

/* Close the file. */
__host_device__ int memfileClose(memfile_t *f)
{
	memfileFreeChunks(f);
	return 0;
}

/* Sync the file.
**
** If the real file has been created, call its xSync method. Otherwise,  syncing an in-memory journal is a no-op. 
*/
__host_device__ int memfileSync(memfile_t *f, int flags)
{
	UNUSED_SYMBOL2(f, flags);
	return 0;
}

/* Query the size of the file in bytes. */
__device__ int64_t memfileFileSize(memfile_t *f, int64_t *size)
{
	*size = (int64_t)f->endpoint.offset;
	return 0;
}




/* Open a journal file.
**
** The behaviour of the journal file depends on the value of parameter nSpill. If nSpill is 0, then the journal file is always create and 
** accessed using the underlying VFS. If nSpill is less than zero, then all content is always stored in main-memory. Finally, if nSpill is a
** positive value, then the journal file is initially created in-memory but may be flushed to disk later on. In this case the journal file is
** flushed to disk either when it grows larger than nSpill bytes in size, or when sqlite3JournalCreate() is called.
*/
__host_device__ int sqlite3JournalOpen(vsystem *vsys, const char *name, vsys_file *f, int flags, int spill) //: sqlite3JournalOpen
{
	// Zero the file-handle object. If nSpill was passed zero, initialize it using the sqlite3OsOpen() function of the underlying VFS. In this
	// case none of the code in this module is executed as a result of calls made on the journal file-handle.
	memset(f, 0, sizeof(memfile_t));
	if (!spill)
		return sqlite3OsOpen(vsys, name, f, flags, 0);

	if (spill > 0) p->chunkSize = spill;
	else { p->chunkSize = 8 + MEMJOURNAL_DFLT_FILECHUNKSIZE - sizeof(fileChunk_t); assert(MEMJOURNAL_DFLT_FILECHUNKSIZE == fileChunkSize(f->chunkSize)); }

	p->method = (const sqlite3_io_methods*)&MemJournalMethods;
	p->spill = nSpill;
	p->flags = flags;
	p->name = name;
	p->vsys = vsys;
	return RC_OK;
}

/* Open an in-memory journal file. */
void sqlite3MemJournalOpen(sqlite3_file *pJfd){
	sqlite3JournalOpen(0, 0, pJfd, 0, -1);
}

#if defined(SQLITE_ENABLE_ATOMIC_WRITE) || defined(SQLITE_ENABLE_BATCH_ATOMIC_WRITE)
/*
** If the argument p points to a MemJournal structure that is not an 
** in-memory-only journal file (i.e. is one that was opened with a +ve
** nSpill parameter or as SQLITE_OPEN_MAIN_JOURNAL), and the underlying 
** file has not yet been created, create it now.
*/
int sqlite3JournalCreate(sqlite3_file *pJfd){
	int rc = SQLITE_OK;
	MemJournal *p = (MemJournal*)pJfd;
	if( p->pMethod==&MemJournalMethods && (
#ifdef SQLITE_ENABLE_ATOMIC_WRITE
		p->nSpill>0
#else
		/* While this appears to not be possible without ATOMIC_WRITE, the
		** paths are complex, so it seems prudent to leave the test in as
		** a NEVER(), in case our analysis is subtly flawed. */
		NEVER(p->nSpill>0)
#endif
#ifdef SQLITE_ENABLE_BATCH_ATOMIC_WRITE
		|| (p->flags & SQLITE_OPEN_MAIN_JOURNAL)
#endif
		)){
			rc = memjrnlCreateFile(p);
	}
	return rc;
}
#endif

/* The file-handle passed as the only argument is open on a journal file. Return true if this "journal file" is currently stored in heap memory, or false otherwise. */
__host_device__ int sqlite3JournalIsInMemory(sqlite3_file *p)
{
	return p->pMethods == &MemJournalMethods;
}

/* Return the number of bytes required to store a JournalFile that uses vfs pVfs to create the underlying on-disk files. */
__host_device__ int sqlite3JournalSize(sqlite3_vfs *pVfs) //: sqlite3JournalSize
{
	return MAX(pVfs->szOsFile, (int)sizeof(MemJournal));
}