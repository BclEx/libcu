#include <stddefcu.h>
#include <stdargcu.h>
#include <fcntlcu.h>

__BEGIN_DECLS;

// FILES
#pragma region FILES

typedef struct __align__(8)
{
	int file;				// reference
	unsigned short id;		// ID of author
	unsigned short threadid;// thread ID of author
} fileRef;

__device__ fileRef __iob_fileRefs[CORE_MAXFILESTREAM]; // Start of circular buffer (set up by host)
volatile __device__ fileRef *__iob_freeFilePtr = __iob_fileRefs; // Current atomically-incremented non-wrapped offset
volatile __device__ fileRef *__iob_retnFilePtr = __iob_fileRefs; // Current atomically-incremented non-wrapped offset
__constant__ int __iob_files[CORE_MAXFILESTREAM+3];

static __device__ __forceinline void writeFileRef(fileRef *ref, int f)
{
	ref->file = f;
	ref->id = gridDim.x*blockIdx.y + blockIdx.x;
	ref->threadid = blockDim.x*blockDim.y*threadIdx.z + blockDim.x*threadIdx.y + threadIdx.x;
}

static __device__ int fileGet()
{
	// advance circular buffer
	size_t offset = (atomicAdd((uintptr_t *)&__iob_freeFilePtr, sizeof(fileRef)) - (size_t)&__iob_fileRefs);
	offset %= (sizeof(fileRef)*CORE_MAXFILESTREAM);
	int offsetId = offset / sizeof(fileRef);
	fileRef *ref = (fileRef *)((char *)&__iob_fileRefs + offset);
	int f = ref->file;
	if (!f) {
		f = __iob_files[offsetId+3];
		writeFileRef(ref, f);
	}
	//f->_file = INT_MAX-CORE_MAXFILESTREAM - offsetId;
	return f;
}

static __device__ void fileFree(int f)
{
	if (!f) return;
	// advance circular buffer
	size_t offset = atomicAdd((uintptr_t *)&__iob_retnFilePtr, sizeof(fileRef)) - (size_t)&__iob_fileRefs;
	offset %= (sizeof(fileRef)*CORE_MAXFILESTREAM);
	fileRef *ref = (fileRef *)((char *)&__iob_fileRefs + offset);
	writeFileRef(ref, f);
}

#pragma endregion

//__device__ int fcntlv(const char *file, int oflag, va_list va)
//{
//	panic("Not Implemented");
//	return 0;
//}

__device__ int openv(const char *file, int oflag, va_list va)
{
	panic("Not Implemented");
	return 0;
}

__device__ int creat_(const char *file, int mode)
{
	panic("Not Implemented");
	return 0;
}

__END_DECLS;
