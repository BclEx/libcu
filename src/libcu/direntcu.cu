#include <cuda_runtimecu.h>
#include <direntcu.h>

/* Open a directory stream on NAME. Return a DIR stream on the directory, or NULL if it could not be opened. */
__device__ DIR *opendir(const char *name)
{
	return nullptr;
}

/* Close the directory stream DIRP. Return 0 if successful, -1 if not.  */
__device__ int closedir(DIR *dirp)
{
	return 0;
}

/* Read a directory entry from DIRP.  Return a pointer to a `struct dirent' describing the entry, or NULL for EOF or error.  The
storage returned may be overwritten by a later readdir call on the same DIR stream.

If the Large File Support API is selected we have to use the appropriate interface.  */
__device__ struct dirent *readdir(DIR *dirp)
{
	return nullptr;
}

#ifdef __USE_LARGEFILE64
__device__ struct dirent64 *readdir64(DIR *dirp)
{
	return nullptr;
}
#endif

/* Rewind DIRP to the beginning of the directory.  */
__device__ void rewinddir(DIR *dirp)
{
}
