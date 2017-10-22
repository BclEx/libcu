# fixed-assets

Libcu uses fixed-assets to solve two issues. Typically librarys have a need for an initialize, and shutdown method.

files, streams, or other assets can be created on the host or device

## crtdefscu.h

This header includes most of the libcu base definitions.

LIBCU_MAXFILESTREAM defines the maximum number of files and streams as hard-assets
```
#ifndef LIBCU_MAXFILESTREAM
#define LIBCU_MAXFILESTREAM 10
#endif
```

LIBCU_MAXHOSTPTR defines the maximum number of host pointers as hard-assets
```
#ifndef LIBCU_MAXHOSTPTR
#define LIBCU_MAXHOSTPTR 10
#endif
```

```
/* IsDevice support.  */
extern "C" __device__ char __cwd[];
#define ISHOSTPATH(path) ((path)[1] == ':' || ((path)[0] != ':' && __cwd[0] == 0))
#define ISHOSTHANDLE(handle) (handle < INT_MAX-LIBCU_MAXFILESTREAM)
#define ISHOSTPTR(ptr) ((hostptr_t *)(ptr) >= __iob_hostptrs && (hostptr_t *)(ptr) <= __iob_hostptrs+LIBCU_MAXHOSTPTR)
```

```
/* Host pointer support.  */
extern "C" __constant__ hostptr_t __iob_hostptrs[LIBCU_MAXHOSTPTR];
extern "C" __device__ hostptr_t *__hostptrGet(void *host);
extern "C" __device__ void __hostptrFree(hostptr_t *p);
template <typename T> __forceinline __device__ T *newhostptr(T *p) { return (T *)(p ? __hostptrGet(p) : nullptr); }
template <typename T> __forceinline __device__ void freehostptr(T *p) { if (p) __hostptrFree((hostptr_t *)p); }
template <typename T> __forceinline __device__ T *hostptr(T *p) { return (T *)(p ? ((hostptr_t *)p)->host : nullptr); }
```

```
/* Reset library */
extern "C" __device__ void libcuReset();
```


## crtdefscu.cu

```
// HOSTPTRS
__constant__ hostptr_t __iob_hostptrs[LIBCU_MAXHOSTPTR];
```

```
extern __device__ void libcuReset();
```


## fsystem.cu

```
// FILES
__constant__ file_t __iob_files[LIBCU_MAXFILESTREAM];
```

```
__device__ char __cwd[MAX_PATH] = ":\\";
__device__ dirEnt_t __iob_root = { { 0, 0, 0, 1, ":\\" }, nullptr, nullptr };
__device__ hash_t __iob_dir = HASHINIT;
```


## stdiocu.cu

```
// STREAMS
__constant__ FILE __iob_streams[LIBCU_MAXFILESTREAM+3];
```



