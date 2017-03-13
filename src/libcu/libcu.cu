#include "ctypecu.cu"
#include "direntcu.cu"
#include "errnocu.cu"
#include "regexcu.cu"
#include "sentinel-gpu.cu"
#include "setjmpcu.cu"
#include "stdiocu.cu"
#include "stdlibcu.cu"
#include "stringcu.cu"
#include "timecu.cu"
#include "unistdcu.cu"
#include "sys/statcu.cu"
#include "ext/hash.cu"
#include "ext/memfile.cu"
#include "fsystem.cu"
#include "fcntlcu.cu"

__device__ void libcuReset()
{
	fsystemReset();
}