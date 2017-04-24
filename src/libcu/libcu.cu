#include "crtdefscu.cu"
#include "ctypecu.cu"
#include "errnocu.cu"
#include "regexcu.cu"
#include "sentinel-gpu.cu"
#include "setjmpcu.cu"
#include "stdiocu.cu"
#include "stdlibcu.cu"
#include "stringcu.cu"
#include "timecu.cu"
#include "unistdcu.cu"
#include "direntcu.cu"
#include "sys/statcu.cu"
#include "sys/timecu.cu"
#include "ext/hash.cu"
#include "ext/memfile.cu"
#include "grpcu.cu"
#include "pwdcu.cu"
#include "fsystem.cu"
#include "fcntlcu.cu"

__device__ void libcuReset()
{
	fsystemReset();
}