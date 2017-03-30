#include <cuda_runtime.h>
#include <stdiocu.h>
#include <crtdefscu.h>
#include <assert.h>

static __global__ void g_crtdefs_test1()
{
	printf("crtdefs_test1\n");
	int a0a = _ROUNDT(3, int); int a0b = _ROUNDT(4, int); int a0c = _ROUNDT(5, int); int a0d = _ROUNDT(-5, int); assert(a0a == 4 && a0b == 4 && a0c == 8 && a0d == 0);
	int a1a = _ROUND8(3); int a1b = _ROUND8(8); int a1c = _ROUND8(9); int a1d = _ROUND8(-5); assert(a1a == 8 && a1b == 8 && a1c == 16 && a1d == 0);
	int a2a = _ROUND64(3); int a2b = _ROUND64(64); int a2c = _ROUND64(65); int a2d = _ROUND64(65); assert(a2a == 64 && a2b == 64 && a2c == 128 && a2d == 0);
	int a3a = _ROUNDN(3, 4); int a3b = _ROUNDN(4, 4); int a3c = _ROUNDN(5, 4); int a3d = _ROUNDN(-5, 4); assert(a3a == 4 && a3b == 4 && a3c == 8 && a3d == 0);
	
	int b0a = _ROUNDDOWN8(3); int b0b = _ROUNDDOWN8(8); int b0c = _ROUNDDOWN8(9); int b0d = _ROUNDDOWN8(-5); assert(b0a == 0 && b0b == 8 && b0c == 8 && b0d == -8);
	int b1a = _ROUNDDOWNN(3, 4); int b1b = _ROUNDDOWNN(4, 4); int b1c = _ROUNDDOWNN(5, 4); int b1d = _ROUNDDOWNN(-5, 4); assert(b1a == 4 && b1b == 4 && b1c == 8, b1d == -8);
	
}
cudaError_t crtdefs_test1() { g_crtdefs_test1<<<1, 1>>>(); return cudaDeviceSynchronize(); }


//* Memory allocation - rounds up to "size" */
//#define _ROUNDN(x, size)	(((size_t)(x)+(size-1))&~(size-1))
//* Memory allocation - rounds down to 8 */
//#define _ROUNDDOWN8(x)		((x)&~7)
//* Memory allocation - rounds down to "size" */
//#define _ROUNDDOWNN(x, size) (((size_t)(x))&~(size-1))
//* Test to see if you are on aligned boundary, affected by BYTEALIGNED4 */
//#ifdef BYTEALIGNED4
//#define HASALIGNMENT8(x) ((((char *)(x) - (char *)0)&3) == 0)
//#else
//#define HASALIGNMENT8(x) ((((char *)(x) - (char *)0)&7) == 0)
//#endif
//* Returns the length of an array at compile time (via math) */
//#define _LENGTHOF(symbol) (sizeof(symbol) / sizeof(symbol[0]))
//* Removes compiler warning for unused parameter(s) */
//#define UNUSED_PARAMETER(x) (void)(x)
//#define UNUSED_PARAMETER2(x,y) (void)(x),(void)(y)