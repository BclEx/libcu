#include <cuda_runtime.h>
#include <stdiocu.h>
#include <crtdefscu.h>
#include <stringcu.h>
#include <assert.h>

static __global__ void g_crtdefs_test1()
{
	printf("crtdefs_test1\n");
	/* Memory allocation - rounds to the type in T */
	int a0a = _ROUNDT(3, int); int a0b = _ROUNDT(4, int); int a0c = _ROUNDT(5, int); int a0d = _ROUNDT(-5, int); assert(a0a == 4 && a0b == 4 && a0c == 8 && a0d == -4);
	/* Memory allocation - rounds up to 8 */	
	int a1a = _ROUND8(3); int a1b = _ROUND8(8); int a1c = _ROUND8(9); int a1d = _ROUND8(-5); assert(a1a == 8 && a1b == 8 && a1c == 16 && a1d == 0);
	/* Memory allocation - rounds up to 64 */
	int a2a = _ROUND64(3); int a2b = _ROUND64(64); int a2c = _ROUND64(65); int a2d = _ROUND64(-65); assert(a2a == 64 && a2b == 64 && a2c == 128 && a2d == -64);
	/* Memory allocation - rounds up to "size" */
	int a3a = _ROUNDN(3, 4); int a3b = _ROUNDN(4, 4); int a3c = _ROUNDN(5, 4); int a3d = _ROUNDN(-5, 4); assert(a3a == 4 && a3b == 4 && a3c == 8 && a3d == -4);

	/* Memory allocation - rounds down to 8 */
	int b0a = _ROUNDDOWN8(3); int b0b = _ROUNDDOWN8(8); int b0c = _ROUNDDOWN8(9); int b0d = _ROUNDDOWN8(-5); assert(b0a == 0 && b0b == 8 && b0c == 8 && b0d == -8);
	/* Memory allocation - rounds down to "size" */
	int b1a = _ROUNDDOWNN(3, 4); int b1b = _ROUNDDOWNN(4, 4); int b1c = _ROUNDDOWNN(5, 4); int b1d = _ROUNDDOWNN(-5, 4); assert(b1a == 0 && b1b == 4 && b1c == 4 && b1d == -8);
	
	/* Test to see if you are on aligned boundary, affected by BYTEALIGNED4 */
	int c0a = _HASALIGNMENT8(3); int c0b = _HASALIGNMENT8(8); int c0c = _HASALIGNMENT8(9); int c0d = _HASALIGNMENT8(-3); assert(!c0a && c0b && !c0c && !c0d);
	/* Returns the length of an array at compile time (via math) */
	int integerArrayOfSixElements[6]; int d0a = _LENGTHOF(integerArrayOfSixElements); assert(d0a == 6);


	/* Determines where you are based on path */
	bool e0a = ISDEVICEPATH("C:\\test"); bool e0b = ISDEVICEPATH("C:/test"); bool e0c = ISDEVICEPATH(":\\test"); bool e0d = ISDEVICEPATH(":/test"); assert(!e0a && !e0b && e0c && e0d);
	strcpy(__cwd, ":\\"); bool e1a = ISDEVICEPATH("."); bool e1b = ISDEVICEPATH("test"); bool e1c = ISDEVICEPATH("\test"); bool e1d = ISDEVICEPATH("/test"); assert(e1a && e1b && e1c && e1d);
	strcpy(__cwd, "\0"); bool e2a = ISDEVICEPATH("."); bool e2b = ISDEVICEPATH("test"); bool e2c = ISDEVICEPATH("\test"); bool e2d = ISDEVICEPATH("/test"); assert(!e2a && !e2b && !e2c && !e2d);
	/* Determines where you are based on number(handle) */
	bool f0a = ISDEVICEHANDLE(1); bool f0b = ISDEVICEHANDLE(INT_MAX-CORE_MAXFILESTREAM); bool f0c = ISDEVICEHANDLE(INT_MAX); assert(!f0a && f0b && f0c);
}
cudaError_t crtdefs_test1() { g_crtdefs_test1<<<1, 1>>>(); return cudaDeviceSynchronize(); }