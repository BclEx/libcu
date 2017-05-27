#include <cuda_runtime.h>
#include <stdiocu.h>
#include <sentinel.h>
#include <assert.h>

static __global__ void g_sentinel_test1()
{
	printf("sentinel_test1\n");

	//// SENTINELDEFAULTEXECUTOR ////
	//	extern bool sentinelDefaultExecutor(void *tag, sentinelMessage *data, int length, char *(**hostPrepare)(void*,char*,char*,intptr_t));

	//// SENTINELSERVERINITIALIZE, SENTINELSERVERSHUTDOWN ////
	//	extern void sentinelServerInitialize(sentinelExecutor *executor = nullptr, char *mapHostName = SENTINEL_NAME, bool hostSentinel = true, bool deviceSentinel = true);
	//	extern void sentinelServerShutdown();

	//// SENTINELDEVICESEND ////
	//	extern __device__ void sentinelDeviceSend(sentinelMessage *msg, int msgLength);

	//// SENTINELCLIENTINITIALIZE, SENTINELCLIENTSHUTDOWN ////
	//	extern void sentinelClientInitialize(char *mapHostName = SENTINEL_NAME);
	//	extern void sentinelClientShutdown();

	//// SENTINELCLIENTSEND ////
	//	extern void sentinelClientSend(sentinelMessage *msg, int msgLength);

	//// SENTINELFINDEXECUTOR, SENTINELREGISTEREXECUTOR, SENTINELUNREGISTEREXECUTOR ////
	//	extern sentinelExecutor *sentinelFindExecutor(const char *name, bool forDevice = true);
	//	extern void sentinelRegisterExecutor(sentinelExecutor *exec, bool makeDefault = false, bool forDevice = true);
	//	extern void sentinelUnregisterExecutor(sentinelExecutor *exec, bool forDevice = true);
	
	//// SENTINELREGISTERFILEUTILS ////
	//	extern void sentinelRegisterFileUtils();
}
cudaError_t sentinel_test1() { g_sentinel_test1<<<1, 1>>>(); return cudaDeviceSynchronize(); }