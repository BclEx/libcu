#include <sentinel.h>
//#include "dcat.cuh"
//#include "dchgrp.cuh"
//#include "dchmod.cuh"
//#include "dchown.cuh"
#include "dcmp.cuh"
//#include "dcp.cuh"
//#include "dgrep.cuh"
//#include "dls.cuh"
//#include "dmkdir.cuh"
//#include "dmore.cuh"
//#include "dmv.cuh"
//#include "drm.cuh"
//#include "drmdir.cuh"

bool sentinelFileUtilsExecutor(void *tag, sentinelMessage *data, int length);
static sentinelExecutor _fileUtilsExecutor = { nullptr, "fileutils", sentinelFileUtilsExecutor, nullptr };
void sentinelRegisterFileUtils()
{
	sentinelRegisterExecutor(&_fileUtilsExecutor, true, false);
}