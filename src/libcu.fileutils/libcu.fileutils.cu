#include <sentinel.h>
#include "dcat.cu"
//#include "dchgrp.cu"
//#include "dchmod.cu"
//#include "dchown.cu"
//#include "dcmp.cu"
//#include "dcp.cu"
//#include "dgrep.cu"
//#include "dls.cu"
#include "dmkdir.cu"
//#include "dmore.cu"
//#include "dmv.cu"
//#include "drm.cu"
#include "drmdir.cu"

bool sentinelFileUtilsExecutor(void *tag, sentinelMessage *data, int length);
static sentinelExecutor _fileUtilsExecutor = { nullptr, "fileutils", sentinelFileUtilsExecutor, nullptr };
void sentinelRegisterFileUtils()
{
	sentinelRegisterExecutor(&_fileUtilsExecutor, true, false);
}