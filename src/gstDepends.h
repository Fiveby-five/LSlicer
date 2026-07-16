#pragma once

#ifdef _WIN32
// On windows
#include "../libs/gstlibs/gst/gst.h"
#include "../libs/gstlibs/gst/app/gstappsrc.h"
#include "../libs/gstlibs/gst/video/video.h"
#include "../libs/gstlibs/ds/includes/nvbufsurface.h"
#include "../libs/gstlibs/ds/includes/nvbufsurftransform.h"
#else
// On jetson
#include <gst/gst.h>
#include <gst/app/gstappsrc.h>
#include <gst/video/video.h>
#include <nvbufsurface.h>
#include <nvbufsurftransform.h>
#endif