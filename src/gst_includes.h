#ifndef GST_INCLUDES_H
#define GST_INCLUDES_H

#ifdef _WIN32
    // Windows GStreamer include directories
    // Core GStreamer
    #include "C:/Program Files/gstreamer/1.0/msvc_x86_64/include/gstreamer-1.0/gst/gst.h"
    #include "C:/Program Files/gstreamer/1.0/msvc_x86_64/include/gstreamer-1.0/gst/app/gstappsink.h"
    #include "C:/Program Files/gstreamer/1.0/msvc_x86_64/include/gstreamer-1.0/gst/app/gstappsrc.h"
    #include "C:/Program Files/gstreamer/1.0/msvc_x86_64/include/gstreamer-1.0/gst/video/video.h"
    
    // GStreamer base classes
    #include "C:/Program Files/gstreamer/1.0/msvc_x86_64/include/gstreamer-1.0/gst/base/gstbasesink.h"
    #include "C:/Program Files/gstreamer/1.0/msvc_x86_64/include/gstreamer-1.0/gst/base/gstbasesrc.h"
    
    // GLib core
    #include "C:/Program Files/gstreamer/1.0/msvc_x86_64/include/glib-2.0/glib.h"
    #include "C:/Program Files/gstreamer/1.0/msvc_x86_64/include/glib-2.0/glib-object.h"
    #include "C:/Program Files/gstreamer/1.0/msvc_x86_64/include/glib-2.0/gmodule.h"
    
    // GObject system
    #include "C:/Program Files/gstreamer/1.0/msvc_x86_64/include/glib-2.0/gobject/gobject.h"
    #include "C:/Program Files/gstreamer/1.0/msvc_x86_64/include/glib-2.0/gio/gio.h"
    
    // GLib configuration
    #include "C:/Program Files/gstreamer/1.0/msvc_x86_64/lib/glib-2.0/include/glibconfig.h"
    // NVIDIA headers: prefer old Jetson API first.
    #if defined(__has_include)
        #if __has_include(<nvbuf_utils.h>)
            #include <nvbuf_utils.h>
        #elif __has_include(<nvbufsurface.h>)
            #include <nvbufsurface.h>
            #if __has_include(<nvbufsurftransform.h>)
                #include <nvbufsurftransform.h>
            #endif
        #endif
    #endif
    
#else
    // Linux/Unix GStreamer paths (standard pkg-config locations)
    #include <gst/gst.h>
    #include <gst/app/gstappsink.h>
    #include <gst/app/gstappsrc.h>
    #include <gst/video/video.h>
    #include <gst/base/gstbasesink.h>
    #include <gst/base/gstbasesrc.h>

    // NVIDIA headers: prefer old Jetson API first.
    #if defined(__has_include)
        #if __has_include(<nvbuf_utils.h>)
            #include <nvbuf_utils.h>
        #elif __has_include(<nvbufsurface.h>)
            #include <nvbufsurface.h>
            #if __has_include(<nvbufsurftransform.h>)
                #include <nvbufsurftransform.h>
            #endif
        #else
            #error "NVIDIA multimedia headers not found: need nvbuf_utils.h or nvbufsurface.h"
        #endif
    #else
        #include <nvbuf_utils.h>
    #endif
#endif

#endif // GST_INCLUDES_H
