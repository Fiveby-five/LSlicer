#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <memory>  
#include <cstdint>
#include <atomic>
#include <thread>
#include <functional> 
#include <fstream>
#include <chrono>

#include <cuda.h>
#include <cuda_runtime.h>
#include "gstDepends.h"

#if defined(_WIN32)
#include <windows.h>
#endif

#define ON_WINDOWS 0
#define ON_LINUX 1

#define UNIFY_MEM 1

namespace Render2{

typedef struct{
    NvBufSurface* surface = nullptr;
    GstBuffer* gstBuffer = nullptr;
    std::atomic<bool> inUse{false};
    uint32_t slotId = 0;
}FrameSlot;

/**
 * @brief Convert bitmap data from gray8 to nv12.
 * 
 * @param grayData Gray bitmap data on GPU. In put
 * @param grayPitch Gray bitmap pitch. Normally same with width
 * @param yPlane y plane data, output
 * @param yPitch y pitch value, if width 3840 then it is 4096, if width 7680 then it is 8192.
 * @param uvPlane uv plane data, output
 * @param uvPitch uv pitch value, if width 3840 then it is 2048, if width 7680 then it is 4096.
 * @param width width of image
 * @param height height of image
 */
__global__
void Gray8ToNV12(const uint8_t* grayData , int grayPitch , uint8_t* yPlane , int yPitch,
                uint8_t* uvPlane , int uvPitch , int width , int height);

/**
 * @brief Host launcher for Gray8ToNV12 kernel.
 *
 * This wrapper hides kernel launch configuration from caller code.
 *
 * @param grayData input GRAY8 device pointer
 * @param grayPitch input pitch in bytes (for current raster output it is usually width)
 * @param yPlane output NV12 Y plane device pointer
 * @param yPitch Y plane pitch in bytes
 * @param uvPlane output NV12 UV plane device pointer (interleaved UVUV...)
 * @param uvPitch UV plane pitch in bytes
 * @param width image width in pixels
 * @param height image height in pixels
 * @param stream CUDA stream used for launch
 */
void LaunchGray8ToNV12(const uint8_t* grayData,
                       int grayPitch,
                       uint8_t* yPlane,
                       int yPitch,
                       uint8_t* uvPlane,
                       int uvPitch,
                       int width,
                       int height,
                       cudaStream_t stream = 0);

class Render2{
    public:
    bool Init(uint32_t width,
              uint32_t height,
              uint32_t poolSize,
              const std::string& outputFile,
              uint32_t frameRate = 60,
              uint32_t bitRate = 8000000,
              uint32_t GPUid = 0,
              NvBufSurfaceLayout layout = NVBUF_LAYOUT_PITCH);

    /**
     * @brief Acquire one writable NV12 surface slot from frame pool.
     *
     * Acquire semantics:
     * 1. Returns immediately when an idle slot is available.
     * 2. If no idle slot and TimeOutMs == 0, returns nullptr immediately.
     * 3. If no idle slot and TimeOutMs > 0, polls until timeout.
     * 4. If no idle slot and TimeOutMs < 0, waits indefinitely (polling mode).
     *
     * State semantics:
     * - A successfully acquired slot is marked inUse=true internally.
     * - Caller must invoke Release(slot) after the slot is no longer used.
     *
     * @param TimeOutMs timeout in milliseconds.
     * @return FrameSlot* valid slot pointer on success, nullptr on timeout or failure.
     */
    FrameSlot* Require(int TimeOutMs = -1);

    /**
     * @brief Release a previously acquired slot back to frame pool.
     *
     * @param slot slot pointer returned by Require().
     */
    void Release(FrameSlot* slot);

    /**
     * @brief Finalize rendering pipeline and release all related resources.
     *
     * One-call shutdown sequence:
     * 1. Drain in-flight frame slots (optional timeout).
     * 2. Send EOS to appsrc.
     * 3. Wait EOS/ERROR from pipeline bus (optional timeout).
     * 4. Set pipeline to NULL and unref GStreamer objects.
     * 5. Destroy all NvBufSurface slots and reset module state.
     *
     * @param DrainTimeoutMs timeout for waiting slot drain.
     *        - 0: check once
     *        - >0: bounded wait
     *        - <0: wait indefinitely
     * @param BusTimeoutMs timeout for waiting EOS/ERROR from gst bus.
     *        - 0: no wait
     *        - >0: bounded wait
     *        - <0: wait indefinitely
     * @return true when shutdown sequence succeeds (or partially succeeds with non-fatal fallback),
     *         false when critical steps fail.
     */
    bool FinalizeAndCleanup(int DrainTimeoutMs = 5000, int BusTimeoutMs = 5000);

    void ShutDown();

    /**
     * @brief Convert one GRAY8 device frame into NV12 and write directly into slot surface.
     *
     * Processing steps:
     * 1. Map surface to access plane pointers.
     * 2. Resolve Y/UV plane pointers and pitch values.
     * 3. Launch Gray8ToNV12 kernel (pixel-level parallel).
     * 4. Synchronize stream and mark surface ready for device consumer.
     * 5. Unmap surface.
     *
     * @param slot acquired slot from Require().
     * @param dGray input GRAY8 device pointer.
     * @param grayPitch input GRAY8 pitch in bytes.
     * @param stream CUDA stream.
     * @return true on success, false on failure.
     */
    bool FillSlotFromGray8(FrameSlot* slot, const uint8_t* dGray, int grayPitch, cudaStream_t stream = 0);

    /**
     * @brief Push one prepared NV12 surface slot into appsrc.
     *
     * Ownership semantics:
     * - This function does not destroy slot->surface.
     * - Slot is automatically released when the GstBuffer is finally unreffed
     *   downstream (asynchronous release), so caller should not call Release()
     *   immediately after successful push.
     *
     * Timestamp semantics:
     * - PTS and duration are filled by internal timeline (ptsNs/frameDurationNs).
     * - On successful push, PTS is advanced to next frame.
     *
     * @param slot prepared slot containing NV12 frame data.
     * @return true if pushed successfully, false on any failure.
     */
    bool PushSlotToAppSrc(FrameSlot* slot);

    /**
     * @brief Batch-end cleanup: wait until all in-flight slots are returned.
     *
     * Typical usage:
     * - Call once after finishing push of one batch.
     * - When return true, all buffers from this batch are already released by
     *   downstream and frame pool is fully reusable.
     *
     * @param TimeOutMs timeout in milliseconds.
     *        - 0: check once and return immediately
     *        - >0: wait up to timeout
     *        - <0: wait indefinitely
     * @param ClearSurface if true, memset all surfaces to zero after drain.
     * @return true when pool drained (and optional clear success), false on timeout/failure.
     */
    bool CleanupBatch(int TimeOutMs = -1, bool ClearSurface = false);

    bool IsInitialized() const { return initialized; }

    GstElement* GetAppSrc() const { return appsrc; }

    uint64_t GetPts() const { return ptsNs; }
    uint64_t GetFrameDuration() const { return frameDurationNs; }
    void AdvancePts() { ptsNs += frameDurationNs; }

    uint32_t Width() const { return pixelWidth; }
    uint32_t Height() const { return pixelHeight; }
    uint32_t PoolSize() const { return static_cast<uint32_t>(slots.size()); }

    protected:
    /**
     * @brief Try to acquire one slot without waiting.
     *
     * @return FrameSlot* slot pointer if available, nullptr otherwise.
     */
    FrameSlot* TryRequire();
    bool InitFramePool(uint32_t width , uint32_t height , uint32_t poolSize , uint32_t GPUid , NvBufSurfaceLayout layout);
    bool InitPipeline(const std::string& outputFile, uint32_t frameRate, uint32_t bitRate, uint32_t width, uint32_t height);

    std::vector<std::unique_ptr<FrameSlot>> slots;
    bool initialized = false;

    uint32_t pixelWidth = 0;
    uint32_t pixelHeight = 0;
    uint32_t gpuId = 0;
    CUcontext cudaCtx = nullptr;
    NvBufSurfaceLayout surfaceLayout = NVBUF_LAYOUT_PITCH;

    GstElement* pipeline = nullptr;
    GstElement* appsrc = nullptr;
    GstBufferPool* nvdsPool = nullptr;

    uint64_t ptsNs = 0;
    uint64_t frameDurationNs = 0;
};

class Render2Process : public Render2 {
    public:

    typedef struct{
        uint64_t Total = 0;
        uint64_t Used = 0;
        uint64_t Free = 0;
        float Usage = 0.0f;
    } MemoryStats;

    /**
     * @brief Initialize the renderer. Only run once
     * 
     * @param width The width of image
     * @param height The height of image
     * @param poolSize How much frames to keep in the pool. Reccommended to be 4
     * @param outputFile Out put dir
     * @param frameRate Frame rate
     * @param bitRate Bit rate
     * @param gpuId GPU ID
     * @param layout Nvbuf param, default is NVBUF_LAYOUT_PITCH
     */
    void InitPipelineAndFramePool(uint32_t width,
                                  uint32_t height,
                                  uint32_t poolSize,
                                  const std::string& outputFile,
                                  uint32_t frameRate = 30,
                                  uint32_t bitRate = 8000000,
                                  uint32_t gpuId = 0,
                                  NvBufSurfaceLayout layout = NVBUF_LAYOUT_PITCH);

    /**
     * @brief Rendering process thread. Designed to run as a thread. Being monitor and take control by exterior thread. 
     * In order to avoid memrory to be full. This thread won't end the pipeline even it ends.
     * 
     * @param d_data Batch of bitmap
     * @param FrameCount How many frames in this batch
     * @param isDone This batch is done
     * @param HoldSignal Recieved para. When recieved true, this thread will stop input bit map to process untill it recieved false.
     * @param Sucess All bit map been processed successfully. When Sucess && isDone == true, means all done and no failure.
     * @param stream Which stream to run on GPU. Default is 0.
     */
    void ProcessThread(uint8_t** d_data , uint32_t FrameCount , std::atomic<bool>& isDone , std::atomic<bool>& HoldSignal , std::atomic<bool>& Sucess ,cudaStream_t stream = 0);

    /**
     * @brief Process a batch of data. Threads allocate and arrange inside.
     * 
     * @param d_batchData Input data
     * @param FrameCount Frames in total
     * @param MemorySpareReserve How many mb you want to reserve free
     * @param stream GPU stream
     */
    void Process(uint8_t** d_batchData , uint32_t FrameCount , uint64_t MemorySpareReserve , cudaStream_t stream = 0);

    /**
     * @brief Monitoring thread. To monitor memory usage. Avoid memory leak.
     * 
     * @param HoldSignal Control the process thread.
     * @param Threshold How much memory you want to leave
     * @param isUnifyMem if this platform has unified memory
     * @param EndSignal Recieve signal to end the thread
     */
    void MonitorringThread(std::atomic<bool>& HoldSignal , uint32_t Threshold , bool isUnifyMem , std::atomic<bool>& EndSignal);

    private:
    
    bool Inited = false;
};

}
