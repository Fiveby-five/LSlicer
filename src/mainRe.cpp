
#include <iostream>
#include <string>
#include <csignal>
#include <cstdlib>
#include <algorithm>
#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#include <sys/time.h>
#endif

#include "ReProcess.hpp"
#if defined(USE_SLICING_G)
#include "SlicingG.hpp"
#else
#include "ReSlicing.hpp"
#endif
#include "cmd.hpp"
#include "Render2.hpp"
#include "Rasterization.hpp"

#define ifDEBUG 1 // Using macro to control debug mode on or off
#define SPEED_TEST 0 // Using macro to control speed test mode on or off
// Caution: ifDEBUG and SPEED_TEST cannot be defined on at the same time

#if ifDEBUG == 1 && SPEED_TEST == 1
#error "ifDEBUG and SPEED_TEST cannot be defined on at the same time"
#endif

#if ifDEBUG == 1
#define SPEED_TEST 0
#endif

#if SPEED_TEST == 1
#define ifDEBUG 0
#endif

#if ifDEBUG == 1 || SPEED_TEST == 1
#include "debug.hpp"
#endif

#define LOG_PATH "/"
#define FILE_NAME "debuglog.txt"
#define MAXLAYER 50000
// Remember to define the macro to control debug and output

// Global flag for signal handling
volatile sig_atomic_t g_signal_received = 0;

// Signal handler
void signal_handler(int signal) {
    g_signal_received = signal;
    std::cerr << "\n[ERROR] Received signal " << signal << ", exiting..." << std::endl;
    std::cerr.flush();
    // Force exit
    std::_Exit(1);
}

Cmd cmd;
Cmd::Parameters parameters;

/*_______________________________________________________________________________________________*/

int main(int argc , char *argv[]){ 
    // Register signal handlers for graceful exit
    signal(SIGINT, signal_handler);   // Ctrl+C
    signal(SIGTERM, signal_handler);  // Termination signal
    #ifndef _WIN32
    signal(SIGQUIT, signal_handler); // Quit signal
    #endif
    
    #if ifDEBUG == 1 || SPEED_TEST == 1
    using namespace DeviceDebug;
    DeviceDebug::Debug debug;
    #endif

    std::string ModelName;          // Define Parameter
    std::string OutputPath;         // Define Parameter

    float Resolution;               // Define Parameter
/*_______________________________________________________________________________________________*/
    // Check device availability first with retry
    int deviceCount = 0;
    cudaError_t CudaStatus;
    int retryCount = 0;
    const int MAX_RETRIES = 5;
    const int RETRY_DELAY_MS = 1000; // 1 second
    
    while(retryCount < MAX_RETRIES){
        CudaStatus = cudaGetDeviceCount(&deviceCount);
        if(CudaStatus == cudaSuccess && deviceCount > 0){
            break; // Success
        }
        retryCount++;
        if(retryCount < MAX_RETRIES){
            std::cerr << "Warning: Device check failed, retrying (" << retryCount << "/" << MAX_RETRIES << ")..." << std::endl;
            #ifdef _WIN32
            Sleep(RETRY_DELAY_MS);
            #else
            usleep(RETRY_DELAY_MS * 1000);
            #endif
        }
    }
    
    if(CudaStatus != cudaSuccess || deviceCount == 0){
        std::cerr << "Error: No CUDA devices available after " << MAX_RETRIES << " retries! Error: " << cudaGetErrorString(CudaStatus) << std::endl;
        std::cerr << "Please check if:" << std::endl;
        std::cerr << "  1) Other CUDA programs are running (check with: ps aux | grep cuda)" << std::endl;
        std::cerr << "  2) Previous program didn't exit cleanly (wait a few seconds and retry)" << std::endl;
        std::cerr << "  3) Device needs to be rebooted" << std::endl;
        return -1;
    }
    
    // Get device properties to check memory
    cudaDeviceProp deviceProp;
    CudaStatus = cudaGetDeviceProperties(&deviceProp, 0);
    if(CudaStatus != cudaSuccess){
        std::cerr << "Warning: Failed to get device properties: " << cudaGetErrorString(CudaStatus) << std::endl;
    } else {
        std::cerr << "CUDA Device: " << deviceProp.name << ", Total Memory: " << deviceProp.totalGlobalMem / (1024*1024) << " MB" << std::endl;
    }
    
    // Set device with retry
    retryCount = 0;
    while(retryCount < MAX_RETRIES){
        CudaStatus = cudaSetDevice(0);
        if(CudaStatus == cudaSuccess){
            break; // Success
        }
        retryCount++;
        if(retryCount < MAX_RETRIES){
            std::cerr << "Warning: Failed to set device, retrying (" << retryCount << "/" << MAX_RETRIES << ")..." << std::endl;
            #ifdef _WIN32
            Sleep(RETRY_DELAY_MS);
            #else
            usleep(RETRY_DELAY_MS * 1000);
            #endif
        }
    }
    
    if(CudaStatus != cudaSuccess) {
        std::cerr << "Error: Failed to set CUDA device after " << MAX_RETRIES << " retries: " << cudaGetErrorString(CudaStatus) << std::endl;
        std::cerr << "This may indicate that the device is busy or previous program didn't exit cleanly." << std::endl;
        std::cerr << "Please check if:" << std::endl;
        std::cerr << "  1) Other CUDA programs are running (check with: ps aux | grep cuda)" << std::endl;
        std::cerr << "  2) Previous program didn't exit cleanly (wait a few seconds and retry)" << std::endl;
        std::cerr << "  3) Device needs to be rebooted" << std::endl;
        return -1;
    }

    // Check device memory availability
    size_t freeMem = 0, totalMem = 0;
    CudaStatus = cudaMemGetInfo(&freeMem, &totalMem);
    if(CudaStatus == cudaSuccess){
        std::cerr << "Device memory: free=" << (freeMem / (1024*1024)) << " MB, total=" << (totalMem / (1024*1024)) << " MB" << std::endl;
    } else {
        std::cerr << "Warning: Failed to get device memory info: " << cudaGetErrorString(CudaStatus) << std::endl;
    }

    // Don't call cudaDeviceSynchronize() here - it's not necessary at initialization
    // and may fail if device is busy. We'll synchronize when needed during execution.

/*_______________________________________________________________________________________________*/
    // Check arguments if satisfied
    if(cmd.ValidateArguments(argc , argv , parameters) == false){
        return -1;
    }
    // Create log directory and check
    std::string LogPath = parameters.LogPath;
    if(cmd.EnsureLogDirectory(LogPath) == false){
        std::cerr << "Error: Failed to create log directory" << LogPath << "\n";
    }
    ModelName = parameters.ModelName;
    Resolution = parameters.Resolution;
    OutputPath = parameters.OutputPath;
    float error = parameters.Error;
    uint32_t frameRate = parameters.FrameRate;
    uint32_t bitRate = parameters.BitRate;
    uint32_t width = parameters.Width;
    uint32_t height = parameters.Height;
    float ScaleX = parameters.ScaleX;
    float ScaleY = parameters.ScaleY;
    float ScaleZ = parameters.ScaleZ;
    std::string VideoName = parameters.VideoName;
    uint32_t chunk = parameters.Chunk;
    float QuantifyParam = parameters.QuantifyParam;
    bool AutoAlternative = parameters.AutoAlternative;
    uint32_t threads = parameters.Threads;
    if(chunk == 0){
        std::cerr << "[ERROR] Chunk cannot be zero" << std::endl;
        return -1;
    }

#if !ifDEBUG
    if(AutoAlternative){
        std::cerr << "[WARNING] AutoAlternative is only for debug mode and will be ignored in non-debug builds." << std::endl;
        AutoAlternative = false;
    }
#endif

    std::cout << "Initializing 3D modle slicer (CUDA C++ acceleration) \n";
    std::cout << "Input model: " << ModelName << "\n";
    std::cout << "Resolution: " << Resolution << "\n";
    std::cout << "Output path: " << OutputPath << "\n";
    std::cout << "Log path: " << LogPath << "\n";
    std::cout << "Error: " << error << "\n";
    std::cout << "Frame rate: " << frameRate << "\n";
    std::cout << "Bit rate: " << bitRate << "\n";
    std::cout << "Video Width: " << width << "\n";
    std::cout << "Video Height: " << height << "\n";
    std::cout << "Chunk: " << chunk << "\n";
    std::cout << "Quantify Param: " << QuantifyParam << std::endl;
    std::cout << "AutoAlternative: " << (AutoAlternative ? "ON" : "OFF") << std::endl;

    TriElements* d_TrisData = nullptr;

    /*_____________________________________________________________________________________________*/
    //Start running
    try{
        std::cout << "Loading model...\n";
        Processing process;
        std::vector<std::vector<float3>> h_SlicedData;

        std::cout << "Pre Preprocessing...\n";
        #if ifDEBUG
        process.InitProcessing(ModelName.c_str(), d_TrisData, Resolution, ScaleX, ScaleY , error);
        #else
        process.InitProcessing(ModelName.c_str(), d_TrisData, Resolution , error);
        #endif

        #ifdef ifDEBUG
        std::cerr << "[DEBUG] mainRe: InitProcessing completed, checking device state..." << std::endl;
        cudaError_t postPreErr = cudaGetLastError();
        if(postPreErr != cudaSuccess){
            std::cerr << "[WARNING] mainRe: Device has error after InitProcessing: "
                    << cudaGetErrorString(postPreErr) << std::endl;
            cudaGetLastError();
        }
        std::cerr << "[DEBUG] mainRe: d_TrisData pointer=" << std::hex << d_TrisData << std::dec
                << ", NumOfTris=" << Processing::NumOfTris << std::endl;
        #endif

        if(Resolution <= 0.0f){
            throw std::runtime_error("[ERROR] Resolution must be > 0");
        }
        if(ScaleZ <= 0.0f){
            throw std::runtime_error("[ERROR] ScaleZ must be > 0");
        }

// Start at Z = 0
const uint32_t numOfLayer = static_cast<uint32_t>(std::floor(ScaleZ / Resolution)) + 1u;
if(numOfLayer >= MAXLAYER){
    std::cerr << "[WARNING] The layer count is too high , the maxium layer is " << MAXLAYER << std::endl;
}

std::cout << "Start slicing...\n";
process.SlicingProcessing(d_TrisData, Resolution, numOfLayer, h_SlicedData , error);
std::cout << "Finished slicing...\n";

if(d_TrisData != nullptr){
    cudaFree(d_TrisData);
    d_TrisData = nullptr;
}

uint32_t TotalFrames = static_cast<uint32_t>(h_SlicedData.size());

        Render2::Render2Process Render;
        Raster::RasterProcess rasterProcess;
        std::vector<std::vector<Raster::Rasterization::qTriangles>> h_rasterTriangles;
        std::string OutPutVideoPath = OutputPath + "/" + VideoName;

        Render.InitPipelineAndFramePool(width, height, chunk, OutPutVideoPath, frameRate, bitRate);
        if (!Render.IsInitialized()) {
            std::cerr << "[ERROR] Main: Render initialization failed, stop before batch processing." << std::endl;
            return -1;
        }
        #ifdef ifDEBUG
        std::cout << "[DEBUG] Main: Starting pipeline" << std::endl;
        #endif

        rasterProcess.RecieveData(
            h_SlicedData,
            h_rasterTriangles,
            width,
            height,
            error,
            QuantifyParam,
            threads,
            AutoAlternative,
            parameters.OutputPath.c_str()
        );

        for(uint32_t beginIdx = 0; beginIdx < TotalFrames; beginIdx += chunk){
            const uint32_t endIdx = std::min(beginIdx + chunk, TotalFrames);
            const uint32_t chunkSize = endIdx - beginIdx;
            const uint32_t chunkNum = beginIdx / chunk;

            // ProcessData uses (ChunkNum * ChunkSize) to recover the global begin layer,
            // so ChunkSize passed to it must stay as the configured chunk, even for tail batches.
            // Allocate the pointer table with full chunk capacity; Render.Process below only consumes
            // the real frame count in this batch.
            uint8_t** d_batchBitmap = new uint8_t*[chunk];
            for(uint32_t i = 0; i < chunk; ++i){
                d_batchBitmap[i] = nullptr;
            }

        #ifdef ifDEBUG
            std::cout << "[DEBUG] Main: Processing chunk from layer: "
              << beginIdx << " to " << endIdx << std::endl;
        #endif

            rasterProcess.ProcessData(
                h_rasterTriangles,
                d_batchBitmap,
                chunk,
                chunkNum
            );

        #ifdef ifDEBUG
            std::cout << "[DEBUG] Main: Frame check output from layer: "
                    << beginIdx << " to " << endIdx << std::endl;
            FrameTest testFrame;
            testFrame.DumpMiddleFrameInChunk(
                beginIdx,
                endIdx,
                d_batchBitmap,
                parameters.Width,
                parameters.Height,
                parameters.OutputPath.c_str(),
                "DebugFrame",
                false
            );
        #endif

            Render.Process(d_batchBitmap, chunkSize , 768);
                
            for (uint32_t i = chunkSize; i < chunk; ++i) {
                if (d_batchBitmap[i] != nullptr) {
                    cudaFree(d_batchBitmap[i]);
                    d_batchBitmap[i] = nullptr;
                }
            }
            delete[] d_batchBitmap;
        }
        
        if(!Render.FinalizeAndCleanup()){
            std::cerr << "[WARNING] Render.FinalizeAndCleanup failed" << std::endl;
        }

        /*___________________________________________________________________________________________*/
        // Clean up
        // Explicitly clean up CUDA resources before program exit
        // This prevents issues with static variable destructors, especially on Jetson with unified memory
        #ifdef ifDEBUG
        std::cerr << "[DEBUG] mainRe: Cleaning up CUDA resources..." << std::endl;
        #endif

        if(d_TrisData != nullptr){
            cudaFree(d_TrisData);
            d_TrisData = nullptr;
        }

        #ifdef ifDEBUG
        std::cerr << "[DEBUG] mainRe: Cleanup completed" << std::endl;
        #endif

        // Synchronize device before exit (with timeout check)
        if(g_signal_received == 0){
            cudaError_t syncErr = cudaDeviceSynchronize();
            if(syncErr != cudaSuccess && syncErr != cudaErrorCudartUnloading){
                std::cerr << "[WARNING] cudaDeviceSynchronize failed: " << cudaGetErrorString(syncErr) << std::endl;
                // Don't throw, just log and exit
            }
        } else {
            std::cerr << "[WARNING] Signal received, skipping synchronization" << std::endl;
        }
        
        if(g_signal_received != 0){
            std::cerr << "[ERROR] Program interrupted by signal " << g_signal_received << std::endl;
            return -1;
        }
        
        return 0;
    }

    /*_______________________________________________________________________________________________*/
    // Unnormed error

    catch(const std::exception &e){
        std::cerr << "[ERROR] Exception caught: " << e.what() << std::endl;
        std::cerr.flush();
        // Force cleanup and exit
        try {
            if(d_TrisData != nullptr){
            cudaFree(d_TrisData);
            d_TrisData = nullptr;
        }
        } catch(...) {
            // Ignore cleanup errors
        }
        return -1;
    }

    catch(...){
        std::cerr << "[ERROR] Unknown exception caught" << std::endl;
        std::cerr.flush();
        // Force cleanup and exit
        try {
            if(d_TrisData != nullptr){
            cudaFree(d_TrisData);
            d_TrisData = nullptr;
        }
        } catch(...) {
            // Ignore cleanup errors
        }
        return -1;
    }

    // Should never reach here, but just in case
    std::cerr << "[ERROR] Unexpected program termination" << std::endl;
    std::cerr.flush();
    return -1;
}
