#pragma once

#ifndef ifDEBUG
#define ifDEBUG 1
#endif

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <memory>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>

namespace CudaBus { 

    /**
     * @brief The Triangles struct. Including coordinates and equation of plane, and z range from min to max
     * 
     */
struct __align__(16) Triangles{
    float3 coords[3];  //Coordinates p0 , p1 , p2
    float4 Equation;  //Ax + By + Cz + D = 0
    float2 zRange;  //Min to max
};

    /* The global tag function cannot be set in any class. Must be defined outside as a global member */

    /**
     * @brief Compute the equations of triangles' planes. Save as a vector. Formate [a0, b0, c0, d0, a1, b1, c1, d1, ...]
     * 
     * @param coords The coords input. Formate [x0, y0, z0, x1, y1, z1, ...]. Must be stored in GPU(device)
     * @param normalsOrient The normals orients input. Formate [nx0, ny0, nz0, nx1, ny1, nz1, ...]. Must be stored in GPU(device)
     * @param equations The output. Formate [a0, b0, c0, d0, a1, b1, c1, d1, ...]. Stored in GPU(device). Allocate it first.
     * @param nums The number of triangles.
     * @return Null
     */
    __global__ 
    void PlanesEqsCompution(float *coords, float *normalsOrient, float *equations, size_t nums);

    /**
     * @brief Compute the central point coordinates of triangles. 
     * 
     * @param coords0 The corner point 1 of triangles. Formate [x0, y0, z0, x1, y1, z1, ...]. Must be stored in GPU(device)
     * @param coords1 The corner point 2 of triangles. Formate [x0, y0, z0, x1, y1, z1, ...]. Must be stored in GPU(device)
     * @param coords2 The corner point 3 of triangles. Formate [x0, y0, z0, x1, y1, z1, ...]. Must be stored in GPU(device)
     * @param Central The output pointer, must be allocated in GPU(device).
     * @param num The number of triangles.
     * @return __global__ 
     */
    __global__ 
    void ComputeCentral(float *coords0 , float *coords1 ,float *coords2, float *Central , size_t num);

    /**
     * @brief Sort the plane equations of triangles. Base on the index of z coordinates of central point.
     * 
     * @param Equations Original equations. Must be stored in GPU(device)
     * @param index The index array sorted by z coordinates of central point. Must be stored in GPU(device)
     * @param EqsOut Output array. Must be allocated in GPU(device)
     * @param num Number of triangles
     * @return __global__ 
     */
    __global__ 
    void SortEqs(float *Equations, unsigned int *index, float4 *EqsOut , size_t num);

    /**
     * @brief Sort the coordinates of triangles. Base ont the index which sorted by z coordinates of central point.
     * 
     * @param coords0 Original coordinates. Formate [x0, y0, z0, x1, y1, z1, ...]. Must be stored in GPU(device)
     * @param coords1 Original coordinates. Formate [x0, y0, z0, x1, y1, z1, ...]. Must be stored in GPU(device)
     * @param coords2 Original coordinates. Formate [x0, y0, z0, x1, y1, z1, ...]. Must be stored in GPU(device)
     * @param coord1Out Output coordinates. Formate float3. [Tri0.coord1.x.y.z , Tri1.coord1.x.y.z , Tri2.coord1.x.y.z, ...].Must be allocated in GPU(device)
     * @param coord2Out Output coordinates. Formate float3. [Tri0.coord2.x.y.z , Tri1.coord2.x.y.z , Tri2.coord2.x.y.z, ...].Must be allocated in GPU(device)
     * @param coord3Out Output coordinates. Formate float3. [Tri0.coord3.x.y.z , Tri1.coord3.x.y.z , Tri2.coord3.x.y.z, ...].Must be allocated in GPU(device)
     * @param num The numbers of triangles
     * @param srcIndex The index array sorted by z coordinates of central point. Must be stored in GPU(device)
     * @return __global__ 
     */
    __global__ 
    void SortCoords(float *coords0 ,float *coords1 , float *coords2 , float3 *coord1Out , float3 *coord2Out , float3 *coord3Out , size_t num , unsigned int *srcIndex);


    /**
     * @brief To calculate the range of Z coordinate of triangles. Out put min and max.
     * 
     * @param coords1 Float3. Coordinates of the first vertex. Must be stored in GPU(device).
     * @param coords2 Float3. Coordinates of the second vertex. Must be stored in GPU(device).
     * @param coords3 Float3. Coordinates of the third vertex. Must be stored in GPU(device).
     * @param zRanges Float2. Output array. Must be allocated in GPU(device). Formate [zRange1.x.y(min and max), zRange2.x.y(min and max), ...]
     * @param num The number of triangles
     * @return __global__ 
     */
    __global__ 
    void CalculateZRanges(float3 *coords1 , float3 *coords2 , float3 *coords3 , float2 *zRanges , size_t num);

    /**
     * @brief Intergrated parameters into structure
     * 
     * @param coords1 
     * @param coords2 
     * @param coords3 
     * @param equations 
     * @param zRanges 
     * @param triangles 
     * @param num 
     * @return __global__ 
     */
    __global__
    void IntergradedParams(float3 *coords1 , float3 *coords2 , float3 *coords3 , float4 *equations , float2 *zRanges , Triangles *triangles , size_t num);

class cudaMemBus {
    public:
        void *d_mem_void = nullptr;
        size_t count = 0;

        /**
         * @brief Initialize memory to GPU
         * 
         * @tparam T Type of data
         * @param h_data 
         * @param ValueName The name of the value. To help you figure out which value goes wrong.
         */
        template <typename T>
        void initToGPU(const std::vector<T> &h_data , std::string ValueName = ""){
            count = h_data.size();
            size_t size = count * sizeof(T);

            // Check device state before allocation
            int currentDevice = -1;
            cudaError_t deviceErr = cudaGetDevice(&currentDevice);
            if(deviceErr != cudaSuccess){
                std::cerr << "[ERROR] Cannot get current device before allocation: " << cudaGetErrorString(deviceErr) << std::endl;
            }
            
            // Check device memory info
            size_t freeMem = 0, totalMem = 0;
            cudaError_t memErr = cudaMemGetInfo(&freeMem, &totalMem);
            if(memErr == cudaSuccess){
                std::cerr << "[DEBUG] Device memory: free=" << (freeMem / (1024*1024)) << " MB, total=" << (totalMem / (1024*1024)) << " MB, requested=" << (size / (1024*1024)) << " MB" << std::endl;
            } else {
                std::cerr << "[WARNING] Cannot get device memory info: " << cudaGetErrorString(memErr) << std::endl;
            }

            cudaError_t err = cudaMalloc(&d_mem_void, size);
            std::cout << "Allocate " << size << " bytes to GPU" << std::endl;
            if (err != cudaSuccess) {
                std::cerr << "[ERROR] cudaMalloc failed for " << (ValueName != "" ? ValueName : "unknown") << ": " << cudaGetErrorString(err) << std::endl;
                if(ValueName != ""){
                    std::cout << "Value name: " << ValueName << std::endl;
                }
                
                if(err == cudaErrorDevicesUnavailable){
                    std::cerr << "[ERROR] CUDA device is unavailable. Diagnostic information:" << std::endl;
                    std::cerr << "  Current device: " << currentDevice << std::endl;
                    if(memErr == cudaSuccess){
                        std::cerr << "  Free memory: " << (freeMem / (1024*1024)) << " MB" << std::endl;
                        std::cerr << "  Total memory: " << (totalMem / (1024*1024)) << " MB" << std::endl;
                        std::cerr << "  Requested: " << (size / (1024*1024)) << " MB" << std::endl;
                    }
                    std::cerr << "  Possible causes:" << std::endl;
                    std::cerr << "    1) Device is busy or locked by another process" << std::endl;
                    std::cerr << "    2) Previous program didn't exit cleanly" << std::endl;
                    std::cerr << "    3) cuda-memcheck or other debugging tool is using the device" << std::endl;
                    std::cerr << "    4) Device context is corrupted" << std::endl;
                    std::cerr << "  Troubleshooting:" << std::endl;
                    std::cerr << "    - Check: ps aux | grep -i cuda | grep -v grep" << std::endl;
                    std::cerr << "    - Try running without cuda-memcheck first" << std::endl;
                    std::cerr << "    - Wait a few seconds and retry" << std::endl;
                    std::cerr << "    - If persistent, reboot the device" << std::endl;
                } else if(err == cudaErrorMemoryAllocation){
                    if(memErr == cudaSuccess){
                        std::cerr << "[ERROR] Out of memory. Free: " << (freeMem / (1024*1024)) << " MB, Requested: " << (size / (1024*1024)) << " MB" << std::endl;
                    }
                }
                throw std::runtime_error("Failed to allocate GPU memory: " + std::string(cudaGetErrorString(err)));
            }

            err = cudaMemcpy(d_mem_void, h_data.data(), size, cudaMemcpyHostToDevice);

            if (err != cudaSuccess) {
                if(d_mem_void){
                    cudaFree(d_mem_void);
                    d_mem_void = nullptr;
                }
                throw std::runtime_error("Failed to copy data to GPU.");
            }
        }

        /**
         * @brief Copy data back to host
         * 
         * 
         * @tparam T 
         * @return std::vector<T> 
         */
        template <typename T>
        std::vector<T> CopyBack(){
            std::vector<T> h_data(count);
            cudaError_t err = cudaMemcpy(h_data.data(), d_mem_void, count * sizeof(T), cudaMemcpyDeviceToHost);
            if (err != cudaSuccess){
                throw std::runtime_error("Failed to copy data back to host.");
            }

            return h_data;
        }

        /**
         * @brief Free GPU memory
         * 
         */
        ~cudaMemBus();
        
        cudaMemBus() = default;

        //Forbidden to copy
        /**
         * @brief Construct a new cuda Mem Bus object. But forbidden to copy
         * 
         */
        cudaMemBus(const cudaMemBus&) = delete;

        cudaMemBus& operator=(const cudaMemBus&) = delete;

        //Allow to move
        /**
         * @brief Construct a new cuda Mem Bus object
         * 
         * @param other 
         */
        cudaMemBus(cudaMemBus&& other) noexcept;

        cudaMemBus& operator=(cudaMemBus&& other) noexcept;
};


class SimdComputation{
    public:
    
    
    /**
     * @brief Inline function to get the X coordinate of a point. Coords array format [x0, y0, z0, x1, y1, z1, ...]
     * 
     * @param coords 
     * @param idx 
     * @return __host__ 
     */
    __host__ __device__ __forceinline__
    static float getX(float *coords , size_t idx);

    __host__ __device__ __forceinline__
    static float getY(float *coords , size_t idx);

    __host__ __device__ __forceinline__
    static float getZ(float *coords , size_t idx);

    /**
     * @brief Sort the triangles by the central Z coordinate. Output the sorted index. Up order.
     * 
     * @param Central Central point coordinates. Formate [x0, y0, z0, x1, y1, z1, ...]. Must be stored in GPU(device).
     * @param num The number of triangles(Central points)
     * @param index The output pointer. Must be allocated in GPU(device).
     */
    __host__
    static void SortZidx(float *Central, size_t num , unsigned int *index);

    /**
     * @brief An inline function to find the range of the Z coordinate of the triangle.
     * 
     * @param coord1 Float3. Coordinate of the first vertex.
     * @param coord2 Float3. Coordinate of the second vertex.
     * @param coord3 Float3. Coordinate of the third vertex.
     * @return Min z and Max z.
     */
    __host__ __device__ __forceinline__
    static float2 FindMaxAndMin(float3 coord1 , float3 coord2 , float3 coord3);
};

    class ShareMemory{
        
    };

}
