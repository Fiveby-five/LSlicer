#pragma once

#define MAX_Y 8192
#define ifDEBUG 1

#ifdef ifDEBUG
#include "debug.hpp"
#endif

#include <cuda_runtime.h>
#include "cuda_runtime_api.h"
#include "vector_types.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <array>
#include <utility>
#include <thread>
#include <string>
#include <limits>
#include <set>
#include <cstdlib>

#include <cstdint>
#include <map>

#include <omp.h>

#define ifDEBUG 1

#define ERROR 0.00001f

constexpr float ZoomParam = 10.0f;
constexpr float HorizonMark = 1.0e30f;
constexpr float VerticalMark = 1.0e-30f;
constexpr int TolerancePixel = 1;
constexpr float epsilon = 1.0e-7f;
/* This module is rewrited from original tasterization process. In order to slove the glitch on x direction. */
/* New data structure "uint32_t** OverLappingPoints", a 2 dimension matrix. */
/* First dimension is y coordinate. Second dimension is the overlapping points' x coordinate.*/

namespace Raster{
    /**
     * @brief Process segments into liner equation parameters. GPU side function
     * 
     * @param d_Segments Segments data. type int, must be quantified first.
     * @param d_LinerParas Parameters of liner equation. k and b. Output
     * @param d_Ranges Ranges of segments. In y direction. Output
     */
    __global__ void ToLinerEquation(float2* d_Segments , float2* d_LinerParas , float2* d_RangesInY , float HorizonMark , float VerticalMark , uint32_t SegmentNum);

    /**
     * @brief Rastery scanning in GPU. Once a layer
     * 
     * @param d_LinerParas Parameters of segments' liner equation. Must be storage in GPU.
     * @param d_RangesInY The segments Y ranges. Must be storage in GPU.
     * @param d_Overlappings The array which storage the overlapping points. Must be storage in GPU.
     * @param d_bitmap Output bitmap. Flat storage in row first order.
     */
    __global__ void RasterizationKernel(float2* d_LinerParas , float2* d_RangesInY , uint8_t* d_bitmap , uint32_t SegmentNum , uint32_t Width, uint32_t Height , float HorizonMark , float VerticalMark);

    class Rasterization{
        protected:
        static size_t ThreadSet;
        static uint32_t PixelHeight;
        static uint32_t PixelWidth;
        static std::string OutputPath;
        static float Error;
        static float AutoQuntifyParam;

        public:
        static bool AutoAlternative; // Ture for debugging. Automatically move the origin to the center of the image and quntify to rasterization window
        /**
         * @brief Triangles structure type. Since that the intersection points which are sliced, oringinally from triangles of stl file.
         * So the points are arranged as 3 points a triangle in chaos order. Find each triangle's result first.
         * 
         * @param p0 Triangle's first point.
         * @param p1 Triangle's second point.
         * @param p2 Triangle's third point.
         * @param Type Triangle's type. 0--standard situation, 2 intersection points from one triangle. 1--There is an corner overlapp with slicing plane.
         * 2--The whole triangle is overlapped with slicing plane.
         */
        typedef struct{
            float2 p0;
            float2 p1;
            float2 p2;
            int Type;
        } qTriangles;
        
        float AutoAlternativeAndQuantify(std::vector<std::vector<float3>> &data);

        /**
         * @brief Remove Z coords.
         * 
         * @param data Data from slicing process.
         * @param PlaneData Output data.
         */
        void inTo2D(std::vector<std::vector<float3>> &data , std::vector<std::vector<float2>> &PlaneData);

        void IntoStructrue(std::vector<std::vector<float2>> &data , std::vector<std::vector<qTriangles>> &PlaneDataStrc , float QuntifyParam);

        /**
         * @brief Construct overlapping points matrix.
         * 
         * @param h_triangles Triangles data. In CPU
         * @param d_OverLappings Output overlapping points matrix. Allocated inside this function.
         * First dimension is y coordinate. Second dimension is the overlapping points' x coordinate.
         * if at one pixel, there are lines intersect here, then at position of y rows, it will be recorded to show scanning should flip or not. Every pint only occurs once
         * If it occurs, means it should be fliped.
         * First element of each array is the amount of overlapping points.
         * Attention this function no longer using.
         */
        void ConstructOverlappingPoints(std::vector<qTriangles> &h_triangles , uint32_t** &d_OverLappings);

        /**
         * @brief Get the Type of the triangle data
         * 
         * @param p0 Coords 1
         * @param p1 Coords 2
         * @param p2 Coords 3
         * @return int -1 stand that this triangle has no intersection points, skip. The others keep the same with structure qTriangles.
         */
        static inline int GetType(float2 p0 , float2 p1 , float2 p2);
        
        static inline bool inTolerance(float x , float y , float tolerance);

        /**
         * @brief Initialize parameters. 
         * 
         * @param width Image width in X direction.
         * @param height Image height in Y direction.
         */
        void InitParams(uint32_t width , uint32_t height , float error , bool alternative , uint32_t Threads);

        /**
         * @brief Prepare data each batch for GPU
         * 
         * @param BatchSize Batch size
         * @param BatchNum The order of batch. Started from 0
         * @param h_triangles Data in
         * @param h_SegsOut Data out. Formate{L0P0, L0P1, L1P0, L1P1, ...}
         */
        void ReadyToKernel(uint32_t BatchSize, uint32_t BatchNum, std::vector<std::vector<Rasterization::qTriangles>> &h_triangles, std::vector<std::vector<float2>> &h_SegsOut);
    };

    class RasterProcess: public Rasterization{
        public:
        /**
         * @brief Recieve data from slicing process. In CPU
         * 
         * @param data Slicing result. In CPU
         * @param h_triangles Into structrue. In CPU.
         */
        void RecieveData(std::vector<std::vector<float3>> &data , std::vector<std::vector<qTriangles>> &h_triangles , uint32_t width , uint32_t height , float error ,
            float QuantifyParam,
            uint32_t Threads,
            bool alternative = false,
            const char *OutputPath = "./"
            );

        /**
         * @brief Process flow. The peak GPU memory usage will be decided by Chunk. Divided all data into batch first.
         * 
         * @param h_triangles Data in structrue. In CPU
         * @param d_ChunkBitmaps Output bitmaps. Format: layers and pixels. Allocated first on GPU.
         * @param width Image width in X direction.
         * @param height Image height in Y direction.
         */
        void ProcessData(std::vector<std::vector<qTriangles>> &h_triangles , uint8_t** d_ChunkBitmaps , uint32_t ChunkSize , uint32_t ChunkNum);
    };


}