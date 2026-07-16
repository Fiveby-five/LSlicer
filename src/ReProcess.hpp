#pragma once

#include "ReadSTL.hpp"
#if defined(USE_SLICING_G)
#include "SlicingG.hpp"
#else
#include "ReSlicing.hpp"
#endif
#include <cuda_runtime.h>

#include <vector>
#include <cmath>

/**
 * @brief Calculate the parameters of each segments' const equation
 * 
 * @param Coords0 The first coords of segments
 * @param Coords1 The second coords
 * @param numOfTris How many group of segments in total
 * @param ConstEquation Output. Formate m,n,p
 */
__global__ void ConstEquationKernel(float3* Coords0 , float3* Coords1 , uint32_t numOfTris , float3* ConstEquation);

/**
 * @brief To construct data structure. Formate type TriElements
 * 
 * @param d_Coords0 Coords 0 of triangles. Allocate and save on GPU first
 * @param d_Coords1 Coords 1 of triangles. Allocate and save on GPU first
 * @param d_Coords2 Coords 2 of triangles. Allocate and save on GPU first
 * @param ConstEq0 Const equations m,b,p of first segments. Allocted and saved on GPU first.
 * @param ConstEq1 Const equations m,b,p of second segments. Allocted and saved on GPU first.
 * @param ConstEq2 Const equations m,b,p of second segments. Allocted and saved on GPU first.
 * @param numOfTris How many triangles in total
 * @param d_Triangles Output. Allocted first on GPU
 */
__global__ void IntoStrcKernel(float3* d_Coords0 , float3* d_Coords1 , float3* d_Coords2 ,
                    float3* ConstEq0 , float3* ConstEq1 , float3* ConstEq2 ,
                    uint32_t numOfTris , TriElements* d_Triangles
);

class Processing{
    public:
    static uint32_t NumOfTris;

    ReadSTL readStl;
    
    /**
     * @brief From file read data to GPU memory
     * 
     * @param filename Filename and path
     * @param d_TrisData Define first, parse nullptr. Also output data.
     * @param ResolutionZ Resolution of z axis
     * @param error Error
     */
    void InitProcessing(const char *filename , TriElements* &d_TrisData , float ResolutionZ, float error);

    /**
     * @brief From file read data to GPU memory. Use this in debug mode
     * 
     * @param filename Filename and path
     * @param d_TrisData Define first, parse nullptr. Also output data.
     * @param ResolutionZ Resolution of z axis
     * @param ScaleX Scale of X axis
     * @param ScaleY Scale of Y axis
     * @param error Error
     */
    void InitProcessing(const char *filename, TriElements* &d_TrisData, float ResolutionZ, float ScaleX, float ScaleY , float error);

    /**
     * @brief Slicing main process. Parrallel in layers.
     * 
     * @param d_TrisData 
     * @param d_interxMatrix 
     * @param ResolutionZ 
     * @param numOfLayer 
     */
    void SlicingProcessing(TriElements* d_TrisData , float ResolutionZ , uint32_t numOfLayer , std::vector<std::vector<float3>> &h_Data , float Epsilon);


};
