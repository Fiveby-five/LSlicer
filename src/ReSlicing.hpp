#pragma once

#include <cuda_runtime.h>

#define SUBCORDINATE_PARAMS 100.0f


/**
 * @brief To storage the triangles's elements. Better for slicing.
 * 
 */
typedef struct{
        float3 Coords0;
        float3 Coords1;
        float3 Coords2;
        float3 ConstEquation0;
        float3 ConstEquation1;
        float3 ConstEquation2;
        float2 zRange;
}TriElements;

/**
 * @brief Get the intersection point's coordination. Device side function.
 * 
 * @param ConstEq Const equation's parameters. m,n,p
 * @param Coords0 The first coord of the segment
 * @param Coords1 The second coord of the segment
 * @param zPlane The slicing plane
 * @param Situation Feed back. 0--No intersection or overflaped with slicing plane
 * @return GetInterx float3 formate. The coordinates of intersection point. When no intersection exist, it will return a NAN,NAN,NAN
 */
__device__ __forceinline__
float3 GetInterx(float3 ConstEq ,float3 Coords0 , float3 Coords1 , float zPlane , int* Situation , float Epsilon);

/**
 * @brief Slicing the model on GPU. Once process the whole model. Parrallel in layers.
 * 
 * @param d_TrisData The data of triangles. An array of strc TriElements. Allocated and saved on GPU first
 * @param d_interxMatrix The data of result. Formate [Layers][Inter]
 * @param d_NumOfTrisOnLayer The data of how many triangles on the slicing plane. Formate [Layers]. Allocated first
 * @param ResolutionZ The slicing resolution on Z axis
 * @param numOfLayer How many layers to slice in total
 * @param NumsOfTris How many triangles in total
 * @attention Will only return 2 intersection points of each triangle, but will keep 3 points a group. And when
 * there is triangle overflaped with slicing plane, it will skip it.
 */
__global__
void ParallelSlicingKernel(TriElements *d_TrisData , float3 *d_interxMatrix , uint32_t *d_TriIdxOnLayers , float ResolutionZ , uint32_t numOfLayer , uint32_t NumsOfTris , float Epsilon);

/**
 * @brief Counting how many triangles in each layer. Kernel function.
 * 
 * @param d_TrisData The data of triangles. An array of strc TriElements. Allocated and saved on GPU first.
 * @param numOfLayer How many layers to slice in total.
 * @param NumsOfTris How many triangles in total.
 * @param d_NumOfTrisOnLayer The data of result. Formate [Layers]. Allocated first.
 * @param ResolutionZ The slicing resolution on Z axis.
 * 
 */
__global__
void CountingLayerTrisKernel(TriElements *d_TrisData , uint32_t numOfLayer , uint32_t NumsOfTris , uint32_t *d_NumOfTrisOnLayer , float ResolutionZ, float Epsilon);

