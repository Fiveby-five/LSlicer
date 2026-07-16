#pragma once

#include <cuda_runtime.h>
#include <cstdint>

/*
 * SlicingG is the Gaussian-enhanced replacement experiment for the original
 * ReSlicing kernels. The Gaussian LUT is used only for layer-hit
 * classification. Real intersection coordinates are still computed in the
 * original geometric coordinate space.
 */

typedef struct{
        float3 Coords0;
        float3 Coords1;
        float3 Coords2;
        float3 ConstEquation0;
        float3 ConstEquation1;
        float3 ConstEquation2;
        float2 zRange;
} TriElements;

// Host-side LUT state. Ownership/cleanup is intentionally left to the process
// controller that initializes the slicing pipeline.
extern uint32_t LutSize;
extern float stepLenth;
extern float *d_GaussianLut;

// Device-side mirror of the LUT state, updated by GaussianLutGenerate().
extern __device__ uint32_t g_LutSizeDev;
extern __device__ float g_StepLengthDev;
extern __device__ float g_GaussianAmpDev;
extern __device__ float* g_GaussianLutDev;

/**
 * @brief Get the intersection point coordinates in original geometry space.
 */
__device__ __forceinline__
float3 GetInterx(float3 ConstEq,
                 float3 Coords0,
                 float3 Coords1,
                 float zPlane,
                 int* Situation,
                 float Epsilon);

/**
 * @brief Slicing the model on GPU. Once process the whole model. Parallel in layers.
 */
__global__
void ParallelSlicingKernel(TriElements *d_TrisData,
                           float3 *d_interxMatrix,
                           uint32_t *d_TriIdxOnLayers,
                           float ResolutionZ,
                           uint32_t numOfLayer,
                           uint32_t NumsOfTris,
                           float Epsilon);

/**
 * @brief Count how many triangles hit each layer.
 */
__global__
void CountingLayerTrisKernel(TriElements *d_TrisData,
                             uint32_t numOfLayer,
                             uint32_t NumsOfTris,
                             uint32_t *d_NumOfTrisOnLayer,
                             float ResolutionZ,
                             float Epsilon);

/**
 * @brief Generate and upload normalized Gaussian LUT for Z hit classification.
 *
 * @param zResolution Z-axis layer height. The table covers roughly one layer
 *        width by default through step = zResolution / LutSize.
 * @param LutSize number of LUT samples. Must be >= 2.
 * @param Sigma Gaussian sigma in normalized table space.
 * @param Zoom local amplification strength. Runtime scale is 1 + Zoom * weight.
 */
void GaussianLutGenerate(float zResolution,
                         uint32_t LutSize,
                         float Sigma = 1.4f,
                         float Zoom = 10.0f);

/**
 * @brief Release Gaussian LUT device memory and reset host/device state.
 */
void GaussianLutRelease();
