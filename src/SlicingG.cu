#include "SlicingG.hpp"

__device__ uint32_t g_LutSizeDev = 0;
__device__ float g_StepLengthDev = 0.0f;
__device__ float g_GaussianAmpDev = 0.0f;
__device__ float* g_GaussianLutDev = nullptr;

namespace {

__device__ __forceinline__
float GaussianWeightFromDz(float dzAbs)
{
    if (g_GaussianLutDev == nullptr || g_LutSizeDev < 2u || g_StepLengthDev <= 0.0f) {
        return 0.0f;
    }

    const float centerF = 0.5f * static_cast<float>(g_LutSizeDev - 1u);
    int idx = __float2int_rn(dzAbs / g_StepLengthDev);
    if (idx < 0) idx = 0;

    const int center = static_cast<int>(centerF);
    if (idx > center) {
        return 0.0f;
    }

    const int lutIdx = center + idx;
    const int last = static_cast<int>(g_LutSizeDev) - 1;
    return g_GaussianLutDev[(lutIdx > last) ? last : lutIdx];
}

__device__ __forceinline__
float AmplifiedAbsDz(float z, float zPlane)
{
    const float dz = z - zPlane;
    const float dzAbs = fabsf(dz);
    const float weight = GaussianWeightFromDz(dzAbs);
    return dzAbs * (1.0f + g_GaussianAmpDev * weight);
}

__device__ __forceinline__
bool EdgeHitsPlaneG(float z0, float z1, float zPlane, float epsilon)
{
    const float dz0 = z0 - zPlane;
    const float dz1 = z1 - zPlane;

    // Strict crossing is always accepted. The Gaussian rule only tightens the
    // near-plane tolerance for endpoint-touching cases.
    if ((dz0 <= 0.0f && dz1 >= 0.0f) || (dz1 <= 0.0f && dz0 >= 0.0f)) {
        return true;
    }

    return (AmplifiedAbsDz(z0, zPlane) <= epsilon) ||
           (AmplifiedAbsDz(z1, zPlane) <= epsilon);
}

__device__ __forceinline__
bool TriangleHitsPlaneG(const TriElements& tri, float zPlane, float epsilon)
{
    const float relaxed = epsilon;
    if (tri.zRange.x > zPlane + relaxed || tri.zRange.y < zPlane - relaxed) {
        return false;
    }

    return EdgeHitsPlaneG(tri.Coords0.z, tri.Coords1.z, zPlane, epsilon) ||
           EdgeHitsPlaneG(tri.Coords1.z, tri.Coords2.z, zPlane, epsilon) ||
           EdgeHitsPlaneG(tri.Coords2.z, tri.Coords0.z, zPlane, epsilon);
}

} // namespace

__device__ __forceinline__
float3 GetInterx(float3 ConstEq,
                 float3 Coords0,
                 float3 Coords1,
                 float zPlane,
                 int* Situation,
                 float Epsilon)
{
    if (!EdgeHitsPlaneG(Coords0.z, Coords1.z, zPlane, Epsilon)) {
        *Situation = 0;
        return make_float3(NAN, NAN, NAN);
    }

    const float dz = fabsf(Coords1.z - Coords0.z);
    if (dz <= Epsilon || fabsf(ConstEq.z) <= Epsilon) {
        *Situation = 0;
        return make_float3(NAN, NAN, NAN);
    }

    float t = (zPlane - Coords0.z) / ConstEq.z;
    t = fminf(1.0f, fmaxf(0.0f, t));

    return make_float3(Coords0.x + t * ConstEq.x,
                       Coords0.y + t * ConstEq.y,
                       zPlane);
}

__global__
void ParallelSlicingKernel(TriElements *d_TrisData,
                           float3 *d_interxMatrix,
                           uint32_t *d_TriIdxOnLayers,
                           float ResolutionZ,
                           uint32_t numOfLayer,
                           uint32_t NumsOfTris,
                           float Epsilon)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numOfLayer) {
        return;
    }

    if (idx > 0) {
        if (d_TriIdxOnLayers[idx] == d_TriIdxOnLayers[idx - 1]) return;
    } else {
        if (d_TriIdxOnLayers[idx] == 0) return;
    }

    const float currentZ = idx * ResolutionZ;
    const uint32_t baseIdx = (idx == 0) ? 0u : d_TriIdxOnLayers[idx - 1] * 3u;
    uint32_t hitted = 0;

    for (uint32_t i = 0; i < NumsOfTris; ++i) {
        const TriElements tri = d_TrisData[i];
        if (!TriangleHitsPlaneG(tri, currentZ, Epsilon)) {
            continue;
        }

        int situation1 = 1;
        int situation2 = 1;
        int situation3 = 1;

        float3 intersection0 = GetInterx(tri.ConstEquation0, tri.Coords0, tri.Coords1, currentZ, &situation1, Epsilon);
        float3 intersection1 = GetInterx(tri.ConstEquation1, tri.Coords1, tri.Coords2, currentZ, &situation2, Epsilon);
        float3 intersection2 = GetInterx(tri.ConstEquation2, tri.Coords2, tri.Coords0, currentZ, &situation3, Epsilon);

        const uint32_t currentIdx = baseIdx + hitted * 3u;
        ++hitted;

        if (situation1 == 0) {
            d_interxMatrix[currentIdx] = intersection1;
            d_interxMatrix[currentIdx + 1] = intersection2;
            d_interxMatrix[currentIdx + 2] = make_float3(NAN, NAN, NAN);
            continue;
        }

        if (situation2 == 0) {
            d_interxMatrix[currentIdx] = intersection0;
            d_interxMatrix[currentIdx + 1] = intersection2;
            d_interxMatrix[currentIdx + 2] = make_float3(NAN, NAN, NAN);
            continue;
        }

        if (situation3 == 0) {
            d_interxMatrix[currentIdx] = intersection0;
            d_interxMatrix[currentIdx + 1] = intersection1;
            d_interxMatrix[currentIdx + 2] = make_float3(NAN, NAN, NAN);
            continue;
        }

        d_interxMatrix[currentIdx] = intersection0;
        d_interxMatrix[currentIdx + 1] = intersection1;
        d_interxMatrix[currentIdx + 2] = intersection2;
    }
}

__global__
void CountingLayerTrisKernel(TriElements *d_TrisData,
                             uint32_t numOfLayer,
                             uint32_t NumsOfTris,
                             uint32_t *d_NumOfTrisOnLayer,
                             float ResolutionZ,
                             float Epsilon)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numOfLayer) {
        return;
    }

    const float currentZ = idx * ResolutionZ;
    uint32_t trisOnLayer = 0;

    for (uint32_t i = 0; i < NumsOfTris; ++i) {
        if (TriangleHitsPlaneG(d_TrisData[i], currentZ, Epsilon)) {
            ++trisOnLayer;
        }
    }

    d_NumOfTrisOnLayer[idx] = trisOnLayer;
}
