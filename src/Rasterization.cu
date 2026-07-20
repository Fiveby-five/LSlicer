#include "Rasterization.hpp"

namespace Raster{

    __global__
    void ToLinerEquation(float2* d_Segments , float2* d_LinerParas , float2* d_RangesInY , float HorizonMark , float VerticalMark , uint32_t SegmentNum){
        const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx >= SegmentNum){
            return;
        }

        const float2 p0 = d_Segments[idx * 2];
        const float2 p1 = d_Segments[idx * 2 + 1];

        const float minY = fmin(p0.y, p1.y);
        const float maxY = fmax(p0.y, p1.y);
        const float minX = fmin(p0.x, p1.x);
        const float maxX = fmax(p0.x, p1.x);
        d_RangesInY[idx] = make_float2(minY, maxY);

        const float dx = p1.x - p0.x;
        const float dy = p1.y - p0.y;

        // Vertical segment: cannot be represented by y = kx + b
        if(fabsf(dy) <= epsilon){
            d_LinerParas[idx] = make_float2(HorizonMark, minY);
            d_RangesInY[idx] = make_float2(minX, maxX); // Horizon range in x
            return;
        }

        if(fabsf(dx) <= epsilon){
            d_LinerParas[idx] = make_float2(VerticalMark, p0.x);
            return;
        }

        const float k = dy / dx;
        const float b = p0.y - k * p0.x;

        d_LinerParas[idx] = make_float2(k, b);
    }

    __global__
    void RasterizationKernel(float2* d_LinerParas , float2* d_RangesInY , uint8_t* d_bitmap , uint32_t SegmentNum , uint32_t Width, uint32_t Height , float HorizonMark , float VerticalMark){
        const uint32_t y = blockIdx.x * blockDim.x + threadIdx.x;
        if(y >= Height){
            return;
        }

        uint32_t ZoomParamI = static_cast<uint32_t>(ZoomParam);

        uint32_t BaseIdx = y * Width;

        // Record all the intersection points with segments. But just mark
        for(uint32_t Segs = 0 ; Segs < SegmentNum ; Segs++){
            float yScan = static_cast<float>(y) * ZoomParam + (ZoomParam / 2);
            float yScanAreaMin = yScan - ZoomParam / 10.0f;
            float yScanAreaMax = yScan + ZoomParam / 10.0f;

            float yMin = d_RangesInY[Segs].x;
            float yMax = d_RangesInY[Segs].y;
            if((yScanAreaMax < yMin || yScanAreaMin > yMax) && d_LinerParas[Segs].x != HorizonMark){
                continue;
            }

            float xCrossF;
            float k = d_LinerParas[Segs].x;
            float b = d_LinerParas[Segs].y;

            const int yHoriPix = __float2int_rn((b - 0.5f * ZoomParam) / ZoomParam);
            if(k == HorizonMark && static_cast<int>(y) != yHoriPix){
                continue;
            }

            if(k == VerticalMark){
                xCrossF = b;
            }
            if(k == HorizonMark){
                int xStart = __float2int_rn(d_RangesInY[Segs].x / ZoomParam);
                int xEnd   = __float2int_rn(d_RangesInY[Segs].y / ZoomParam);

                if(xStart > xEnd){
                    int t = xStart; xStart = xEnd; xEnd = t;
                }

                xStart = max(0, xStart);
                xEnd   = min(static_cast<int>(Width) - 1, xEnd);

                if(xStart <= xEnd){
                    for(int xi = xStart; xi <= xEnd; ++xi){
                        d_bitmap[BaseIdx + xi] = 128;
                    }
                    d_bitmap[BaseIdx + xStart] = 255;
                d_bitmap[BaseIdx + xEnd] = 255;
                }
                continue;
            }

            if(k != HorizonMark && k != VerticalMark){
                xCrossF = (yScan - b) / k;
            }

            if(xCrossF < 0.0f || xCrossF >= static_cast<float>(Width * ZoomParamI)){
                continue;
            }
            int xCrossI = __float2int_rn(xCrossF / ZoomParam);
            if(xCrossI < 0 || xCrossI >= Width){
                continue;
            }
            d_bitmap[BaseIdx + xCrossI] = 255;
        }

        bool xPrev = false;
        bool xCurrent = false;
        int cooldownRemain = 0;
        for(uint32_t x = 0; x < Width; ++x){
            if(cooldownRemain > 0){
                d_bitmap[BaseIdx + x] = static_cast<uint8_t>(255 * xCurrent);
                --cooldownRemain;
                continue;
            }

            if(d_bitmap[BaseIdx + x] == 128){
                xCurrent = true;
            }

            if(d_bitmap[BaseIdx + x] == 255){
                if(xPrev == false){ xCurrent = true; }
                if(xPrev == true){ xCurrent = false; }
                cooldownRemain = TolerancePixel;
            }

            if(xCurrent != xPrev){
                xPrev = xCurrent;
            }

            d_bitmap[BaseIdx + x] = static_cast<uint8_t>(255 * xCurrent);
        }
    }

}