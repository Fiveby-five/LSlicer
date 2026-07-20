#include "errorRepair.hpp"

__global__
void FirstDetectErrorLines(uint8_t** d_BitmapBatch , ErrorLineInfo* d_ErrorLocation , 
                            uint32_t width , uint32_t height , uint8_t* d_extremLayer){
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    uint32_t totalErrorLines = 0;
    uint32_t ErrorCapacityLayer = (height * RemainForErrLine) / 100;
    uint32_t ErrorBaseIdx = ErrorCapacityLayer * idx;
    for(uint32_t i = 0 ; i < height ; i++){
        if(d_BitmapBatch[idx][(i+1) * width - 1] == 255){
            d_ErrorLocation[ErrorBaseIdx + totalErrorLines].LayerIdx = idx;
            d_ErrorLocation[ErrorBaseIdx + totalErrorLines].LineIdx = i;
            d_ErrorLocation[ErrorBaseIdx + totalErrorLines].LineStates = 127;
            totalErrorLines++;
        }
        if(totalErrorLines > ErrorCapacityLayer - 1){
            d_extremLayer[idx] = 255;
            break;
        }
    }
}

__global__
void SecondDetectErrorLines(uint8_t** d_BitmapBatch , ErrorLineInfo* d_ErrorLocation , uint32_t* TotalErrorLines , 
                            uint32_t width , uint32_t height){
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    uint32_t totalErrorLines = *TotalErrorLines;

    uint64_t Summerize = 0;
    uint64_t SubSum1 = 0;
    uint64_t SubSum2 = 0;

    uint32_t ErrorLayerIdx = d_ErrorLocation[idx].LayerIdx;
    uint32_t ErrorLineIdx = d_ErrorLocation[idx].LineIdx;
    uint32_t ReferenceLineIdx;
    for(uint32_t i = 0 ; i < CORRECT_RADIUS ; i++){
        if(d_BitmapBatch[ErrorLayerIdx][(ErrorLineIdx + i) * width - 1] != 255 && ErrorLineIdx + i < height){
            ReferenceLineIdx = ErrorLineIdx + i;
            break;
        }
        if(d_BitmapBatch[ErrorLayerIdx][(ErrorLineIdx - i) * width - 1] != 255 && ErrorLineIdx - i >= 0){
            ReferenceLineIdx = ErrorLineIdx - i;
            break;
        }
    }

    for(uint32_t i = 0 ; i < width ; i++){
        SubSum1 += d_BitmapBatch[ErrorLayerIdx][ReferenceLineIdx * width + i];
        SubSum2 += d_BitmapBatch[ErrorLayerIdx][ErrorLineIdx * width + i];
    }

    Summerize = SubSum2 - SubSum1;
    if(Summerize > SubSum2){
        d_ErrorLocation[idx].LineStates = 255;
    }
}
/*
__global__ 
void RepairErrorLinesEz(uint8_t** d_BitmapBatch , ErrorLineInfo* d_ErrorLocation , 
                                   uint32_t* d_LayerStartIdx , uint32_t* d_LayerIdx , uint32_t* d_ErrorLinesCount,
                                   uint32_t width , uint32_t height , uint32_t** d_MiddleLine){ 
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;  // Parrelel by frame.

    uint32_t ErrorLinesStartIdx = d_LayerStartIdx[idx]; // Where the 
    uint32_t ErrorLinesCount = d_ErrorLinesCount[idx];

    uint32_t StartErrorAt = d_ErrorLocation[ErrorLinesStartIdx].LineIdx;
    uint32_t CurrentLayerIdx = d_LayerIdx[idx];

    if(ErrorLinesCount >= EZ_ERROR_LINES){
        goto END;
    }

    if(StartErrorAt == 0 || StartErrorAt == height - 1){
        for(uint32_t i = 0 ; i < width ; i++){
            d_BitmapBatch[d_LayerIdx[idx]][d_ErrorLocation[ErrorLinesStartIdx].LineIdx * width + i] = 0;
        }
        goto END;
    }

    for(uint32_t i = 0 ; i < ErrorLinesCount ; i++){ // For Frame
        uint32_t FirstRefLineIdx = d_ErrorLocation[ErrorLinesStartIdx + i].LineIdx - 1;
        uint32_t SecondRefLineIdx;
        uint32_t Distance = 0;

        for(uint32_t j = 1 ; j < CORRECT_RADIUS ; j++){
            if(d_BitmapBatch[CurrentLayerIdx][(FirstRefLineIdx + 1 + j) * width - 1] != 255 && FirstRefLineIdx + 1 + j < height){
                SecondRefLineIdx = FirstRefLineIdx + 1 + j;
                Distance = j;
                break;
            }
        }

        for(uint32_t k = 0 ; k < width ; k++){
            d_MiddleLine[idx][k] = (d_BitmapBatch[CurrentLayerIdx][FirstRefLineIdx * width + k] + d_BitmapBatch[CurrentLayerIdx][SecondRefLineIdx * width + k] / Distance + 1);
        }

        uint32_t TransitionCount = 0;
        uint32_t TransitionCountPrev = 0;
        bool TransistFlag;
        for(uint32_t p = 0 ; p < width + 1 ; p++){ // For single row
            if(d_MiddleLine[idx][p] != d_MiddleLine[idx][p + 1]){
                TransistFlag = ~TransistFlag;
            }
            if(d_MiddleLine[idx][p] > 255 && d_MiddleLine[idx][p] < 510 && TransistFlag == true){
                TransitionCount++;
            }
            if(TransitionCount == TransitionCountPrev){
                goto RebuildRow;
            }
            TransitionCountPrev = TransitionCount;

            RebuildRow:
            uint32_t NormalSummary = TransitionCount * 2 * 255;
            uint32_t Summary = TransitionCount * d_MiddleLine[idx][p - 1];
            uint32_t EdgeLenth = TransitionCount * (Summary / NormalSummary);
            if(d_MiddleLine[idx][p] == 0){
                for(uint32_t y = 0; y < TransitionCount ; y++){
                    d_MiddleLine[idx][p - y] = 0;
                }
                for(uint32_t t = 0; t < EdgeLenth; t++){
                    d_MiddleLine[idx][p - t] = 256;
                }
            }
            if(d_MiddleLine[idx][p] > 509){
                for(uint32_t y = 0; y < TransitionCount; y++){
                    d_MiddleLine[idx][p - TransitionCount + y] = 0;
                }
                for(uint32_t t = 0; t < EdgeLenth; t++){
                    d_MiddleLine[idx][p - EdgeLenth + t] = 256;
                }
            }
            
        }

        for(uint32_t r = 0; r < width; r++){
            if(d_MiddleLine[idx][r] > 255){
                d_BitmapBatch[d_LayerIdx[idx]][r] = 255;
            }
            d_MiddleLine[idx][r] = 0;
        }

    }

    END:
    return;
}
*/


__global__
void RepairErrorLinesEz(uint8_t** d_BitmapBatch,
                        ErrorLineInfo* d_ErrorLocation,
                        uint32_t* d_LayerStartIdx,
                        uint32_t* d_LayerIdx,
                        uint32_t* d_ErrorLinesCount,
                        uint32_t width,
                        uint32_t height,
                        uint32_t** d_MiddleLine)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    /*
     * Preconditions:
     * 1. Kernel launch size exactly matches error-layer group count.
     * 2. d_ErrorLocation has already been memory-shrunk.
     * 3. d_ErrorLocation is grouped by layer.
     * 4. Error rows inside each layer are sorted from bottom to top.
     * 5. d_BitmapBatch is a device-side pointer table.
     */

    if(width == 0 || height == 0){
        return;
    }

    uint32_t errorLinesStartIdx = d_LayerStartIdx[idx];
    uint32_t errorLinesCount = d_ErrorLinesCount[idx];

    if(errorLinesCount == 0){
        return;
    }

    /*
     * Easy repair only.
     * Heavy layers are intentionally skipped.
     */
    if(errorLinesCount >= EZ_ERROR_LINES){
        return;
    }

    uint32_t currentLayerIdx = d_LayerIdx[idx];

    /*
     * This is only a pointer alias, not a bitmap copy.
     * The function directly reads and writes the original batch bitmap.
     */
    uint8_t* bitmap = d_BitmapBatch[currentLayerIdx];
    uint32_t* middleLine = d_MiddleLine[idx];

    if(bitmap == nullptr || middleLine == nullptr){
        return;
    }

    for(uint32_t i = 0; i < errorLinesCount; ++i){
        ErrorLineInfo currentError = d_ErrorLocation[errorLinesStartIdx + i];

        if(currentError.LineStates != 255){
            continue;
        }

        uint32_t errorY = currentError.LineIdx;

        if(errorY >= height){
            continue;
        }

        uint32_t errorBase = errorY * width;

        /*
         * Boundary rows do not have enough repair context.
         * Easy path clears them directly.
         */
        if(errorY == 0 || errorY == height - 1){
            for(uint32_t x = 0; x < width; ++x){
                bitmap[errorBase + x] = 0;
                middleLine[x] = 0;
            }
            continue;
        }

        /*
         * Bottom-up repair:
         * lowerY is always the immediate lower row.
         * Because rows are repaired from bottom to top, this row is assumed
         * to be normal or already repaired.
         */
        uint32_t lowerY = errorY - 1;
        uint32_t lowerBase = lowerY * width;

        /*
         * Find upper reference row using sorted ErrorLineInfo.
         *
         * The original idea:
         *   We do not scan bitmap pixels to judge whether a row is normal.
         *   Since the error list is already shrunk, grouped, and sorted,
         *   we only need to skip the continuous error rows above current row.
         *
         * Example:
         *   current errorY = 10
         *   next error rows = 11, 12
         *   then upper reference row = 13
         *
         * If next error row is not adjacent:
         *   current errorY = 10
         *   next errorY = 15
         *   then upper reference row = 11
         */
        uint32_t upperY = errorY + 1;
        uint32_t checkIdx = i + 1;

        while(checkIdx < errorLinesCount){
            ErrorLineInfo nextError = d_ErrorLocation[errorLinesStartIdx + checkIdx];

            if(nextError.LineIdx == upperY){
                /*
                 * The candidate upper row is also an error row.
                 * Skip it and continue searching upward.
                 */
                ++upperY;
                ++checkIdx;
                continue;
            }

            /*
             * Since rows are sorted from bottom to top, once the next error
             * row is above upperY, upperY itself is normal.
             */
            break;
        }

        /*
         * If the upper reference row is outside image range or too far away,
         * this easy repair path falls back to copying lower row.
         */
        if(upperY >= height || upperY <= errorY || (upperY - errorY) > CORRECT_RADIUS){
            for(uint32_t x = 0; x < width; ++x){
                bitmap[errorBase + x] = bitmap[lowerBase + x];
                middleLine[x] = 0;
            }
            continue;
        }

        uint32_t upperDistance = upperY - errorY;
        uint32_t upperBase = upperY * width;

        /*
         * Build middle repair row.
         *
         * Repair concept:
         *   lower row is adjacent and trusted, so its weight is always 1.
         *   upper row may be farther away, so its weight is 1 / upperDistance.
         *
         * middleLine value meaning:
         *   0              -> no solid influence
         *   255            -> lower row alone exists
         *   510            -> lower and adjacent upper both exist
         *   255 < v < 510  -> transition area
         *
         * Since upperValue is divided by upperDistance:
         *   if upperDistance == 1, upper contribution is 255
         *   if upperDistance > 1, upper contribution is weaker
         */
        for(uint32_t x = 0; x < width; ++x){
            uint32_t lowerValue = bitmap[lowerBase + x];
            uint32_t upperValue = bitmap[upperBase + x];

            middleLine[x] = lowerValue + upperValue / upperDistance;

            /*
             * Clear target row first. Rebuild pass writes repaired pixels.
             */
            bitmap[errorBase + x] = 0;
        }

        /*
         * Scan middleLine.
         * Stable 510 pixels are kept directly.
         * Continuous transition ranges are repaired in RebuildRow.
         */
        uint32_t p = 0;

        uint32_t transitionStart = 0;
        uint32_t transitionEnd = 0;
        uint32_t transitionCount = 0;
        uint32_t transitionSum = 0;

        bool leftSolid = false;
        uint32_t keepPixels = 0;

    ScanMiddleLine:
        if(p >= width){
            goto FinishCurrentErrorRow;
        }

        if(middleLine[p] == 510u){
            bitmap[errorBase + p] = 255;
            ++p;
            goto ScanMiddleLine;
        }

        if(middleLine[p] > 255u && middleLine[p] < 510u){
            transitionStart = p;
            transitionSum = 0;
            transitionCount = 0;

            while(p < width && middleLine[p] > 255u && middleLine[p] < 510u){
                transitionSum += middleLine[p];
                ++transitionCount;
                ++p;
            }

            transitionEnd = p;
            goto RebuildRow;
        }

        /*
         * 0 or 255 is not directly kept in this repair model.
         * 255 means lower row alone exists, but without upper support it is
         * treated as insufficient for the reconstructed error row.
         */
        bitmap[errorBase + p] = 0;
        ++p;
        goto ScanMiddleLine;

    RebuildRow:
        /*
         * A transition segment has been found:
         *   [transitionStart, transitionEnd)
         *
         * The amount to keep is derived by:
         *
         *   keepPixels = transitionSum / (510 * transitionCount) * transitionCount
         *
         * which simplifies to:
         *
         *   keepPixels = transitionSum / 510
         *
         * The +255 term is integer rounding.
         */
        keepPixels = (transitionSum + 255u) / 510u;

        if(keepPixels > transitionCount){
            keepPixels = transitionCount;
        }

        /*
         * Decide which side of the transition band should be preserved.
         *
         * If the left side is already stable solid 510, this transition is
         * likely an exiting edge, so keep pixels from the left.
         *
         * Otherwise, treat it as an entering edge and keep pixels near the
         * right side, close to the next stable solid region.
         */
        leftSolid = false;

        if(transitionStart > 0 && middleLine[transitionStart - 1] == 510u){
            leftSolid = true;
        }

        for(uint32_t t = 0; t < transitionCount; ++t){
            uint32_t outX = transitionStart + t;
            bitmap[errorBase + outX] = 0;
        }

        if(leftSolid){
            for(uint32_t t = 0; t < keepPixels; ++t){
                uint32_t outX = transitionStart + t;
                if(outX < transitionEnd){
                    bitmap[errorBase + outX] = 255;
                }
            }
        }else{
            for(uint32_t t = 0; t < keepPixels; ++t){
                uint32_t outX = transitionEnd - 1u - t;
                if(outX >= transitionStart && outX < transitionEnd){
                    bitmap[errorBase + outX] = 255;
                }
            }
        }

        goto ScanMiddleLine;

    FinishCurrentErrorRow:
        /*
         * Clear temporary line buffer.
         * This does not clear the bitmap. It only clears the middle repair row.
         */
        for(uint32_t x = 0; x < width; ++x){
            middleLine[x] = 0;
        }
    }
}