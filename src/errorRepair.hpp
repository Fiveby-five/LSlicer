#include <cuda_runtime.h>

#include <vector>

/* Error repair functions for srror flipping lines */
/* These is a full device side function, input and output stay in a same memory and pointer */

constexpr uint32_t CORRECT_RADIUS = 10;
constexpr uint32_t EZ_ERROR_LINES = 10;

typedef struct{
    uint32_t LayerIdx;
    uint32_t LineIdx;
    uint8_t LineStates;
} ErrorLineInfo;

/** @brief Detects error lines in a batch of bitmaps。
 *  @param d_BitmapBatch Pointer to the batch of bitmaps
 *  @param d_ErrorLocation Pointer to the array of error locations and states.
 *  @param width Width of each bitmap
 *  @param height Height of each bitmap
 *  @param chunkSize Size of each chunk
 *  @attention Kernel parrelize by layer. Every thread responsible for one frame.
 */
__global__ void FirstDetectErrorLines(uint8_t** d_BitmapBatch , ErrorLineInfo* d_ErrorLocation , 
    uint32_t width , uint32_t height , uint32_t chunkSize , uint32_t* TotalErrorLines);


/**
 * @brief Detects error lines in a batch of bitmaps (second pass), to sure the line is actually an error line.
 * @param d_BitmapBatch Pointer to the batch of bitmaps
 * @param d_ErrorLocation Pointer to the array of error locations and states.
 * @attention Kernel parrelize by error line. Every thread responsible for one error line.
 */
__global__ void SecondDetectErrorLines(uint8_t** d_BitmapBatch , ErrorLineInfo* d_ErrorLocation , uint32_t* TotalErrorLines , 
                            uint32_t width , uint32_t height);

/**
 * @brief Shrinks the memory of error location array to only keep the valid error lines.
 * @param d_ErrorLocation Pointer to the array of error locations and states, original size is all the lines in the batch. On GPU.
 * @param TotalErrorLines Total number of error lines found before.
 * @param d_ShrinkedErrorLocation Pointer to the array of error locations and states, new size is only the valid error lines.
 * @note This fuction is not only for first detection. TotalErrorLines means valid error lines.
 */
void MemoryShrink(ErrorLineInfo* d_ErrorLocation , uint32_t TotalErrorLines , ErrorLineInfo* d_ShrinkedErrorLocation);


/**
 * @brief Finds the starting index of each layer in the error location array.
 * @param d_ErrorLocation Pointer to the array of error locations and states. On GPU.
 * @param TotalErrorLines Total number of error lines found.
 * @param d_LayerStartIdx Pointer to the array of starting indices for each layer in d_ErrorLocation. On GPU.
 * @param d_LayerIdx Pointer to the array of layer indices corresponding to each error line in d_ErrorLocation. On GPU.
 * @param ErrorLinesCount Pointer to the array of counts of error lines for each layer in d_ErrorLocation. On GPU.
 */
void MemoryFindLayer(ErrorLineInfo* d_ErrorLocation , uint32_t TotalErrorLines , uint32_t* d_LayerStartIdx , uint32_t* d_LayerIdx , uint32_t* d_ErrorLinesCount);

/** @brief Detects and repairs error lines in a batch of bitmaps. Wrap the cuda kernel calls and memory management.
 *  @param d_BitmapBatch Pointer to the batch of bitmaps
 *  @param width Width of each bitmap
 *  @param height Height of each bitmap
 *  @param chunkSize Size of each chunk
 *  @param d_ErrorLocation Pointer to the array of error locations and states
 *  @param TotalErrorLines Pointer to the total number of error lines found
 */
void DetectErrorLines(uint8_t** d_BitmapBatch , uint32_t width , uint32_t height , uint32_t chunkSize ,
     ErrorLineInfo* d_ErrorLocation , uint32_t* TotalErrorLines);


/**
 * @brief Repairs the error lines in a batch of bitmaps.
 * @param d_BitmapBatch Pointer to the batch of bitmaps
 * @param d_ErrorLocation Pointer to the array of error locations and states
 * @param d_LayerStartIdx Pointer to the array of starting indices for each layer, for d_ErrorLocation.
 * @param d_LayerIdx Pointer to the array of layer indices corresponding to each error line, for d_ErrorLocation.
 * @param d_ErrorLinesCount Pointer to the array of counts of error lines for each layer, for d_ErrorLocation.
 * @param width Width of each bitmap
 * @param height Height of each bitmap
 * @param d_MiddleLine Used for rebuild the error lines. Middle value when calculate the average of two reference lines. 
 * Allocted on GPU first with the size of width.
 * @attention Kernel parrelize by frames. Every thread responsible for one error frame.
 */
__global__ void RepairErrorLinesEz(uint8_t** d_BitmapBatch , ErrorLineInfo* d_ErrorLocation , 
                                   uint32_t* d_LayerStartIdx , uint32_t* d_LayerIdx , uint32_t* d_ErrorLinesCount,
                                   uint32_t width , uint32_t height , uint32_t** d_MiddleLine);



void RepairProcess(uint8_t** d_BitmapBatch , uint32_t width , uint32_t height , uint32_t chunkSize);



