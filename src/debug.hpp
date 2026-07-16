#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>  // For isnan
#include <sys/stat.h>  // For directory checking
#include <chrono>
#include <algorithm>
#include <array>
#include <limits>
#include <iomanip>
#include <numeric>
#include <random>
#include <sstream>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include "cudaBus.hpp"

#ifndef LOG_PATH 
#define LOG_PATH "/"
#endif

#ifndef FILE_NAME
#define FILE_NAME "debuglog.txt"
#endif

/**
 * @brief Helper function to join path and filename with proper separator
 * C++14 compatible path joining (no std::filesystem)
 */
inline std::string joinPath(const std::string& path, const std::string& filename) {
    if (path.empty()) {
        return filename;
    }
    // Check if path ends with separator
    if (path.back() == '/' || path.back() == '\\') {
        return path + filename;
    }
    // Add separator if needed
    return path + "/" + filename;
}

/**
 * @brief A simple debug class to help debug data on GPU
 * Attetion this class will cause huge performance decrease, only use it in debug mode.
 * 
 */
namespace DeviceDebug{
    class Debug{
        public:
            Debug(){
                std::cout << "Debug mode is on" << std::endl;
            }

            /**
             * @brief Simpliy print a data array
             * 
             * @tparam T 
             * @param valName 
             * @param data 
             */
            template<typename T>
            void Print(const std::string &valName , T* data){
                std::cout << valName << ": " << *data << std::endl;
            }

            /**
             * @brief Print the data on GPU to console
             * 
             * @tparam T 
             * @param valName Entry a name
             * @param d_data The data pointer in GPU  
             * @param count The number of data
             */
            template<typename T>
            void DeviceDataPrint(const std::string &valName , T* d_data , size_t count){
                cudaDeviceSynchronize();

                std::vector<T> h_data(count);
                cudaError_t err = cudaMemcpy(h_data.data(), d_data, count * sizeof(T), cudaMemcpyDeviceToHost);
                if(err != cudaSuccess){
                    std::cerr << "Error: CUDA memcpy failed in DeviceDataPrint: " << cudaGetErrorString(err) << std::endl;
                    return;
                }
                
                // Print the first some elements for debug
                std::cout << valName << " (first 10 elements): ";
                size_t print_count = std::min(count, static_cast<size_t>(10));
                for(size_t i = 0; i < print_count; ++i) {
                    std::cout << h_data[i] << " ";
                }
                if(count > 10) std::cout << "...";
                std::cout << std::endl;
            }

            /**
             * @brief Log the data on GPU to a txt file
             * 
             * @tparam T 
             * @param valName Entry a name
             * @param d_data The data pointer in GPU  
             * @param count The number of data
             * @param path The path to save the file
             * @param filename The file you want to name
             */
            template<typename T>
            void DeviceDataLogger(const std::string &valName , T* d_data , 
                size_t count , const std::string &path = "./" , const std::string &filename = "debuglog.txt"){
                cudaDeviceSynchronize();

                std::vector<T> h_data(count);
                cudaError_t err = cudaMemcpy(h_data.data(), d_data, count * sizeof(T), cudaMemcpyDeviceToHost);
                if(err != cudaSuccess){
                    std::cerr << "Error: CUDA memcpy failed in DeviceDataLogger: " << cudaGetErrorString(err) << std::endl;
                    return;
                }

                std::string full_path = joinPath(path, filename);
                std::ofstream outfile(full_path);
                
                if(outfile.is_open()){
                    outfile << "Data type: " << valName << std::endl;
                    outfile << "Count: " << count << std::endl;
                    for(size_t i = 0; i < count; ++i) {
                        outfile << h_data[i] << std::endl;
                    }
                    outfile.close();
                    std::cout << "Data logged to: " << full_path << std::endl;
                } else {
                    std::cerr << "Error: Could not open file for writing: " << full_path << std::endl;
                }
            }

            /**
             * @brief Log the structure array on GPU to a txt file. This function only for Triangles structure.
             * 
             * @param d_Tris The structure array pointer in GPU  
             * @param count The number of structure array
             * @param path The path to save the file
             * @param filename The file you want to name
             */
            void DeviceStrcLogger(CudaBus::Triangles* d_Tris , size_t count , 
                                  const std::string &path = "./" , const std::string &filename = "Triangleslog.txt"){
                cudaDeviceSynchronize();

                // Use std::vector for RAII memory management (C++14 compatible)
                std::vector<CudaBus::Triangles> h_Tris(count);
                cudaError_t err = cudaMemcpy(h_Tris.data(), d_Tris, count * sizeof(CudaBus::Triangles), cudaMemcpyDeviceToHost);
                if(err != cudaSuccess){
                    std::cerr << "Error: CUDA memcpy failed in DeviceStrcLogger: " << cudaGetErrorString(err) << std::endl;
                    return;
                }

                std::string full_path = joinPath(path, filename);
                std::ofstream outfile(full_path);
                if(outfile.is_open()){
                    outfile << "Total Triangles: " << count << "\n";
                    for(size_t i = 0 ; i < count ; i++){
                        outfile << "Num [" << i << "]:" << std::endl;
                        outfile << " Coords0: x" << h_Tris[i].coords[0].x << " y" << h_Tris[i].coords[0].y << " z" << h_Tris[i].coords[0].z << std::endl;
                        outfile << " Coords1: x" << h_Tris[i].coords[1].x << " y" << h_Tris[i].coords[1].y << " z" << h_Tris[i].coords[1].z << std::endl;
                        outfile << " Coords2: x" << h_Tris[i].coords[2].x << " y" << h_Tris[i].coords[2].y << " z" << h_Tris[i].coords[2].z << std::endl;
                        outfile << " Equation: " << h_Tris[i].Equation.x << "x + " << h_Tris[i].Equation.y << "y + "
                                << h_Tris[i].Equation.z << "z + " << h_Tris[i].Equation.w << " = 0" << std::endl;
                        outfile << " zRange: minZ " << h_Tris[i].zRange.x << " maxZ " << h_Tris[i].zRange.y << std::endl;
                    }
                    outfile.close();
                    std::cout << "Triangle data logged to: " << full_path << std::endl;
                } else {
                    std::cerr << "Error: Could not open file for writing: " << full_path << std::endl;
                }
            }

            /**
             * @brief Save slicing results to output directory (DEBUG mode only)
             * This function is only compiled when ifDEBUG is defined.
             * Saves each layer's intersection points to separate text files.
             * 
             * @param layerData 2D vector containing intersection points for each layer
             * @param outputPath The output directory path
             * @param resolution The resolution used for slicing (for Z height calculation)
             */
            #ifdef ifDEBUG
            void SaveSlicingResults(const std::vector<std::vector<float3>>& layerData, 
                                   const std::string& outputPath, 
                                   float resolution) {
                if(layerData.empty()) {
                    std::cerr << "[WARNING] SaveSlicingResults: No data to save (layerData is empty)" << std::endl;
                    return;
                }
                
                // Ensure output directory exists
                struct stat info;
                if(stat(outputPath.c_str(), &info) != 0 || !S_ISDIR(info.st_mode)) {
                    std::cerr << "[ERROR] SaveSlicingResults: Output path does not exist or is not a directory: " << outputPath << std::endl;
                    return;
                }
                
                std::cout << "[DEBUG] Saving slicing results to: " << outputPath << std::endl;
                std::cout << "[DEBUG] Total layers: " << layerData.size() << std::endl;
                
                // Save each layer to a separate file
                for(size_t layerIdx = 0; layerIdx < layerData.size(); layerIdx++) {
                    const auto& layer = layerData[layerIdx];
                    
                    // Skip empty layers
                    if(layer.empty()) {
                        continue;
                    }
                    
                    // Create filename: layer_XXXX.txt (4-digit zero-padded)
                    std::string layerNum = std::to_string(layerIdx);
                    while(layerNum.length() < 4) {
                        layerNum = "0" + layerNum;
                    }
                    std::string filename = joinPath(outputPath, "layer_" + layerNum + ".txt");
                    
                    // Open file for writing
                    std::ofstream outfile(filename);
                    if(!outfile.is_open()) {
                        std::cerr << "[ERROR] SaveSlicingResults: Failed to open file for writing: " << filename << std::endl;
                        continue;
                    }
                    
                    // Write header
                    outfile << "# Layer " << layerIdx << " intersection points" << std::endl;
                    outfile << "# Z height: " << (layerIdx * resolution) << " mm" << std::endl;
                    outfile << "# Total points: " << layer.size() << std::endl;
                    outfile << "# Format: x y z (one point per line, NaN indicates invalid point)" << std::endl;
                    
                    // Write points (including NaN points for observation)
                    size_t validPoints = 0;
                    size_t invalidPoints = 0;
                    for(const auto& point : layer) {
                        const bool xNan = std::isnan(point.x);
                        const bool yNan = std::isnan(point.y);
                        const bool zNan = std::isnan(point.z);
                        const bool isValid = (!xNan && !yNan && !zNan);

                        if(isValid) {
                            validPoints++;
                        } else {
                            invalidPoints++;
                        }

                        outfile
                            << (xNan ? "nan" : std::to_string(point.x)) << " "
                            << (yNan ? "nan" : std::to_string(point.y)) << " "
                            << (zNan ? "nan" : std::to_string(point.z));
                        if(!isValid) {
                            outfile << "  # invalid";
                        }
                        outfile << std::endl;
                    }
                    
                    outfile.close();
                    
                    if(layerIdx < 5 || layerIdx % 100 == 0) {
                        std::cout << "[DEBUG] Layer " << layerIdx << ": Saved "
                                  << validPoints << " valid points, "
                                  << invalidPoints << " invalid points to "
                                  << filename << std::endl;
                    }
                }
                
                std::cout << "[DEBUG] All layers saved successfully" << std::endl;
            }
            #endif // ifDEBUG
/*
            template<typename T>
            void MatrixOutput(T* &matrix){
                cudaDeviceSynchronize();
                for(size_t i = 0 ; i < matrix.size() ; i++){
                    std::cout << "Col:" << i << ": " ;
                    for(size_t j = 0 ; j < matrix[i].size() ; j++){
                        std::cout << matrix[i][j] << " ";
                    } 
                }
            }

            template<typename T>
            void MatrixLogger(T* &matrix , const std::string &path = "./" , const std::string &filename = "debuglog.txt"){
                cudaDeviceSynchronize();
                std::ofstream outfile(path + filename);
                if(outfile.is_open()){
                    outfile << "Matrix size: " << matrix.size() << "x" << matrix[0].size() << std::endl;
                    for(size_t i = 0 ; i < matrix.size() ; i++){
                        outfile << "Col:" << i << ": ";
                        for(size_t j = 0 ; j < matrix[j].size() ; j++){
                            outfile << matrix[i][j] << " ";
                        }
                        outfile << std::endl;
                    }
                    outfile.close();
                }
            }
I dont know why I program them but didnt use in any place. And they cause bugs. So just comment them out.
            */


    };

    class FrameTest{
        public:
        /**
         * @brief Dump a single frame to disk in PNG
         * 
         * @param d_data The frame data on GPU
         * @param width The width of the frame
         * @param height The height of the frame
         * @param filename The full path to the output file
         * @return true 
         * @return false 
         */
        bool dumpSingleFrame(uint8_t* d_data , uint32_t width , uint32_t height , const char *filename);

        bool dumpSingleFrame(uint8_t* d_data , uint32_t width , uint32_t height , const char *filename , bool isColToRow);

        /**
         * @brief Pick random frame index in global layers.
         * 
         * @param frameNum How many frames in total
         * @param seed Seed of random
         * @return An array including 5 random frame index.
         */
        static std::array<uint32_t , 5> PickRandomFrameIdx(uint32_t frameNum , uint32_t seed = 790221u);

        /**
         * @brief Dump random frame in chunk in PNG
         * 
         * @param BeginIdx The begin index of chunk
         * @param EndIdx The end index of chunk
         * @param d_ChunkFrames The image data of this chunk
         * @param width The width of the frame in pixel
         * @param height The height of the frame in pixel
         * @param OutPutDir The output directory
         * @param FilePrefix The prefix of output image
         * @return A vector including the index of the frame that should be output
         */
        void DumpMiddleFrameInChunk(uint32_t BeginIdx , uint32_t EndIdx ,  uint8_t** d_ChunkFrames,
        uint32_t width , uint32_t height , const char *OutPutDir , const char *FilePrefix = "DebugFrame" , bool isColToRow = true);

        void SingleColToRow(uint8_t *ColData , uint8_t *RowData , uint32_t Width , uint32_t Height);

        void PolygonOutput(std::vector<float2> &h_Segments, const char *filename , const char *OutPutDir, uint32_t Width , uint32_t Height , uint32_t Thickness , uint32_t Layer , float QuintifyParam = 1.0f);

        void SegmentsRawDataOut(std::vector<float2> &h_Segments, const char *filename , const char *OutPutDir , uint32_t Layer , float QuintifyParam = 1.0f);
    };
}

#if defined(SPEED_TEST) && (SPEED_TEST == 1)

class SpeedTest{
private:
    using Clock = std::chrono::steady_clock;
    std::vector<Clock::time_point> StartStamp;
    std::vector<Clock::time_point> StopStamp;
    std::vector<std::string> ChannelName;

public:
    explicit SpeedTest(uint8_t channelNum){
        std::cout << "Speed test mode is on" << std::endl;
        StartStamp.resize(channelNum);
        StopStamp.resize(channelNum);
        ChannelName.resize(channelNum);
    }

    void StartTest(int channel , std::string name){
        if(channel >= static_cast<int>(StartStamp.size())){
            StartStamp.resize(channel + 1);
            StopStamp.resize(channel + 1);
            ChannelName.resize(channel + 1);
        }
        StartStamp[channel] = Clock::now();
        ChannelName[channel] = std::move(name);
    }

    void StopTest(int channel , std::string /*name*/){
        if(channel >= 0 && channel < static_cast<int>(StopStamp.size())){
            StopStamp[channel] = Clock::now();
        }
    }

    void PrintResult(){
        std::cout << "Speed test result:" << std::endl;
        for(size_t i = 0 ; i < ChannelName.size() ; i++){
            auto us = std::chrono::duration_cast<std::chrono::microseconds>(StopStamp[i] - StartStamp[i]).count();
            std::cout << ChannelName[i] << ": " << (us / 1000.0) << " ms" << std::endl;
        }
    }
};

#endif
