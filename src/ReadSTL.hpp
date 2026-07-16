#pragma once

#include <stl_reader.h>
#include <vector>
#include <sstream>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <memory>

#include <cuda_runtime.h> 
#include <device_launch_parameters.h>
/* For developing on the device without CUDA environment. Please quote them before you complie the code.*/

#include "cudaBus.hpp"

#if defined(__CUDACC__ ) || defined(__CUDA_ARCH__)
    #define CUDA_POSSIBLE 1
    #include <cuda_runtime.h>
    #include <device_launch_parameters.h>
#else
    #define CUDA_POSSIBLE 0
#endif

#ifndef ifDEBUG
#define ifDEBUG 1
#endif


using std::vector;

class ReadSTL {
    stl_reader::StlMesh<float, unsigned int> mesh;
    public:
        static std::string filename;
        static uint32_t numsOfTriangles;
        vector<float> Coords;
        vector<float> Coords1;
        vector<float> Coords2;
        vector<float> NormalsOrient;

        
    /**
     * @brief Set the Filename object
     * 
     * @param filename Such as "home/user/file.stl"
     */
        void setFilename(const char *filename) {
            this->filename = filename;
            mesh.read_file(filename);
            numsOfTriangles = mesh.num_tris();
            std::cout << "STL file loaded successfully. Target name: " << filename << std::endl;
        }


        ReadSTL() {
        }

        ~ReadSTL() {
        }


        /**
         * @brief Read the file into vector
         * 
         * @param uint The unit of the file. Default is mm(1.0f), 10.0f for cm
         */
        void readIntoVector(float uint = 1.0f) {
            try{
                for(uint64_t i = 0; i < numsOfTriangles; i++){
                    Coords.push_back(mesh.tri_corner_coords(i, 0)[0] * uint);
                    Coords.push_back(mesh.tri_corner_coords(i, 0)[1] * uint);
                    Coords.push_back(mesh.tri_corner_coords(i, 0)[2] * uint);

                    Coords1.push_back(mesh.tri_corner_coords(i, 1)[0] * uint);
                    Coords1.push_back(mesh.tri_corner_coords(i, 1)[1] * uint);
                    Coords1.push_back(mesh.tri_corner_coords(i, 1)[2] * uint);

                    Coords2.push_back(mesh.tri_corner_coords(i, 2)[0] * uint);
                    Coords2.push_back(mesh.tri_corner_coords(i, 2)[1] * uint);
                    Coords2.push_back(mesh.tri_corner_coords(i, 2)[2] * uint);

                    NormalsOrient.push_back(mesh.tri_normal(i)[0]);
                    NormalsOrient.push_back(mesh.tri_normal(i)[1]);
                    NormalsOrient.push_back(mesh.tri_normal(i)[2]);
                }

            uint32_t meshNum = mesh.num_tris();
            uint32_t fileSize = meshNum * 4 * 4 * 3;
            uint32_t memSize = Coords.size() * 4 + Coords1.size() * 4 + Coords2.size() * 4 + NormalsOrient.size() * 4;
            std::cout << "Read mesh successfully, total count: " << meshNum << std::endl;
            std::cout << "File size: " << fileSize << " byte" << std::endl;
            std::cout << "Memory using: " << memSize << " float" << std::endl;
            }
            catch(std::exception &e){
                std::cout << "Memory Error: " << e.what() << std::endl;
            }
        }


};

class IntoEquationGPU : public ReadSTL {
    public:
    CudaBus::cudaMemBus d_coords0;
    CudaBus::cudaMemBus d_normalsOrient;
    CudaBus::cudaMemBus d_Equations;
    CudaBus::cudaMemBus d_coords1;
    CudaBus::cudaMemBus d_coords2;
    CudaBus::cudaMemBus d_TriCentral;
    CudaBus::cudaMemBus d_Trinum;
        IntoEquationGPU() = default;

        ~IntoEquationGPU() {
        }

        /**
        * @brief Initialize the data into GPU
        * 
        */
        void initIntoGPU() {
            int deviceCount;
            cudaError_t err1 = cudaGetDeviceCount(&deviceCount);
            if(err1 != cudaSuccess || deviceCount == 0){
                throw std::runtime_error("No GPU device found");
            }else{
                std::cout << "GPU device found" << deviceCount << std::endl;
            }

            d_coords0.initToGPU<float>(Coords , "Coord 0");
            d_normalsOrient.initToGPU<float>(NormalsOrient , "Normals Orient");
            d_coords1.initToGPU<float>(Coords1 , "Coord 1");
            d_coords2.initToGPU<float>(Coords2 , "Coord 2");

            vector<float> SpaceOfEqs(numsOfTriangles * 4 , 0.0f);
            vector<float> TriCentral(numsOfTriangles * 3 , 0.0f);
            d_Equations.initToGPU<float>(SpaceOfEqs , "Equations' space");
            d_TriCentral.initToGPU<float>(TriCentral , "TriCentral's space");
            vector<uint64_t> TriNum(1 , numsOfTriangles);
            d_Trinum.initToGPU<uint64_t>(TriNum , "TriNum's space");

            Coords.clear();
            NormalsOrient.clear();
            Coords1.clear();
            Coords2.clear();
            
        }
    };



