#pragma once

#ifndef ifDEBUG
#define ifDEBUG 1
#endif

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sys/stat.h>  // 
#include <sys/types.h>
#include <errno.h>     // for error
#include <unistd.h>    // For ac

class Cmd{
    public:
    typedef struct{
        std::string ModelName;
        std::string OutputPath;
        std::string LogPath;
        float Resolution;
        float Error;
        uint32_t Width;
        uint32_t Height;
        uint32_t FrameRate;
        uint32_t BitRate;
        float ScaleX;
        float ScaleY;
        float ScaleZ;
        std::string VideoName;
        uint32_t Chunk;
        float QuantifyParam;
        bool AutoAlternative;
        uint32_t Threads;
    } Parameters;
    /**
     * @brief Print usage example of the program
     * 
     * @param programName Your program name
     */
    void PrintUsage(const char* programName){
        std::cout << "Usage: " << programName << " <ModelName> <Resolution(unit mm)> <OutputPath> <LogPath> ";
        std::cout << "<Error(unit mm)> <Width> <Height> <FramRate> <BitRate> <ScaleX> <ScaleY> <VideoName> <Chunk> <QuantifyParam> <AutoAlternative(0/1)>\n" << std::endl;
        std::cout << "Example: " << programName << " .home/Model.stl 0.1 ./home ./home 0.001 3840 2160 1 80000 35.0 20.0 video 256 100.0 0" << std::endl;
        std::cout << "Note: " << std::endl;
        std::cout << "1. Resolution(unit mm) is the z resolution of the slicing layer. " << std::endl;
        std::cout << "2. Error(unit mm) is the error tolerance of the slicing and raster scanning." << std::endl;
        std::cout << "3. Width and Height are the width and height pixels of the output video. " << std::endl;
        std::cout << "4. ScaleX, ScaleY, ScaleZ are the physical scale of the printing chamber. " << std::endl;
        std::cout << "5. VideoName: Name your video." << std::endl;
        std::cout << "6. Chunk: If process all the layer at once, it will occupy too much memory. Divide into chunks to asynchronizly process." << std::endl;
        std::cout << "7. QuantifyParam: The quantification parameter for the raster process. " << std::endl;
        std::cout << "8. AutoAlternative: Debug helper switch. 1 means force auto transform and auto quantify. Do not enable it in non-debug mode." << std::endl;
        std::cout << "9. Threads: The number of threads to use. In order to optimize performence by multi threads, decided by platform. Recommend that do not more than phyisical core " << std::endl;
    }

    /**
     * @brief Check if a file exists
     * 
     * @param filepath Path to the file
     * @return true if file exists
     * @return false if file does not exist
     */
    bool FileExists(const std::string& filepath) {
        struct stat buffer;
        return (stat(filepath.c_str(), &buffer) == 0 && S_ISREG(buffer.st_mode));
    }

    /**
     * @brief Check if a directory exists
     * 
     * @param dirpath Path to the directory
     * @return true if directory exists
     * @return false if directory does not exist
     */
    bool DirectoryExists(const std::string& dirpath) {
        struct stat buffer;
        return (stat(dirpath.c_str(), &buffer) == 0 && S_ISDIR(buffer.st_mode));
    }

    /**
     * @brief To validate the arguments if they are valid
     * 
     * @param argc Arguments count
     * @param argv Arguments
     * @param ModelName The name of your stl model file
     * @param Resolution Slicing resolution. unit mm
     * @param OutputPath Out put path
     * @return true 
     * @return false 
     */
    bool ValidateArguments(int argc, char *argv[] , Parameters &parameters){ 
        if(argc < 18){
            std::cerr << "Insufficient arguments.\n";
            PrintUsage(argv[0]);
            return false;
        }

        parameters.ModelName = argv[1];
        parameters.Resolution = std::stof(argv[2]);
        parameters.OutputPath = argv[3];
        parameters.LogPath = argv[4];
        parameters.Error = std::stof(argv[5]);
        parameters.Width = std::stoi(argv[6]);
        parameters.Height = std::stoi(argv[7]);
        parameters.FrameRate = std::stoi(argv[8]);
        parameters.BitRate = std::stoi(argv[9]);
        parameters.ScaleX = std::stof(argv[10]);
        parameters.ScaleY = std::stof(argv[11]);
        parameters.ScaleZ = std::stof(argv[12]);
        parameters.VideoName = argv[13];
        parameters.Chunk = std::stoi(argv[14]);
        parameters.QuantifyParam = std::stof(argv[15]);
        int autoAlternativeValue = std::stoi(argv[16]);
        parameters.Threads = std::stoi(argv[17]);
        parameters.AutoAlternative = (autoAlternativeValue != 0);

        if(!FileExists(parameters.ModelName)){
            std::cerr << "Model file " << parameters.ModelName << " does not exist.\n";
            return false;
        }

        if(!DirectoryExists(parameters.OutputPath)){
            std::cerr << "Output path " << parameters.OutputPath << " does not exist.\n";
            return false;
        }

        if(autoAlternativeValue != 0 && autoAlternativeValue != 1){
            std::cerr << "AutoAlternative must be 0 or 1.\n";
            return false;
        }

        return true;
    }

    /**
     * @brief Create directory recursively (C++14 compatible, no std::filesystem)
     * 
     * @param path Directory path to create
     * @return true if successful
     */
    bool createDirectoryRecursive(const std::string& path) {
        if (path.empty()) return false;
        
        size_t pos = 0;
        std::string current_path;
        
        // Handle absolute paths starting with /
        if (path[0] == '/') {
            current_path = "/";
            pos = 1;
        }
        
        while ((pos = path.find('/', pos + 1)) != std::string::npos) {
            current_path = path.substr(0, pos);
            if (current_path.empty() || current_path == "/") continue;
            
            struct stat info;
            if (stat(current_path.c_str(), &info) != 0) {
                if (mkdir(current_path.c_str(), 0755) != 0 && errno != EEXIST) {
                    return false;
                }
            }
        }
        
        // Create the final directory
        if (mkdir(path.c_str(), 0755) != 0 && errno != EEXIST) {
            return false;
        }
        return true;
    }

    /**
     * @brief To validate if the log directory exists, create if not
     * 
     * @param logPath Enter the log directory
     * @return true if directory exists or was created successfully
     * @return false if directory creation failed
     */
    bool EnsureLogDirectory(const std::string& logPath) {
        if (DirectoryExists(logPath)) {
            return true;
        }
        
        // Try to create directory (supports multi-level paths)
        if (createDirectoryRecursive(logPath)) {
            return true;
        }
        
        // If creation failed, try simple mkdir as fallback
        return (mkdir(logPath.c_str(), 0755) == 0) || (errno == EEXIST);
    }
};
