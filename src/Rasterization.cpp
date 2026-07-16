#include "Rasterization.hpp"

namespace Raster{
    uint32_t Rasterization::PixelHeight = 2160;
    uint32_t Rasterization::PixelWidth = 3840;
    float Rasterization::Error = 0.00001f;
    bool Rasterization::AutoAlternative = false;
    size_t Rasterization::ThreadSet = 8;
    std::string Rasterization::OutputPath = "./";
    float Rasterization::AutoQuntifyParam = 1.0f;

    float Rasterization::AutoAlternativeAndQuantify(std::vector<std::vector<float3>> &data){
        const float FrameWidthMM = 35.0f;
        const float FrameHeightMM = 18.0f;

        if(data.empty()){
            return 1.0f;
        }

        float minX = std::numeric_limits<float>::max();
        float minY = std::numeric_limits<float>::max();
        float minZ = std::numeric_limits<float>::max();

        float maxX = -std::numeric_limits<float>::max();
        float maxY = -std::numeric_limits<float>::max();
        float maxZ = -std::numeric_limits<float>::max();

        bool hasValidPoint = false;

    #pragma omp parallel
        {
            const uint32_t Layers = static_cast<uint32_t>(data.size());
            const uint32_t ThreadIdx = static_cast<uint32_t>(omp_get_thread_num());
            const uint32_t ThreadNum = static_cast<uint32_t>(omp_get_num_threads());

            uint32_t start = (Layers * ThreadIdx) / ThreadNum;
            uint32_t end = (Layers * (ThreadIdx + 1)) / ThreadNum;

            float tMinX = std::numeric_limits<float>::max();
            float tMinY = std::numeric_limits<float>::max();
            float tMinZ = std::numeric_limits<float>::max();

            float tMaxX = -std::numeric_limits<float>::max();
            float tMaxY = -std::numeric_limits<float>::max();
            float tMaxZ = -std::numeric_limits<float>::max();

            int tValid = 0;

        for(uint32_t Layers = start; Layers < end; ++Layers){
            const auto &vec = data[Layers];
            for(size_t i = 0; i < vec.size(); ++i){ 
                const float3 &p = vec[i];
                if(std::isnan(p.x) || std::isnan(p.y) || std::isnan(p.z)){
                    continue;
                }

                tValid = 1;

                if(p.x < tMinX) tMinX = p.x;
                if(p.y < tMinY) tMinY = p.y;
                if(p.z < tMinZ) tMinZ = p.z;

                if(p.y > tMaxY) tMaxY = p.y;
                if(p.x > tMaxX) tMaxX = p.x;
                if(p.z > tMaxZ) tMaxZ = p.z;
            }
        }

    #pragma omp critical
        {
            if(tValid){
                if(!hasValidPoint){
                    minX = tMinX; minY = tMinY; minZ = tMinZ;
                    maxX = tMaxX; maxY = tMaxY; maxZ = tMaxZ;
                    hasValidPoint = true;
                }
            else{
                if(tMinX < minX) minX = tMinX;
                if(tMinY < minY) minY = tMinY;
                if(tMinZ < minZ) minZ = tMinZ;

                if(tMaxX > maxX) maxX = tMaxX;
                if(tMaxY > maxY) maxY = tMaxY;
                if(tMaxZ > maxZ) maxZ = tMaxZ;
                }
            }
        }
        }

        if(!hasValidPoint){
            return 1.0f;
        }

        float deltaX = maxX - minX;
        float deltaY = maxY - minY;
        float deltaZ = maxZ - minZ;

        if(deltaX < Error) deltaX = Error;
        if(deltaY < Error) deltaY = Error;
        if(deltaZ < Error) deltaZ = Error;

        // Uniform scale: fit model into 35mm * 18mm frame
        const float scaleX = FrameWidthMM / deltaX;
        const float scaleY = FrameHeightMM / deltaY;
        const float uniformScale = std::min(scaleX, scaleY);

        // Move model to first quadrant first, then scale, then keep it inside the frame
        for(size_t layer = 0; layer < data.size(); ++layer){
            for(size_t i = 0; i < data[layer].size(); ++i){
                float3 &p = data[layer][i];

                if(std::isnan(p.x) || std::isnan(p.y) || std::isnan(p.z)){
                    continue;
                }

                p.x = (p.x - minX) * uniformScale;
                p.y = (p.y - minY) * uniformScale;
                p.z = (p.z - minZ) * uniformScale;
            }
        }

        // Quantify parameter derived from the debug frame physical size
        const float quantifyX = static_cast<float>(PixelWidth) / FrameWidthMM;
        const float quantifyY = static_cast<float>(PixelHeight) / FrameHeightMM;
        const float quantify = std::min(quantifyX, quantifyY);

#ifdef ifDEBUG
        std::cout << "[DEBUG] AutoAlternative enabled" << std::endl;
        std::cout << "[DEBUG] Original bounds: "
                  << "X[" << minX << ", " << maxX << "] "
                  << "Y[" << minY << ", " << maxY << "] "
                  << "Z[" << minZ << ", " << maxZ << "]" << std::endl;
        std::cout << "[DEBUG] Uniform scale: " << uniformScale << std::endl;
        std::cout << "[DEBUG] Debug frame(mm): " << FrameWidthMM << " x " << FrameHeightMM << std::endl;
        std::cout << "[DEBUG] QuantifyParam: " << quantify << std::endl;
#endif

        return quantify;
    }

    void Rasterization::InitParams(uint32_t width , uint32_t height , float error , bool alternative , uint32_t Threads){ 
        Error = error;
        PixelWidth = width;
        PixelHeight = height;
        AutoAlternative = alternative;

        const uint32_t availThreads = static_cast<uint32_t>(std::max(1 , omp_get_num_procs()));
        uint32_t usedThreads = (Threads == 0) ? availThreads : Threads;
        if(usedThreads > availThreads){
            usedThreads = availThreads;
            std::cout << "[WARNING] In Rasterization.cpp::InitParams(), Threads set is out of maxium allowed threads. Using " << usedThreads << " threads instead.";
        }
        if(usedThreads == 0){
            usedThreads = 1;
            std::cout << "[WARNING] In Rasterization.cpp::InitParams(), Threads set is 0. Using 1 thread instead.";
        }

        ThreadSet = usedThreads;
    }


    void Rasterization::inTo2D(std::vector<std::vector<float3>> &data , std::vector<std::vector<float2>> &PlaneData){
        uint32_t Layers = data.size();
        PlaneData.resize(Layers);
        for(uint32_t i = 0; i < Layers; i++){
            uint32_t Points = data[i].size();
            PlaneData[i].resize(Points);
            for(uint32_t j = 0; j < Points; j++){
                PlaneData[i][j].x = data[i][j].x;
                PlaneData[i][j].y = data[i][j].y;
            }
        }
    }

    void Rasterization::IntoStructrue(std::vector<std::vector<float2>> &data , std::vector<std::vector<qTriangles>> &PlaneDataStrc , float QuntifyParam){ 
        uint32_t layers = static_cast<uint32_t>(data.size());
        PlaneDataStrc.clear();
        PlaneDataStrc.resize(layers);

        if(layers == 0){
            return;
        }

    #pragma omp parallel
    {
        uint32_t ThreadIdx = static_cast<uint32_t>(omp_get_thread_num());
        uint32_t ThreadNum = static_cast<uint32_t>(omp_get_num_threads());

        uint32_t start = ThreadIdx * layers / ThreadNum;
        uint32_t end = (ThreadIdx + 1) * layers / ThreadNum;

        for(uint32_t Layer = start; Layer < end; ++Layer){
            const auto &VecIn = data[Layer];
            auto &VecOut = PlaneDataStrc[Layer];

            const uint32_t OringinalPoints = static_cast<uint32_t>(VecIn.size());
            const uint32_t PaddedPoints = ((OringinalPoints + 2u) / 3u) * 3u;

            VecOut.clear();
            VecOut.reserve(PaddedPoints / 3u);

            for(uint32_t i = 0; i < PaddedPoints / 3; ++i){
                uint32_t i0 = i * 3u;
                uint32_t i1 = i0 + 1u;
                uint32_t i2 = i0 + 2u;

                float2 p0 = (i0 < OringinalPoints) ? VecIn[i0] : make_float2(std::nanf("") , std::nanf(""));
                float2 p1 = (i1 < OringinalPoints) ? VecIn[i1] : make_float2(std::nanf("") , std::nanf(""));
                float2 p2 = (i2 < OringinalPoints) ? VecIn[i2] : make_float2(std::nanf("") , std::nanf(""));

                int type = GetType(p0 , p1 , p2);
                if(type == -1){
                    continue;
                }

                qTriangles Temp;
                Temp.p0 = make_float2(p0.x * QuntifyParam * ZoomParam, p0.y * QuntifyParam * ZoomParam);
                Temp.p1 = make_float2(p1.x * QuntifyParam * ZoomParam, p1.y * QuntifyParam * ZoomParam);

                if(type != 2){
                    Temp.p2 = make_float2(-100000.0f, -100000.0f);
                }else{
                    Temp.p2 = make_float2(p2.x * QuntifyParam * ZoomParam, p2.y * QuntifyParam * ZoomParam);
                }

                Temp.Type = type;
                VecOut.push_back(Temp);
            }
        }
    }
    }

    int Rasterization::GetType(float2 p0 , float2 p1 , float2 p2){ 
        if(std::isnan(p0.x) && std::isnan(p1.x) && std::isnan(p2.x)){
            return -1;
        }

        if(!std::isnan(p2.x)){
            return 2;
        }

        if(std::isnan(p2.x)){
            if(inTolerance(p0.x, p1.x, Error) && inTolerance(p0.y, p1.y, Error)){
                return 1;
            }else{
                return 0;
            }
        }
        throw std::runtime_error("Invalid triangle slice pattern");
    }

    bool Rasterization::inTolerance(float x , float y , float tolerance){ 
        if(std::abs(x - y) <= tolerance){
            return true;
        }else{
            return false;
        }
        return false;
    }

    void Rasterization::ConstructOverlappingPoints(std::vector<qTriangles> &h_triangles , uint32_t** &d_OverLappings){
        std::vector<float2> Points;
        std::vector<std::vector<uint32_t>> h_overLappingPoints;
        h_overLappingPoints.resize(PixelHeight);
        std::vector<uint32_t*> h_RowDatas(PixelHeight , nullptr);
        auto err = cudaGetLastError();

        if(h_triangles.empty()){
            goto EmptyLayer;
        }

        for(uint32_t i = 0; i < h_triangles.size(); i++){
            switch(h_triangles[i].Type){
                case 0:
                Points.push_back(h_triangles[i].p0);
                Points.push_back(h_triangles[i].p1);
                break;

                case 1:
                Points.push_back(h_triangles[i].p0);
                Points.push_back(h_triangles[i].p1);
                break;

                case 2:
                Points.push_back(h_triangles[i].p0);
                Points.push_back(h_triangles[i].p1);
                Points.push_back(h_triangles[i].p1);
                Points.push_back(h_triangles[i].p2);
                Points.push_back(h_triangles[i].p2);
                Points.push_back(h_triangles[i].p0);
                break;

                default:
                throw std::runtime_error("[ERROR] ConstructOverlappingPoints: Invalid triangle type");
                break;
            }
        }
            std::sort(Points.begin(), Points.end(), [](const float2& a, const float2& b){
                if(a.x != b.x) return a.x < b.x;
                return a.y < b.y;
            });

            for(uint32_t i = 1; i < Points.size(); i++){
                if(Points[i].x == Points[i-1].x && Points[i].y == Points[i-1].y){
                    if(Points[i].y >= 0 && Points[i].y < static_cast<int>(PixelHeight) &&
                        Points[i].x >= 0 && Points[i].x < static_cast<int>(PixelWidth)){
                        h_overLappingPoints[Points[i].y].push_back(static_cast<uint32_t>(Points[i].x));
                    }
                }
            }


            
            for(uint32_t i = 0; i < PixelHeight; i++){
                uint32_t xSize = h_overLappingPoints[i].size();
                err = cudaMalloc(&h_RowDatas[i], (xSize + 1) * sizeof(uint32_t));
                if(err != cudaSuccess){
                    throw std::runtime_error("[ERROR] ConstructOverlappingPoints: In overlapping constructing, cudaMalloc failed");
                }
                std::vector<uint32_t> temp;
                temp.reserve(xSize + 1);
                temp.push_back(xSize);
                temp.insert(temp.end(), h_overLappingPoints[i].begin(), h_overLappingPoints[i].end());

                err = cudaMemcpy(h_RowDatas[i], temp.data(), (xSize + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice);
                if(err != cudaSuccess){
                    throw std::runtime_error("[ERROR] ConstructOverlappingPoints: In overlapping constructing, cudaMemcpy failed");
                }
                temp.clear();
            }

            err = cudaMalloc(&d_OverLappings, PixelHeight * sizeof(uint32_t*));
            if(err != cudaSuccess){
                throw std::runtime_error("[ERROR] ConstructOverlappingPoints: In overlapping constructing, cudaMalloc for d_OverLappingPoints failed");
            }
            err = cudaMemcpy(d_OverLappings, h_RowDatas.data(), PixelHeight * sizeof(uint32_t*), cudaMemcpyHostToDevice);
            if(err != cudaSuccess){
                throw std::runtime_error("[ERROR] ConstructOverlappingPoints: In overlapping constructing, cudaMemcpy for d_OverLappingPoints failed");
            }
            return;

            EmptyLayer:
            for(uint32_t i = 0; i < PixelHeight; i++){
                err = cudaMalloc(&h_RowDatas[i], sizeof(uint32_t));
                if(err != cudaSuccess){
                    throw std::runtime_error("[ERROR] ConstructOverlappingPoints: In overlapping constructing, empty cudaMalloc failed");
                }

                uint32_t empty[1] = {0};

                err = cudaMemcpy(h_RowDatas[i], empty, sizeof(uint32_t), cudaMemcpyHostToDevice);
                if(err != cudaSuccess){
                    throw std::runtime_error("[ERROR] ConstructOverlappingPoints: In overlapping constructing, empty cudaMemcpy failed");
                }
            }
            err = cudaMalloc(&d_OverLappings, PixelHeight * sizeof(uint32_t*));
            if(err != cudaSuccess){
                throw std::runtime_error("[ERROR] ConstructOverlappingPoints: In overlapping constructing, cudaMalloc for empty d_OverLappingPoints failed");
            }
            err = cudaMemcpy(d_OverLappings, h_RowDatas.data(), PixelHeight * sizeof(uint32_t*), cudaMemcpyHostToDevice);
            if(err != cudaSuccess){
                throw std::runtime_error("[ERROR] ConstructOverlappingPoints: In overlapping constructing, cudaMemcpy for empty d_OverLappingPoints failed");
            }
            return;
        }

        void Rasterization::ReadyToKernel(uint32_t BatchSize, uint32_t BatchNum, std::vector<std::vector<Rasterization::qTriangles>> &h_triangles, std::vector<std::vector<float2>> &h_SegsOut){
            const uint32_t BeginLayer = BatchNum * BatchSize;
            const uint32_t EndLayer = std::min(BeginLayer + BatchSize, static_cast<uint32_t>(h_triangles.size()));
            
             if(BeginLayer >= EndLayer){
                h_SegsOut.clear();
                return;
            }

            uint32_t LocalLayers = EndLayer - BeginLayer;

            h_SegsOut.clear();
            h_SegsOut.resize(LocalLayers);

            #pragma omp parallel
            {
                uint32_t ThreadIdx = static_cast<uint32_t>(omp_get_thread_num());
                uint32_t ThreadNum = static_cast<uint32_t>(omp_get_num_threads());

                uint32_t Start = LocalLayers * ThreadIdx / ThreadNum;
                uint32_t End = LocalLayers * (ThreadIdx + 1) / ThreadNum;

                for(uint32_t local = Start ; local < End ; ++local){
                    uint32_t globalLayer = BeginLayer + local;
                    const auto &Tris = h_triangles[globalLayer];
                    auto &OutSegs = h_SegsOut[local];

                    OutSegs.clear();
                    OutSegs.reserve(Tris.size()*6);

                    for(size_t i = 0 ; i < Tris.size() ; ++i){
                        const auto &Tri = Tris[i];
                        switch (Tri.Type){
                            case 0:
                            OutSegs.push_back(Tri.p0);
                            OutSegs.push_back(Tri.p1);
                            break;

                            case 1:
                            OutSegs.push_back(Tri.p0);
                            OutSegs.push_back(Tri.p1);
                            break;

                            case 2:
                            OutSegs.push_back(Tri.p0);
                            OutSegs.push_back(Tri.p1);

                            OutSegs.push_back(Tri.p1);
                            OutSegs.push_back(Tri.p2);

                            OutSegs.push_back(Tri.p2);
                            OutSegs.push_back(Tri.p0);
                            break;

                            default:
                            break;
                        }
                    }
                }
            }
            
        }

    void RasterProcess::RecieveData(std::vector<std::vector<float3>> &data,
                                std::vector<std::vector<qTriangles>> &h_triangles,
                                uint32_t width,
                                uint32_t height,
                                float error,
                                float QuantifyParam,
                                uint32_t Threads,
                                bool alternative,
                                const char *OutputPath){
            InitParams(width, height, error, alternative , Threads);
            if(OutputPath != nullptr){
                Rasterization::OutputPath = OutputPath;
            }

            omp_set_dynamic(0);
            omp_set_nested(0);
            omp_set_num_threads(static_cast<int>(ThreadSet));

        #ifdef ifDEBUG
        std::cout << "[DEBUG] Rasterization Init. Parralle process in " << ThreadSet << "threads. " << std::endl;
        #endif

            if(AutoAlternative){
                QuantifyParam = AutoAlternativeAndQuantify(data);
                AutoQuntifyParam = QuantifyParam;
            }

            std::vector<std::vector<float2>> planeData;
            inTo2D(data, planeData);

            IntoStructrue(planeData, h_triangles, QuantifyParam);
        }


    void RasterProcess::ProcessData(std::vector<std::vector<qTriangles>> &h_triangles,
                                uint8_t** d_ChunkBitmaps,
                                uint32_t ChunkSize,
                                uint32_t ChunkNum){
    if(d_ChunkBitmaps == nullptr){
        throw std::runtime_error("[ERROR] ProcessData: d_ChunkBitmaps is null");
    }

    const uint32_t beginLayer = ChunkNum * ChunkSize;
    const uint32_t endLayer = std::min(beginLayer + ChunkSize,
                                       static_cast<uint32_t>(h_triangles.size()));

    if(beginLayer >= endLayer){
        return;
    }

    const uint32_t realChunkSize = endLayer - beginLayer;

    std::vector<std::vector<float2>> h_SegsOut;
    ReadyToKernel(ChunkSize, ChunkNum, h_triangles, h_SegsOut);

    #ifdef ifDEBUG
            const uint32_t debugLocalLayer = realChunkSize / 2u;
            const uint32_t debugGlobalLayer = beginLayer + debugLocalLayer;

            DeviceDebug::FrameTest TestOut;
            TestOut.SegmentsRawDataOut(
                h_SegsOut[debugLocalLayer],
                "Segments.txt",
                OutputPath.c_str(),
                debugGlobalLayer,
                ZoomParam);
            TestOut.PolygonOutput(
                h_SegsOut[debugLocalLayer],
                "Polygon",
                OutputPath.c_str(),
                PixelWidth,
                PixelHeight,
                3,
                debugGlobalLayer,
                ZoomParam);
    #endif

    for(uint32_t localLayer = 0; localLayer < realChunkSize; localLayer++){
        uint32_t globalLayer = beginLayer + localLayer;

        float2* d_Segments = nullptr;
        float2* d_LinerParas = nullptr;
        float2* d_RangesInY = nullptr;

        try{

            // 2. Prepare segment data
            std::vector<float2> &h_layerSegs = h_SegsOut[localLayer];
            const uint32_t pointCount = static_cast<uint32_t>(h_layerSegs.size());

            // Empty layer: directly output blank bitmap
            if(pointCount == 0){
                cudaError_t err = cudaMalloc(&d_ChunkBitmaps[localLayer],
                                             PixelWidth * PixelHeight * sizeof(uint8_t));
                if(err != cudaSuccess){
                    throw std::runtime_error("[ERROR] ProcessData: cudaMalloc bitmap failed on empty layer");
                }

                err = cudaMemset(d_ChunkBitmaps[localLayer], 0,
                                 PixelWidth * PixelHeight * sizeof(uint8_t));
                if(err != cudaSuccess){
                    throw std::runtime_error("[ERROR] ProcessData: cudaMemset bitmap failed on empty layer");
                }

                continue;
            }

            if((pointCount & 1u) != 0u){
                throw std::runtime_error("[ERROR] ProcessData: segment point count is odd");
            }

            const uint32_t segmentNum = pointCount / 2u;

            // 3. Copy segment endpoints to GPU
            cudaError_t err = cudaMalloc(&d_Segments, pointCount * sizeof(float2));
            if(err != cudaSuccess){
                throw std::runtime_error("[ERROR] ProcessData: cudaMalloc d_Segments failed");
            }

            err = cudaMemcpy(d_Segments,
                             h_layerSegs.data(),
                             pointCount * sizeof(float2),
                             cudaMemcpyHostToDevice);
            if(err != cudaSuccess){
                throw std::runtime_error("[ERROR] ProcessData: cudaMemcpy d_Segments failed");
            }

            // 4. Allocate line parameters and y-ranges
            err = cudaMalloc(&d_LinerParas, segmentNum * sizeof(float2));
            if(err != cudaSuccess){
                throw std::runtime_error("[ERROR] ProcessData: cudaMalloc d_LinerParas failed");
            }

            err = cudaMalloc(&d_RangesInY, segmentNum * sizeof(float2));
            if(err != cudaSuccess){
                throw std::runtime_error("[ERROR] ProcessData: cudaMalloc d_RangesInY failed");
            }

            // 5. Allocate output bitmap for this layer
            err = cudaMalloc(&d_ChunkBitmaps[localLayer],
                             PixelWidth * PixelHeight * sizeof(uint8_t));
            if(err != cudaSuccess){
                throw std::runtime_error("[ERROR] ProcessData: cudaMalloc bitmap failed");
            }

            err = cudaMemset(d_ChunkBitmaps[localLayer], 0,
                             PixelWidth * PixelHeight * sizeof(uint8_t));
            if(err != cudaSuccess){
                throw std::runtime_error("[ERROR] ProcessData: cudaMemset bitmap failed");
            }

            // 6. Launch kernels
            dim3 blockSeg(256);
            dim3 gridSeg((segmentNum + blockSeg.x - 1) / blockSeg.x);
            ToLinerEquation<<<gridSeg, blockSeg>>>(d_Segments, d_LinerParas, d_RangesInY , HorizonMark , VerticalMark , segmentNum);

            err = cudaGetLastError();
            if(err != cudaSuccess){
                throw std::runtime_error("[ERROR] ProcessData: ToLinerEquation launch failed");
            }

            dim3 blockRaster(256);
            dim3 gridRaster((PixelHeight + blockRaster.x - 1) / blockRaster.x);
            RasterizationKernel<<<gridRaster, blockRaster>>>(
                d_LinerParas,
                d_RangesInY,
                d_ChunkBitmaps[localLayer],
                segmentNum,
                PixelWidth,
                PixelHeight,
                HorizonMark,
                VerticalMark
            );

            err = cudaGetLastError();
            if(err != cudaSuccess){
                throw std::runtime_error("[ERROR] ProcessData: RasterizationKernel launch failed");
            }

            err = cudaDeviceSynchronize();
            if(err != cudaSuccess){
                throw std::runtime_error("[ERROR] ProcessData: cudaDeviceSynchronize failed");
            }

            // 7. Free per-layer temporary buffers
            cudaFree(d_Segments);
            cudaFree(d_LinerParas);
            cudaFree(d_RangesInY);

        }
        catch(...){
            if(d_Segments != nullptr) cudaFree(d_Segments);
            if(d_LinerParas != nullptr) cudaFree(d_LinerParas);
            if(d_RangesInY != nullptr) cudaFree(d_RangesInY);

            throw;
        }
    }
}

}
