# 项目信息

## 1. 开发背景
本项目的目标是在本地端完成3D模型的高速切片，光栅化和视频输出。原有Python实现虽然便于验证算法，但是在高分辨率，高层数和设备端部署场景下性能不足，因此使用CUDA C++重新实现核心流程。

项目当前主要面向具备NVIDIA GPU的设备，尤其是Jetson这类CPU和GPU共享物理内存的平台。尽管最初目标平台为Ubuntu 18.04，g++7.5.0，CUDA 10.2，但代码设计上尽量避免与单一平台深度绑定，除渲染后端外，大部分计算模块可以迁移到传统独立显存架构的计算机上运行。

最终输出不是普通图片序列，而是连续视频流。这样做的原因是切片结果层数较多，若全部保存为图片，会带来大量文件I/O和存储压力；视频流更适合连续层数据的保存，传输和后续处理。

## 2. 设计目标
项目设计时主要考虑以下目标：

1. 在4K分辨率下保持较高的切片和光栅化速度。
2. 尽量减少CPU和GPU之间的大数据传输。
3. 以矢量数据作为中间表达，仅在最终阶段生成位图。
4. 使用批次处理控制显存占用，避免一次性生成全部帧。
5. 在Jetson上使用NVIDIA硬件编码链路，将GPU位图直接推入视频管线。
6. 保留调试输出能力，便于定位切片层，光栅化结果和渲染流程中的错误。

总体思路是：前端尽量保持几何数据的精确性，中间流程尽量保持数据紧凑，后端再将结果转为适合视频编码的NV12帧。

## 3. 总体架构
项目流程可以概括为：

```text
STL读取 -> 三角形预处理 -> GPU切片 -> 线段/交点重组 -> 光栅化 -> NV12转换 -> GStreamer视频编码
```

代码模块大体分为以下几类：

1. 输入解析模块：负责读取STL文件，获得三角形顶点和法向量。
2. 切片模块：依据Z方向层高，对三角形和平面求交。
3. 光栅化模块：将每层线段集合转换为二维位图。
4. 渲染模块：将GRAY8位图转换为NV12，并推入NVIDIA编码管线。
5. 调试模块：输出中间层结果，辅助定位几何和内存问题。
6. 命令行模块：解析运行参数，组织主流程输入。

当前主工作链路位于 `mainRe.cpp`，切片使用 `ReProcess / ReSlicing`，光栅化使用 `Rasterization`，渲染使用 `Render2`。

## 4. 数据设计
### 4.1 STL数据特点
STL文件本质上只描述模型外表面。它由大量三角形构成，每个三角形包含三个顶点和一个法向量。因此STL数据具有以下特点：

**空心**

STL只记录外轮廓，不直接记录实体内部。切片时需要通过轮廓的内外关系判断哪些区域应被填充。

**离散**

曲面在STL中会被离散为大量三角形。模型精度越高，三角形数量通常越多。

**矢量化**

顶点，边，法向量和平面方程都可以用矢量表达，这使得GPU并行计算更容易组织。

### 4.2 核心数据结构
项目中重要的数据结构包括：

1. 三角形结构体：保存三角形顶点，平面方程和Z轴范围。
2. 切片字典：记录每一层可能相交的三角形索引。（注：新版中弃用）
3. 层数据容器：以 `std::vector<std::vector<float3>>` 保存每层切片结果于主机端，数据形式仍为矢量。
4. 光栅化线段结构：将切片交点重组为适合扫描线算法的线段数据。
5. 批次位图指针表：以 `uint8_t**` 管理当前批次的GPU位图，每帧扁平，行优先，帧序列为第二维度。需要注意，主流程中的批次指针表通常位于主机端，表内元素才是GPU位图指针；若核函数需要按层访问该表，应在外层控制函数中额外构建设备端指针表。
6. Render2帧池：以 `NvBufSurface` 作为视频编码前的帧存储结构。

### 4.3 内存结构思想
项目中最重要的内存原则是：大数据不要频繁跨CPU/GPU传输。

切片阶段保留矢量数据，因为每一层的有效轮廓点数量远小于完整位图。以4K图像为例，一张GRAY8位图约为8MB，而同一层的矢量轮廓数据通常远小于这个规模。因此中间流程尽量保存点，线段，方程和索引，只有进入光栅化后才生成位图。

渲染阶段位图仍保存在GPU上，不再回传CPU。Render2通过CUDA/EGL interop将GPU上的GRAY8数据写入NVIDIA Surface，再由GStreamer硬件编码器生成视频文件。

## 5. 算法特征
### 5.1 平面切片
切片的基本思想是：给定某一层的Z坐标，使用该Z平面与三角形所在平面求交。如果交线与三角形边相交，则得到该层轮廓上的交点。

对线段端点 `p0` 和 `p1`，可以使用参数方程表示：

```text
p = p0 + t * (p1 - p0)
```

当 `t` 位于 `[0,1]` 时，交点在线段内部。将该表达代入切片平面即可求得交点。

### 5.2 切片字典
若每一层都遍历全部三角形，复杂度会随层数和三角形数量共同增长。项目中在预处理阶段先计算每个三角形的Z范围，并构建切片字典。

这样在处理某一层时，只需要访问可能与该层相交的三角形，而不是全部三角形。该方法在三角形数量较大，层数较多时能显著减少无效计算。


### 5.3 Z方向局部高斯查表层归属判定
在切片阶段，少量光栅化错误翻转可能来自相邻层边界点被 `error` 容差带吸收到当前层。减小 `error` 可以减少这种现象，但容差不能无限减小，因为 STL 数据和浮点数表示本身仍然存在不可避免的数值误差。

计划中的改进思路是在切片阶段引入独立的 Z 方向层归属判定空间。真实几何坐标仍然用于最终交点计算，而层命中判断使用以当前切片平面为中心的局部高斯放大：

```text
dz = z - zPlane
u = abs(dz) * invSigma
weight = GaussianLUT[u]
dzAmplified = dz * (1 + amplify * weight)
```

其中 `GaussianLUT` 保存预先计算好的归一化高斯曲线值，例如定义在 `u in [0, 3]` 范围内。运行时 GPU 只需要将 `abs(dz)` 乘以当前尺度下的归一化系数，量化为查表索引，再读取对应权重。这样可以避免在核函数内频繁调用 `expf` 或 `powf`，同时保留局部非线性放大的效果。

该设计不改变模型真实几何。它只影响某条三角形边是否应该被当前切片层接收；真正的交点仍然使用原始坐标计算：

```text
t = (zPlane - z0) / (z1 - z0)
```

这样可以形成局部抑制机制：真正接近切片平面的点仍然会被接收，而位于容差边缘的可疑点会被局部放大后推出有效命中范围。距离当前层较远的点由于高斯权重衰减到接近 0，基本不受影响。

从性能角度，查表应保持较小规模，并尽量放在 GPU 只读数据区域，例如 CUDA constant memory。表可以在运行时初始化，使 `sigma`、放大强度和有效范围根据 `error`、`resolution` 与模型尺度调节；当参数稳定后，也可以固化为编译期静态表。

该判定规则必须在切片统计和实际切片阶段保持一致。也就是说，统计每层命中三角形数量的 kernel 和真正写入交点的 kernel 必须使用同一套命中判断，否则交点矩阵的分配数量和实际写入数量可能不一致。

### 5.4 光栅扫描线
该方法为最核心的加速策略，借助GPU的大规模并行能力，简单粗暴的以行为单位，确定轮廓内/外。

光栅化阶段使用扫描线思想判断轮廓内外。对每一行像素，统计该行与轮廓线段的交点，然后依据奇偶翻转原则填充内部区域。

该方法相比显式构建闭合拓扑树更适合GPU实现，因为它主要由线性计算，数组访问和布尔翻转构成，分支逻辑较少。

然而其容易受到浮点数误差干扰，即当模型精度不高时，会出现少许的错误线。

在传统的拓步学方法中，实际模型中经常出现多个线段共享端点的情形。如果射线法的射线刚好经过这些端点，计数会发生错判，使得轮廓层次判断出错。而扫描线算法的状态机特点，使得其完美避开了这个缺陷。


### 5.5 位图错误线后处理判定
除切片阶段的 Z 方向层归属优化外，另一类纠错思路是在光栅化后的位图上直接识别错误线。该方法面向扫描线错误翻转产生的横向异常线，尤其适用于极端模型边界层中 STL 精度不足或错层点密集导致的局部错误。

设某一层位图为：

$$
B(y,x), \quad y \in [0,H-1],\; x \in [0,W-1]
$$

其中 $H$ 为图像高度，$W$ 为图像宽度。首先扫描位图最左侧一列像素，若第 $e$ 行满足：

$$
B(e,0) \ne 0
$$

则将该行记录为潜在错误线，记潜在错误线集合为：

$$
\mathcal{E}=\{e\mid B(e,0)\ne 0\}
$$

对潜在错误线，从 $y$ 轴最高值向下寻找第一个潜在错误行 $y=e$。取其上一条正常行为参考行：

$$
y_n=e+1, \quad y_n \notin \mathcal{E}
$$

对当前潜在错误行和上一正常行在所有 $x_m$ 坐标上的像素值进行比较，定义判定量：

$$
S(e)=\sum_{m=0}^{W-1}\left[-B(e,x_m)+B(e+1,x_m)\right]
$$

同时定义潜在错误行自身的像素总量：

$$
P(e)=\sum_{m=0}^{W-1}B(e,x_m)
$$

若满足：

$$
S(e)>P(e)
$$

则判定第 $e$ 行为真正错误行：

$$
e\in\mathcal{E}_{real}
$$

该判据的核心思想是：错误翻转行通常在最左侧边界处异常非零，并且其整行像素分布与上一正常行之间存在明显反相关系。通过对潜在错误行取相反数并与上一正常行相加，可以将这种异常差异转换为可比较的全行统计量。当统计量超过该行自身像素总量时，说明该行更可能是由错误翻转产生，而不是模型真实轮廓。

识别到真正错误行后，可以进一步执行行级修复，例如使用上一正常行、下一正常行或相邻层对应行替代该错误行。若某一位图中错误行数量过多，则可将该层判定为异常边界层，并考虑直接使用相邻层位图替代。这样会损失一层 Z 方向精度，但在 STL 精度不足或极端边界变化场景下，通常比保留大面积错误翻转更可接受。


#### 错误线重建
在识别出真正错误行后，需要对错误行进行高效重建。设从上往下扫描得到的第一个错误行为：

$$
y=e
$$

其上一行为正常行：

$$
y_u=e+1
$$

若上一正常行包含 $i$ 个轮廓区间，则该行共有 $2i$ 个轮廓端点。记上方正常行的第 $k$ 个端点为：

$$
X_{u,k}, \quad k=1,2,\dots,2i
$$

错误行下方距离 $n$ 行处存在另一正常行：

$$
y_b=e-n
$$

若该下方正常行也包含 $i$ 个轮廓区间，即同样具有 $2i$ 个端点，则认为该错误行满足第一型重建条件。记下方正常行第 $k$ 个端点为：

$$
X_{b,k}, \quad k=1,2,\dots,2i
$$

对于待重建错误行的第 $j$ 个轮廓区间，其左端点为第 $2j-1$ 个端点，右端点为第 $2j$ 个端点：

$$
X_{e,2j-1},\quad X_{e,2j}, \quad j=1,2,\dots,i
$$

第一型重建使用上方正常行和下方正常行的对应端点进行线性插值。对任意端点 $k$，有：

$$
X_{e,k}=\frac{nX_{u,k}+X_{b,k}}{n+1}, \quad k=1,2,\dots,2i
$$

因此第 $j$ 个重建轮廓区间为：

$$
\left[X_{e,2j-1},\;X_{e,2j}\right]
$$

当上方正常行与下方正常行的轮廓数量不一致时，认为该错误行不满足第一型重建条件。此时采用第二情形处理，直接以上方正常行覆盖错误行：

$$
B(e,x)=B(e+1,x), \quad x=0,1,\dots,W-1
$$

若多个错误行连续出现，则按 $y$ 方向自上而下递推处理。对连续错误行集合：

$$
\mathcal{C}=\{e,e-1,e-2,\dots,e-r+1\}
$$

先使用上方正常行重建最高的错误行 $e$，随后将已重建的上一行作为新的上方参考行，继续重建下一错误行。对于第 $q$ 个错误行：

$$
y_q=e-q, \quad q=0,1,\dots,r-1
$$

其上方参考行可写为：

$$
y_{u,q}=y_q+1
$$

若下方对应正常行存在且轮廓数量一致，则使用同样的端点插值公式递推计算；否则使用上一参考行复制覆盖。该递推策略可以在连续错误线场景下逐行恢复轮廓，同时保持重建结果在 $y$ 方向上的连续性。

#### 当前 `errorRepair` 实现状态与改进方向
当前 `errorRepair` 模块处于草稿实现阶段，目标是在 `Rasterization` 生成当前批次GPU位图之后，`Render2` 推入视频管线之前，对少量扫描线错误翻转进行后处理修复。模块计划由以下阶段组成：

1. `FirstDetectErrorLines`：在GPU上快速扫描每层位图边界列，记录潜在错误线。
2. `SecondDetectErrorLines`：对潜在错误线进行行级统计比较，将真正错误线标记为确认状态。
3. `MemoryShrink`：压缩错误线数组，只保留有效错误线。
4. `MemoryFindLayer`：按层整理错误线，为每个错误层记录起始位置、层号和错误线数量。
5. `RepairErrorLinesEz`：处理少量错误线的简单层，复杂层保留给CPU或后续重修复流程。

该模块需要特别区分主机端指针表和设备端指针表。当前主流程中，`Rasterization::ProcessData` 生成的批次位图通常表现为：

```text
host pointer table + device bitmap buffers
```

也就是说，`uint8_t**` 指针表本身位于CPU内存，而每个表项指向一张GPU位图。若直接将该主机端指针表传入核函数，GPU线程访问 `d_BitmapBatch[layer]` 时会把主机指针表当作设备内存访问，从而产生错误或不稳定行为。因此，`RepairProcess` 应先在主机侧构建设备端指针表：

```cpp
uint8_t** d_bitmapTable = nullptr;
cudaMalloc(&d_bitmapTable, chunkSize * sizeof(uint8_t*));
cudaMemcpy(d_bitmapTable,
           h_bitmapTable,
           chunkSize * sizeof(uint8_t*),
           cudaMemcpyHostToDevice);
```

需要强调的是，上述复制只复制GPU位图指针值，不复制位图数据本身。随后所有错误检测和修复核函数都应使用 `d_bitmapTable`。

`RepairErrorLinesEz` 的定位应保持保守。它适合处理错误行数量较少、上下参考行容易确定、轮廓结构变化不大的普通错误层。若某层错误行数量超过阈值，例如 `EZ_ERROR_LINES`，或连续错误行过长、上下参考行轮廓数量不一致频繁出现，则应将该层标记为复杂层，交由CPU侧后处理或后续专门的重修复模块处理。这样可以避免在单个CUDA核函数中塞入过多分支、搜索和递推逻辑。

当前建议的简单修复策略是：

```text
对每个错误层并行；
每个线程负责一个错误层；
对层内少量错误行自上而下处理；
寻找上方正常行和下方正常行；
若上下参考行轮廓结构一致，则做行间插值；
若结构不一致，则复制上方正常行；
若参考行缺失，则复制可用参考行或清零。
```

在实现上，临时中间行不建议使用设备端二维指针数组，而应使用连续缓冲：

```cpp
uint32_t* middleLine = d_MiddleLine + groupIdx * width;
```

这样可以减少设备端二级指针带来的间接访问、内存不连续和指针表管理问题。核函数中也不应使用设备端 `malloc/free` 管理长期或大块临时数组；这些工作区应由 `RepairProcess` 在主机侧统一 `cudaMalloc`、统一释放。

该阶段的核心工程原则是：GPU负责发现问题和修复少量简单问题，CPU负责少数极端复杂层。这样既保留了GPU后处理的吞吐优势，又避免了将大量分支复杂度压入单个核函数。

## 6. 并行优化
### 6.1 GPU并行原则
GPU适合大量简单，相似，分支少的任务。因此项目中的CUDA核函数尽量遵循以下原则：

1. 避免复杂分支和深层循环。
2. 使用连续内存提高访问效率。
3. 将大任务拆分为可并行的小任务。
4. 核函数不主动管理长期生命周期内存，内存由外层控制函数分配和释放。
5. 中间结果尽量以数组和索引表达，而不是复杂对象表达。

### 6.2 批次处理
项目不会一次性生成全部层的位图。原因是位图数据非常大，若全部保留在显存中，会很快耗尽内存。

当前使用运行时参数 `chunk` 控制批次大小。每次只处理一部分层，生成当前批次的GPU位图，然后立即送入渲染管线。

需要区分两个概念：

1. `chunk`：用户指定的标准批次大小，用于光栅化层号映射。
2. `chunkSize`：当前批次真实帧数，最后一个批次通常小于 `chunk`。

光栅化阶段依赖 `chunk` 计算全局层号，渲染阶段依赖 `chunkSize` 判断当前批次实际要推入多少帧。二者不能混用。

### 6.3 CPU并行
在部分CPU侧预处理阶段，例如层数据重组和重合点构建，可以使用多线程或OpenMP进行并行。原则是每个线程处理独立层数据，避免多个线程同时写入同一个容器。

CPU并行主要用于减轻GPU前置准备的压力，而不是替代GPU计算。

## 7. 渲染链路
当前渲染链路使用Render2模块，目标是保持GPU数据路径，避免将大位图复制回CPU。

单帧渲染流程为：

```text
GRAY8 GPU位图 -> CUDA/EGL写入NvBufSurface -> NV12帧 -> appsrc -> nvv4l2h264enc -> h264parse -> qtmux -> filesink
```

其中GRAY8到NV12的转换在GPU上完成。Render2通过 `NvBufSurfaceMapEglImage` 将Surface映射为EGLImage，再通过CUDA注册并获得Y/UV平面指针，最后调用CUDA核函数写入NV12数据。

所有批次写入同一个GStreamer管线。也就是说，分批只是为了控制显存和计算调度，最终输出仍然是一个连续视频文件。只有在全部批次结束后，才调用最终收尾函数发送EOS并完成视频封装。


### 7.1 Jetson端DeepStream与NvBufSurface依赖
Render2依赖Jetson平台的NVIDIA多媒体与DeepStream相关组件。除了普通GStreamer开发包外，必须能够找到 `nvbufsurface.h`、`nvbufsurftransform.h` 以及对应的运行库，否则无法创建和推送 `NvBufSurface`，渲染主程序也不会被构建。

关键依赖包括：

```text
GStreamer:
  gstreamer-1.0
  gstreamer-app-1.0
  gstreamer-video-1.0
  gstreamer-allocators-1.0

NVIDIA / DeepStream:
  nvbufsurface.h
  nvbufsurftransform.h
  libnvbufsurface.so
  libnvbuf_utils.so
  libnvdsgst_helper.so
```

在Jetson设备上，建议先安装与JetPack版本匹配的DeepStream或Jetson Multimedia API。安装完成后，常见头文件位置包括：

```text
/opt/nvidia/deepstream/deepstream/sources/includes
/opt/nvidia/deepstream/deepstream-6.0/sources/includes
/usr/src/jetson_multimedia_api/include
/usr/include/aarch64-linux-gnu
/usr/include
```

常见库文件位置包括：

```text
/usr/lib/aarch64-linux-gnu/tegra
/usr/lib/aarch64-linux-gnu
/opt/nvidia/deepstream/deepstream/lib
/opt/nvidia/deepstream/deepstream-6.0/lib
```

项目中的 `gstDepends.h` 会在Jetson端包含系统安装的 `gst/gst.h`、`gst/app/gstappsrc.h`、`gst/video/video.h`、`nvbufsurface.h` 和 `nvbufsurftransform.h`。CMake会自动探测上述头文件和库文件，若缺少 `nvbufsurface`、`nvbuf_utils` 或 `nvdsgst_helper`，则会提示渲染依赖缺失，并跳过Render2主程序构建。

Render2当前稳定链路以 `NvBufSurface` 为核心。程序先创建NV12格式的Surface池，再通过CUDA/EGL interop把GRAY8位图写入Y/UV平面，最后将携带 `NvBufSurface` 元数据的 `GstBuffer` 推入 `appsrc`，交给 `nvv4l2h264enc` 完成硬件编码。因此，DeepStream/NvBufSurface并不是可选调试依赖，而是Jetson视频输出后端的关键依赖。

安装时需要先确认JetPack版本，再安装与JetPack匹配的DeepStream版本，不建议盲目安装最新版。以Jetson Xavier常见的JetPack 4.6 / DeepStream 6.0链路为例，安装步骤如下：

1. 使用NVIDIA SDK Manager或官方JetPack镜像完成Jetson系统安装，确保CUDA、TensorRT、cuDNN、Jetson Multimedia API等基础组件可用。
2. 安装GStreamer基础依赖：

```bash
sudo apt update
sudo apt install \
  libgstreamer1.0-0 \
  gstreamer1.0-tools \
  gstreamer1.0-plugins-good \
  gstreamer1.0-plugins-bad \
  gstreamer1.0-plugins-ugly \
  gstreamer1.0-libav \
  libgstrtspserver-1.0-0 \
  libgstreamer-plugins-base1.0-dev
```

3. 更新Jetson端NVIDIA多媒体和GStreamer插件：

```bash
sudo apt update
sudo apt install --reinstall nvidia-l4t-gstreamer
sudo apt install --reinstall nvidia-l4t-multimedia
sudo apt install --reinstall nvidia-l4t-core
```

4. 安装DeepStream。可以使用SDK Manager安装，也可以下载Jetson对应的tar包或deb包。tar包安装形式如下：

```bash
sudo tar -xvf deepstream_sdk_v6.0.0_jetson.tbz2 -C /
cd /opt/nvidia/deepstream/deepstream-6.0
sudo ./install.sh
sudo ldconfig
```

若使用deb包，则形式如下：

```bash
sudo apt-get install ./deepstream-6.0_6.0.0-1_arm64.deb
sudo ldconfig
```

5. 安装后检查关键文件是否存在：

```bash
find /opt/nvidia/deepstream -name nvbufsurface.h
find /opt/nvidia/deepstream -name nvbufsurftransform.h
find /usr -name 'libnvbufsurface.so*'
find /usr -name 'libnvbuf_utils.so*'
find /opt/nvidia/deepstream -name 'libnvdsgst_helper.so*'
```

6. 检查GStreamer是否能找到NVIDIA编码器：

```bash
gst-inspect-1.0 nvv4l2h264enc
gst-inspect-1.0 nvvidconv
```

7. 若 `nvv4l2h264enc`、`nvbufsurface.h` 或 `libnvbufsurface.so` 缺失，应优先检查JetPack/DeepStream版本是否匹配，以及是否安装了 `nvidia-l4t-gstreamer` 和 `nvidia-l4t-multimedia`。

安装完成后，CMake应能探测到 `NVBUF_INCLUDE_DIR`、`NVBUFUTILS_LIB`、`NVBUFSURFACE_LIB` 和 `NVDSGST_HELPER_LIB`。只有这些依赖全部存在，Render2视频后端才会被构建。

## 8. 使用方法
程序主要通过命令行参数运行。典型调用形式如下：

```bash
./Slicer <model.stl> <resolution> <output_path> <log_path> <error> <width> <height> <frame_rate> <bit_rate> <scale_x> <scale_y> <scale_z> <video_name> <chunk> <quantify_param> <auto_alternative> <threads>
```

参数含义如下：

1. `model.stl`：输入STL模型路径。
2. `resolution`：Z方向切片层厚。
3. `output_path`：输出文件目录。
4. `log_path`：日志输出目录。
5. `error`：几何误差容忍参数。
6. `width / height`：视频分辨率。
7. `frame_rate`：输出视频帧率。
8. `bit_rate`：编码码率。
9. `scale_x / scale_y / scale_z`：打印空间尺寸参数。
10. `video_name`：输出视频文件名。
11. `chunk`：每批处理的标准层数。
12. `quantify_param`：光栅化量化参数。
13. `auto_alternative`：调试模式下的自动替代参数。
14. `threads`：CPU侧并行线程数。

示例：

```bash
./SlicerRe /home/xube/Downloads/slicerAndRender-SlicingModuleReVersion/reference/sample.stl 0.05 /home/xube/Downloads/slicerAndRender-SlicingModuleReVersion/reference/output /home/xube/Downloads/slicerAndRender-For-old-version-nvcc/reference/output 0.001 3840 2160 1 80000 35.0 20.0 20.0 videoSample 128 10.0 1 4
```

## 9. 调试与输出
调试模式下程序会输出以下信息：

1. CUDA设备信息和显存状态。
2. STL读取结果和三角形数量。
3. 当前处理批次的层号范围。
4. 中间层线段和多边形调试图。
5. Render2帧写入，推流和收尾状态。

调试输出用于开发阶段定位问题。生产模式下应减少不必要的逐帧日志，以免日志I/O影响性能。

### 9.1 调试模式特点
调试模式使用宏控制，宏位于mainRe.cpp顶部，Rasterization.hpp顶部，ReProcess.hpp顶部，VectorDev.hpp顶部（该模块弃用，被Rasterization替代）。

删去宏定义，即可关闭调试模式。

调试模式下，程序会输出更详细信息，以及自动将输入的模型挪到切片窗口中间，且自动缩放到合适的大小，方便调试。

### 9.2 自动变换模式
若处于测试模式下，通常打开自动变换参数auto_alternative。生产模式下务必关闭

**注意：渲染模块由于本身较慢，且信息关键，不适用调试模式的关闭。**

## 10. 当前维护原则
当前代码已经可以完成端到端运行，并正常输出视频。因此后续维护应以稳定为优先。

建议遵循以下原则：

1.不要改动主要流程代码，尤其是关键算法部分。  
2.不要改动主要数据结构，尤其是数据结构定义。  
3.以参数优化为主，尤其是容差值，它直接控制了浮点数误差导致的错误线多寡。  
4.仅在融入前端适配时关闭调试模式  

该文档为简化项目说明，保留设计思路和当前稳定链路。更详细的开发记录仍可参考 `InstructionManual_zh.md`。

## 11. 调参
一些参数需要根据需求调整：

### 11.1 resolution 层高
即用户所期望的z轴分辨率，越小层数越多，速度越慢，z轴分辨率越精确，同时受到浮点数误差影响更大，可能出现更多错误线

### 11.2 error 误差
该值代表程序会将差距多大的坐标判定为重合，太小会丢失重合点，太大会严重放大浮点数误差，导致错误线多

### 11.3 scale_x / scale_y / scale_z 打印空间大小
实际的打印空间物理尺寸，直接影响打印结果是否能真实还原设计尺寸，务必设置正确

### 11.4 quantify_param 量化参数
该参数由前端传入，若用户不希望对模型进行缩放，请固定为1，它的作用实际上是在子坐标系变换中降低量化为整数的误差

### 11.5 chunk 批次大小
批次大小决定了程序一次性处理多少张位图，值越大处理越快，io开销越小，但是显存占用线性升高，请检查剩余显存大小决定批次大小
单帧显存占用理论计算公式为
**原始GRAY8大小 = width * height * 1byte**
**NV12 Surface大小 = yPitch * height + uvPitch * height / 2** 注：Pitch值通常取大于分辨率的2的次幂，如1920取2048
**总峰值占用 = 原始GRAY8大小 + NV12 Surface大小 * 批次大小**  
请不要让峰值内存使用超过总显存。  

### 11.6 Threads 线程数
线程数决定了在CPU侧对数据的预处理和后处理所使用的线程数，线程数越多，程序越快，但是CPU占用率越高，请根据CPU核数决定。
x86_64架构下，线程数建议取物理核心数，不超过最大线程数
ARM架构下，线程数建议取逻辑核心数，不超过最大线程数

## 12. 性能  
该项目的性能超过了设计目标，最终输出帧率在chunk分配64时为60帧左右，纯切片不包含渲染速度高达每秒3000层以上。在不同平台，不同硬件下性能差异将会很大，主要取决于GPU的性能和内存的大小，内存更大意味着能分配更大的批次空间一次性处理，降低内存分配，io开销。  
更详细的性能表现待后续测试。  







