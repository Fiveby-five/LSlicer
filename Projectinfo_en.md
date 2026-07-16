# Project Information

## 1. Development Background
This project is designed to perform high-speed local slicing, rasterization, and video output for 3D models. The original Python implementation was useful for algorithm validation, but it was not efficient enough for high resolution, large layer counts, or device-side deployment. Therefore, the core workflow was reimplemented in CUDA C++.

The project currently targets NVIDIA GPU devices, especially Jetson platforms where the CPU and GPU share physical memory. Although the original target environment was Ubuntu 18.04, g++ 7.5.0, and CUDA 10.2, the code is designed to avoid unnecessary platform coupling. Except for the rendering backend, most compute modules can be migrated to traditional discrete-GPU systems.

The final output is not an image sequence, but a continuous video stream. This reduces file I/O and storage pressure when the number of sliced layers is large, and it is more suitable for storage, transmission, and subsequent processing of continuous layer data.

## 2. Design Goals
The project is designed around the following goals:

1. Maintain high slicing and rasterization throughput at 4K resolution.
2. Minimize large CPU/GPU data transfers.
3. Use vector data as the intermediate representation and generate bitmaps only at the final stage.
4. Use batch processing to control GPU memory usage instead of generating all frames at once.
5. Use the NVIDIA hardware encoding path on Jetson and push GPU bitmaps directly into the video pipeline.
6. Keep debug output available for locating slicing, rasterization, and rendering issues.

The general idea is: preserve geometric accuracy in the front end, keep intermediate data compact, and convert the result to NV12 frames only at the video backend.

## 3. Overall Architecture
The project workflow can be summarized as:

```text
STL loading -> Triangle preprocessing -> GPU slicing -> Segment/intersection reorganization -> Rasterization -> NV12 conversion -> GStreamer video encoding
```

The code modules are roughly divided into:

1. Input parsing module: reads STL files and obtains triangle vertices and normals.
2. Slicing module: intersects triangles with Z layers.
3. Rasterization module: converts each layer's segment set into a 2D bitmap.
4. Rendering module: converts GRAY8 bitmaps to NV12 and pushes them into the NVIDIA encoding pipeline.
5. Debug module: outputs intermediate layer results to help locate geometry and memory issues.
6. Command-line module: parses runtime parameters and organizes the main workflow.

The current main workflow is in `mainRe.cpp`. Slicing uses `ReProcess / ReSlicing`, rasterization uses `Rasterization`, and rendering uses `Render2`.

## 4. Data Design
### 4.1 STL Data Characteristics
An STL file only describes the outer surface of a model. It consists of many triangles, and each triangle contains three vertices and one normal vector. Therefore STL data has the following characteristics:

**Hollow**

STL stores only the outer contour and does not directly describe internal solid regions. During slicing, filled regions must be inferred from contour inside/outside relationships.

**Discrete**

Curved surfaces are discretized into many triangles. Higher model precision usually means more triangles.

**Vectorized**

Vertices, edges, normals, and plane equations can all be represented as vectors, which makes GPU parallel computation easier to organize.

### 4.2 Core Data Structures
Important data structures include:

1. Triangle structures: store triangle vertices, plane or edge parameters, and Z ranges.
2. Slicing dictionary: records possible triangle intersections for each layer. This is kept as a historical concept; the newer Re pipeline no longer relies on the old dictionary path.
3. Layer data container: `std::vector<std::vector<float3>>` stores slicing results per layer on the host, still as vector data.
4. Rasterization segment structure: reorganizes slicing intersections into segment data suitable for scanline rasterization.
5. Batch bitmap pointer table: `uint8_t**` manages GPU bitmaps for the current batch. Each frame is flat, row-major, and the frame sequence is represented by the pointer table. In the main workflow, this table is usually a host-side pointer table whose elements are GPU bitmap pointers. If a CUDA kernel needs to index the table by layer, an additional device-side pointer table must be created by the outer control function.
6. Render2 frame pool: uses `NvBufSurface` as the frame storage before video encoding.

### 4.3 Memory Design
The most important memory rule is: do not frequently transfer large data across CPU and GPU.

The slicing stage keeps vector data because the number of valid contour points in a layer is far smaller than a full bitmap. For example, a 4K GRAY8 bitmap is about 8 MB, while the vector contour data of the same layer is usually much smaller. Therefore, intermediate stages store points, segments, equations, and indices as much as possible, and generate bitmaps only when entering rasterization.

During rendering, bitmaps remain on the GPU and are not copied back to the CPU. Render2 uses CUDA/EGL interop to write GPU GRAY8 data into NVIDIA surfaces, then uses the GStreamer hardware encoder to generate the video file.

## 5. Algorithm Characteristics
### 5.1 Plane Slicing
The basic slicing idea is: given a Z coordinate for a layer, intersect the Z plane with triangle edges. If the intersection lies on an edge, that point becomes part of the contour for that layer.

For an edge with endpoints `p0` and `p1`, the parametric form is:

```text
p = p0 + t * (p1 - p0)
```

When `t` lies in `[0, 1]`, the intersection point is inside the segment. Substituting this expression into the slicing plane gives the intersection point.

### 5.2 Slicing Dictionary
If every layer scans every triangle, complexity grows with both layer count and triangle count. The original design computed the Z range of each triangle during preprocessing and built a slicing dictionary so that each layer only visited triangles that could intersect it.

This concept is still useful for understanding the old optimization strategy. The current Re slicing path uses a simpler direct layer-parallel workflow with `TriElements` and explicit Z-range checks.


### 5.3 Z-Layer Local Gaussian Lookup for Slicing Stability
During slicing, a small number of erroneous rasterization flips can be caused by neighboring-layer boundary points being absorbed into the current layer by the tolerance band. Reducing `error` can reduce this effect, but the tolerance cannot be reduced indefinitely because STL data and floating-point representation still contain unavoidable numerical error.

A planned improvement is to introduce a separate Z-layer classification space. The original geometric coordinates are still used for the actual intersection calculation, while the classification step uses a local Gaussian-based amplification around the current slicing plane:

```text
dz = z - zPlane
u = abs(dz) * invSigma
weight = GaussianLUT[u]
dzAmplified = dz * (1 + amplify * weight)
```

The lookup table stores precomputed values of a normalized Gaussian curve, for example over `u in [0, 3]`. At runtime, the GPU only needs to quantize `abs(dz)` into a table index and read the corresponding weight. This avoids expensive per-thread `expf` or `powf` calls while preserving the desired local nonlinear behavior.

The purpose of this design is not to modify the model geometry. It only affects whether a triangle edge should be accepted by the current slicing layer. The actual intersection point is still computed from the original coordinates:

```text
t = (zPlane - z0) / (z1 - z0)
```

This gives the algorithm a local suppression mechanism: points truly near the slicing plane can still be accepted, while suspicious points near the edge of the tolerance band can be pushed outside the effective hit range. Farther points receive almost no amplification because the Gaussian weight decays to zero.

For performance, the table should preferably be small and stored as read-only GPU data, such as CUDA constant memory. The table may be initialized at runtime so that `sigma`, amplification strength, and the valid range can be tuned according to `error`, `resolution`, and model scale. Once the parameters are stable, the table can also be fixed at compile time.

This idea should be applied consistently in the slicing hit tests. The triangle-counting kernel and the actual slicing kernel must use the same hit rule; otherwise, the allocated intersection matrix may not match the number of written results.

### 5.4 Raster Scanline
Raster scanline is the core acceleration strategy. It uses GPU parallelism to determine inside/outside regions by row.

For each image row, the algorithm records intersections between the scanline and contour segments, then fills the row according to odd-even parity. Compared with explicitly building contour topology, this method is more suitable for GPU execution because it mainly consists of linear computation, array access, and boolean state changes.

However, the method is sensitive to floating-point error. When model precision is insufficient, a few erroneous lines may appear. The state-machine nature of the scanline method avoids many endpoint-counting problems found in traditional ray/topology approaches, especially when multiple segments share vertices.


### 5.5 Bitmap Error-Line Post-Processing Criterion
Besides Z-layer classification improvements during slicing, another correction strategy is to detect error lines directly on the rasterized bitmap. This method targets horizontal abnormal lines caused by scanline parity errors, especially in extreme boundary layers where STL precision is insufficient or misplaced neighboring-layer points are dense.

Let one layer bitmap be:

$$
B(y,x), \quad y \in [0,H-1],\; x \in [0,W-1]
$$

where $H$ is image height and $W$ is image width. First scan the leftmost column. If row $e$ satisfies:

$$
B(e,0) \ne 0
$$

then this row is recorded as a potential error line. The potential error-line set is:

$$
\mathcal{E}=\{e\mid B(e,0)\ne 0\}
$$

For potential error lines, search downward from the highest Y value and take the first potential error row $y=e$. Its previous normal row is used as the reference row:

$$
y_n=e+1, \quad y_n \notin \mathcal{E}
$$

For all $x_m$ positions, define the comparison statistic between the potential error row and the previous normal row:

$$
S(e)=\sum_{m=0}^{W-1}\left[-B(e,x_m)+B(e+1,x_m)\right]
$$

Also define the pixel sum of the potential error row itself:

$$
P(e)=\sum_{m=0}^{W-1}B(e,x_m)
$$

If:

$$
S(e)>P(e)
$$

then row $e$ is classified as a real error line:

$$
e\in\mathcal{E}_{real}
$$

The core idea is that a parity-flip error line is usually abnormally nonzero at the left boundary, and its row distribution has an obvious inverse relationship with the previous normal row. By negating the potential error row and adding the corresponding pixels from the previous normal row, this abnormal relation becomes a row-level statistic. When the statistic exceeds the potential row's own pixel sum, the row is more likely caused by an erroneous flip than by the true model contour.

After a real error line is detected, row-level repair can be applied by replacing it with the previous normal row, the next normal row, or a corresponding row from a neighboring layer. If a bitmap contains too many real error lines, the whole layer can be treated as an abnormal boundary layer and replaced by a neighboring layer. This sacrifices one layer of Z precision, but in cases where STL precision is insufficient or the boundary changes sharply, it is usually preferable to keeping a large erroneous flip region.


#### Error-Line Reconstruction
After real error lines are detected, the next step is to reconstruct them efficiently. Let the first error row found by scanning from top to bottom be:

$$
y=e
$$

Its previous row is a normal row:

$$
y_u=e+1
$$

If the previous normal row contains $i$ contour intervals, then it has $2i$ contour endpoints. Let the $k$-th endpoint on the upper normal row be:

$$
X_{u,k}, \quad k=1,2,\dots,2i
$$

Assume another normal row exists below the error row at distance $n$:

$$
y_b=e-n
$$

If this lower normal row also contains $i$ contour intervals, meaning it also has $2i$ endpoints, then the error row satisfies the first reconstruction case. Let the $k$-th endpoint on the lower normal row be:

$$
X_{b,k}, \quad k=1,2,\dots,2i
$$

For the $j$-th contour interval on the reconstructed error row, the left endpoint is the $(2j-1)$-th endpoint and the right endpoint is the $2j$-th endpoint:

$$
X_{e,2j-1},\quad X_{e,2j}, \quad j=1,2,\dots,i
$$

The first reconstruction case uses linear interpolation between corresponding endpoints of the upper and lower normal rows. For any endpoint $k$:

$$
X_{e,k}=\frac{nX_{u,k}+X_{b,k}}{n+1}, \quad k=1,2,\dots,2i
$$

Therefore, the $j$-th reconstructed contour interval is:

$$
\left[X_{e,2j-1},\;X_{e,2j}\right]
$$

If the upper normal row and the lower normal row do not contain the same number of contour intervals, the error row does not satisfy the first reconstruction case. In this second case, the error row is directly overwritten by the upper normal row:

$$
B(e,x)=B(e+1,x), \quad x=0,1,\dots,W-1
$$

If multiple error rows appear continuously, reconstruction is performed recursively from top to bottom. For a continuous error-row set:

$$
\mathcal{C}=\{e,e-1,e-2,\dots,e-r+1\}
$$

The highest error row $e$ is reconstructed first using the upper normal row. Then the already reconstructed previous row is treated as the new upper reference row for reconstructing the next error row. For the $q$-th error row:

$$
y_q=e-q, \quad q=0,1,\dots,r-1
$$

its upper reference row is:

$$
y_{u,q}=y_q+1
$$

If a corresponding lower normal row exists and has the same contour count, the same endpoint interpolation formula is applied recursively. Otherwise, the previous reference row is copied over the current error row. This recursive strategy restores continuous error-line regions row by row while keeping the reconstructed contour continuous along the Y direction.

#### Current `errorRepair` Implementation Status and Improvement Direction
The current `errorRepair` module is still a draft implementation. Its goal is to run after `Rasterization` has generated the current batch of GPU bitmaps and before `Render2` pushes those bitmaps into the video pipeline. It is designed to repair a small number of scanline parity-flip errors as a bitmap post-processing step.

The planned pipeline is:

1. `FirstDetectErrorLines`: quickly scan boundary columns of each bitmap on the GPU and record potential error rows.
2. `SecondDetectErrorLines`: confirm potential rows with row-level statistics and mark real error lines.
3. `MemoryShrink`: compact the error-line array and keep only valid error rows.
4. `MemoryFindLayer`: regroup error rows by layer and record each error layer's start index, layer index, and error-line count.
5. `RepairErrorLinesEz`: repair simple layers with only a small number of error rows. Complex layers are reserved for CPU repair or a later heavy-repair path.

This module must carefully distinguish a host pointer table from a device pointer table. In the current main workflow, the bitmap batch produced by `Rasterization::ProcessData` is usually:

```text
host pointer table + device bitmap buffers
```

That means the `uint8_t**` table itself lives in CPU memory, while each element points to a GPU bitmap. Passing this host-side table directly to a CUDA kernel is unsafe because GPU threads would index the host pointer table as if it were device memory. Therefore, `RepairProcess` should first create a device-side pointer table:

```cpp
uint8_t** d_bitmapTable = nullptr;
cudaMalloc(&d_bitmapTable, chunkSize * sizeof(uint8_t*));
cudaMemcpy(d_bitmapTable,
           h_bitmapTable,
           chunkSize * sizeof(uint8_t*),
           cudaMemcpyHostToDevice);
```

This copy transfers only the pointer values, not the bitmap data itself. All error-detection and repair kernels should then use `d_bitmapTable`.

`RepairErrorLinesEz` should remain conservative. It is suitable for ordinary error layers with only a few error rows, easy-to-find upper and lower reference rows, and relatively stable contour structure. If a layer has too many error rows, for example above `EZ_ERROR_LINES`, or if continuous error regions are too long or reference rows frequently have incompatible contour structures, that layer should be marked as complex and handled by CPU post-processing or a dedicated heavy-repair module. This avoids placing too much branching, searching, and recursive logic into a single CUDA kernel.

The recommended simple repair strategy is:

```text
Parallelize by error layer;
let one thread handle one error layer;
process the small number of error rows from top to bottom;
search for upper and lower normal reference rows;
if the reference rows have compatible contour structure, interpolate between rows;
if the structure is incompatible, copy the upper normal row;
if one reference row is missing, copy the available row or clear the row.
```

For temporary middle rows, device-side two-level pointer arrays are not recommended. A contiguous workspace is preferred:

```cpp
uint32_t* middleLine = d_MiddleLine + groupIdx * width;
```

This avoids extra pointer indirection, non-contiguous allocation, and device pointer-table management. Kernels should also avoid device-side `malloc/free` for long-lived or large temporary arrays. Workspaces should be allocated and released by `RepairProcess` on the host side.

The core engineering principle of this stage is: let the GPU detect problems and repair a small number of simple cases, while the CPU handles the rare complex boundary layers. This keeps the throughput advantage of GPU post-processing without forcing excessive branch complexity into one kernel.

## 6. Parallel Optimization
### 6.1 GPU Parallel Principles
GPU kernels in this project follow these principles:

1. Avoid complex branches and deep loops when possible.
2. Use contiguous memory to improve access efficiency.
3. Split large tasks into small independent tasks.
4. Do not let kernels manage long-lived memory; allocation and release are handled by outer control functions.
5. Express intermediate results with arrays and indices rather than complex objects.

### 6.2 Batch Processing
The project does not generate all layer bitmaps at once. Bitmaps are large, and keeping all of them in GPU memory would quickly exhaust memory.

The runtime parameter `chunk` controls batch size. Each batch generates part of the GPU bitmaps and immediately sends them into the render pipeline.

Two concepts must be separated:

1. `chunk`: the standard batch size specified by the user, also used for rasterization layer indexing.
2. `chunkSize`: the actual frame count of the current batch. The last batch is usually smaller than `chunk`.

Rasterization relies on `chunk` to recover global layer indices, while rendering relies on `chunkSize` to know how many valid frames should be pushed. These two values must not be confused.

### 6.3 CPU Parallelism
Some CPU-side preprocessing steps, such as layer reorganization and overlap construction, can use multithreading or OpenMP. The principle is that each thread processes independent layer data and avoids writing to shared containers at the same time.

CPU parallelism reduces GPU preparation overhead; it does not replace GPU computation.

## 7. Render Pipeline
The current render path uses the Render2 module. Its goal is to keep the GPU data path and avoid copying large bitmaps back to the CPU.

Single-frame render flow:

```text
GRAY8 GPU bitmap -> CUDA/EGL write into NvBufSurface -> NV12 frame -> appsrc -> nvv4l2h264enc -> h264parse -> qtmux -> filesink
```

The GRAY8 to NV12 conversion is done on the GPU. Render2 maps an `NvBufSurface` to an EGLImage with `NvBufSurfaceMapEglImage`, registers it with CUDA, obtains Y/UV plane pointers, and launches a CUDA kernel to write NV12 data.

All batches are written into the same GStreamer pipeline. Batching only controls memory usage and scheduling; the final output is still one continuous video file. EOS is sent only after all batches are finished.


### 7.1 Jetson DeepStream and NvBufSurface Dependencies
Render2 depends on NVIDIA multimedia and DeepStream components on Jetson. Besides standard GStreamer development packages, the system must provide `nvbufsurface.h`, `nvbufsurftransform.h`, and the corresponding runtime libraries. Otherwise, the program cannot create or push `NvBufSurface` objects, and the render-enabled executable will not be built.

Key dependencies include:

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

On Jetson, install the DeepStream version that matches the installed JetPack version. Do not blindly install the newest DeepStream release. For a common Jetson Xavier setup using JetPack 4.6 and DeepStream 6.0, the installation flow is:

1. Install the Jetson system with NVIDIA SDK Manager or an official JetPack image. Make sure CUDA, TensorRT, cuDNN, Jetson Multimedia API, and other base components are available.
2. Install the basic GStreamer dependencies:

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

3. Reinstall or update the NVIDIA multimedia and GStreamer plugins on Jetson:

```bash
sudo apt update
sudo apt install --reinstall nvidia-l4t-gstreamer
sudo apt install --reinstall nvidia-l4t-multimedia
sudo apt install --reinstall nvidia-l4t-core
```

4. Install DeepStream. SDK Manager can be used, or the Jetson tar/deb package can be downloaded manually. Tar package installation:

```bash
sudo tar -xvf deepstream_sdk_v6.0.0_jetson.tbz2 -C /
cd /opt/nvidia/deepstream/deepstream-6.0
sudo ./install.sh
sudo ldconfig
```

Deb package installation:

```bash
sudo apt-get install ./deepstream-6.0_6.0.0-1_arm64.deb
sudo ldconfig
```

5. Check that the required files exist:

```bash
find /opt/nvidia/deepstream -name nvbufsurface.h
find /opt/nvidia/deepstream -name nvbufsurftransform.h
find /usr -name 'libnvbufsurface.so*'
find /usr -name 'libnvbuf_utils.so*'
find /opt/nvidia/deepstream -name 'libnvdsgst_helper.so*'
```

6. Check that GStreamer can find the NVIDIA encoder and converter plugins:

```bash
gst-inspect-1.0 nvv4l2h264enc
gst-inspect-1.0 nvvidconv
```

7. If `nvv4l2h264enc`, `nvbufsurface.h`, or `libnvbufsurface.so` is missing, first check whether the JetPack and DeepStream versions match, and whether `nvidia-l4t-gstreamer` and `nvidia-l4t-multimedia` are installed.

After installation, CMake should be able to detect `NVBUF_INCLUDE_DIR`, `NVBUFUTILS_LIB`, `NVBUFSURFACE_LIB`, and `NVDSGST_HELPER_LIB`. Render2 is built only when all these dependencies are available.

## 8. Usage
The program is mainly run through command-line parameters:

```bash
./Slicer <model.stl> <resolution> <output_path> <log_path> <error> <width> <height> <frame_rate> <bit_rate> <scale_x> <scale_y> <scale_z> <video_name> <chunk> <quantify_param> <auto_alternative> <threads>
```

Parameter meanings:

1. `model.stl`: input STL path.
2. `resolution`: Z-axis slicing layer height.
3. `output_path`: output directory.
4. `log_path`: log directory.
5. `error`: geometric tolerance parameter.
6. `width / height`: video resolution.
7. `frame_rate`: output video frame rate.
8. `bit_rate`: encoding bitrate.
9. `scale_x / scale_y / scale_z`: physical print-space dimensions.
10. `video_name`: output video file name.
11. `chunk`: standard layer count per batch.
12. `quantify_param`: rasterization quantization parameter.
13. `auto_alternative`: debug-mode automatic transform parameter.
14. `threads`: CPU-side thread count.

Example:

```bash
./SlicerRe /home/xube/Downloads/slicerAndRender-SlicingModuleReVersion/reference/sample.stl 0.05 /home/xube/Downloads/slicerAndRender-SlicingModuleReVersion/reference/output /home/xube/Downloads/slicerAndRender-For-old-version-nvcc/reference/output 0.001 3840 2160 1 80000 35.0 20.0 20.0 videoSample 128 10.0 1 4
```

## 9. Debugging and Output
In debug mode, the program outputs:

1. CUDA device information and memory status.
2. STL loading results and triangle count.
3. Layer range of each processed batch.
4. Intermediate segment and polygon debug images.
5. Render2 frame writing, pushing, and finalization status.

Debug output is for development. In production mode, unnecessary per-frame logs should be reduced to avoid I/O overhead.

### 9.1 Debug Mode Characteristics
Debug mode is controlled by macros located near the top of `Rasterization.hpp`, `ReProcess.hpp`, and `VectorDev.hpp` (the latter is deprecated and replaced by `Rasterization`).

Removing the macro definition disables debug mode.

In debug mode, the program prints more detailed information, automatically moves the input model into the slicing window, and scales it to a convenient size for debugging.

### 9.2 Automatic Transform Mode
In test mode, `auto_alternative` is usually enabled. It must be disabled in production mode.

**Note: the render module itself is relatively slow and its information is important, so it is not controlled by the debug-mode switch in the same way.**

## 10. Current Maintenance Principles
The current code can complete end-to-end execution and output video normally. Future maintenance should prioritize stability.

Recommended principles:

1. Do not modify the main process code unless necessary, especially core algorithm sections.
2. Do not modify major data structures unless necessary.
3. Prefer parameter tuning, especially the tolerance value, which directly affects how many floating-point-error lines appear.
4. Disable debug mode only when integrating with the front-end production workflow.

This document is a simplified project description that keeps the design ideas and the current stable workflow. More detailed development records are in `InstructionManual_zh.md`.

## 11. Parameter Tuning
Some parameters should be adjusted according to usage requirements.

### 11.1 `resolution` Layer Height
This is the desired Z-axis resolution. A smaller value produces more layers, runs slower, gives more precise Z resolution, and is more sensitive to floating-point error, which may produce more erroneous lines.

### 11.2 `error` Tolerance
This value defines how close coordinates must be to be considered coincident. If it is too small, coincident points may be missed. If it is too large, floating-point error may be amplified and generate more erroneous lines.

### 11.3 `scale_x / scale_y / scale_z` Print Space Size
These are the physical dimensions of the actual print space. They directly affect whether the printed result restores the intended design size, so they must be set correctly.

### 11.4 `quantify_param` Quantization Parameter
This parameter is provided by the front end. If the user does not want the model to be scaled, keep it at `1`. Its actual purpose is to reduce integer quantization error through the rasterization sub-coordinate transform.

### 11.5 `chunk` Batch Size
Batch size determines how many bitmaps are processed at once. A larger value is faster and reduces I/O overhead, but GPU memory usage increases linearly. Choose it according to available GPU memory.

The theoretical single-frame memory usage is:

**Original GRAY8 size = width * height * 1 byte**

**NV12 Surface size = yPitch * height + uvPitch * height / 2**

Note: pitch is often aligned to a power of two greater than the resolution, e.g. 1920 may align to 2048.

**Total peak usage = Original GRAY8 size + NV12 Surface size * batch size**

### 11.6 `threads` Thread Count
This controls how many CPU threads are used for preprocessing and postprocessing. More threads improve speed but increase CPU usage. Choose according to CPU core count.  

On x86_64, the recommended value is the physical core count, not exceeding the maximum thread count.  

On ARM, the recommended value is the logical core count, not exceeding the maximum thread count.  





