# SandPileCUDA: SandPile Fractal Simulator with CUDA

## Introduction

### Sandpile Fractals
The **Abelian Sandpile Model** is a classic cellular automaton that demonstrates self-organized criticality. When grains of sand are dropped onto a grid, sites become unstable and topple when exceeding a critical height, distributing grains to neighboring sites. This simple mechanism generates complex fractal patterns known as sandpile fractals, which exhibit mathematical beauty and physical significance in modeling avalanche dynamics.

![Triangular Sandpile](https://github.com/YunTing-k/SandPileCUDA/blob/master/img/pile_tri.png?raw=true)

![Quadrilateral Sandpile](https://github.com/YunTing-k/SandPileCUDA/blob/master/img/pile_quad.png?raw=true)

![Hexagonal Sandpile](https://github.com/YunTing-k/SandPileCUDA/blob/master/img/pile_hex.png?raw=true)

### Key Features
This simulation implements the sandpile model with three distinct lattice geometries:
1. **Triangular lattice** (3 neighbors)
2. **Square lattice** (4 neighbors)
3. **Hexagonal lattice** (6 neighbors)

### Performance & Visualization
- **CUDA Acceleration**: Leverages NVIDIA GPUs for massive parallel computation, enabling simulations of large grids at interactive speeds
- **FFmpeg Integration**: Directly renders simulation results as:
  - Video animations (MP4, AVI)
  - Image sequences (PNG, JPEG, BMP)

### Customization
**Fully configurable parameters** include:
 - grid size
 - lattice type
 - initial sand number
 - max iterations
 - color lut
 - ... 

---
**Note**: more details are in Parameters section

## Installation

### Prerequisites
1. **Windows OS** with Microsoft C/C++ Compiler (MSVC)
2. **CUDA Toolkit (e.g., 11.3)** installed
3. The following libraries are required:
   - spdlog (e.g., 1.15.1)
   - nlohmann JSON (e.g., 3.11.3)
   - FFmpeg (shared libraries)
   - Eigen3 (3.4.0)
   - mimalloc (e.g., 2.2.3)
4. **NVIDIA GPU** with compatible drivers

### Compilation Steps
1. Navigate to project directory
2. Execute build script with shell:
```bat
make.bat <build_type> <link_type>
```
Where `<build_type>` is:
- `1` for **Release** mode
- Any other value for **Debug** mode

Where `<link_type>` is:
- `1` for **Dynamic** link
- Any other value for **Static** link

### Compiled Program by Author
You can also download the static-linked compiled version of program provided by author (With all necessary libs). However, you must also follow the license and policy of the corresponding libs.

### Library Paths (Modify in script if needed)
```bat
SPDLOG_INCLUDE=C:\Users\admin\Desktop\C++File\Libs\spdlog\include
SPDLOG_LIB=C:\Users\admin\Desktop\C++File\Libs\spdlog\build\Release
NLOHMANNJSON_INCLUDE=C:\Users\admin\Desktop\C++File\Libs\nlohmannjson\single_include
FFMPEG_INCLUDE=C:\Users\admin\Desktop\C++File\Libs\ffmpeg\include
FFMPEG_LIB=C:\Users\admin\Desktop\C++File\Libs\ffmpeg\lib
EIGEN_INCLUDE=C:\Users\admin\Desktop\C++File\Libs\eigen
MIMALLOC_INCLUDE=C:\Users\admin\Desktop\C++File\Libs\mimalloc\include
MIMALLOC_LIB=C:\Users\admin\Desktop\C++File\Libs\mimalloc\out\msvc-x64\Release
CUDA_INCLUDE=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.3\include
CUDA_LIB=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.3\lib\x64
```
---
**Note**:
`MKL_COMPILER_LIB`=`...Intel\oneAPI\compiler\2025.1\lib` is used for static link, because openMP is used by the program. **This lib can be ignored** if you don't have Intel oneAPI installed or don't want to fully static link. For more details please check the compile script.

### Output
- Executable: `sandpile.exe` in project root
- Automatic cleanup of intermediate files (.obj, .o, etc.)

---

**Note**: Remember to
1. Replace all **absolute paths** with your actual library locations
2. Verify CUDA version matches your installed toolkit
3. Ensure all required DLLs (FFmpeg, mimalloc) are in PATH or executable directory

## Usage
1. Navigate to project directory (make sure the program is compiled propoerly)
2. Execute program with shell:
```bat
sandpile.exe <config_path>
```
Where `<config_path>` is:
- The path of the configuration files in json format
- e.g., `.\config\` (Use paths with slashes `\` in the end)

## Parameters
The simulation behavior is controlled by a JSON configuration file with the following parameters:

### Pile Parameter (pile_param.json)
| JSON Key      | Type | Allowed Values           | Example | Description                                                                 |
|---------------|------|--------------------------|---------|-----------------------------------------------------------------------------|
| `shape`       | int  | `1`, `2`, `3`            | 1       | Grid geometry: `1`=Triangular, `2`=Quadrilateral (Square), `3`=Hexagonal    |
| `width`       | int  | `1~INT_MAX`                 | 1920     | Grid width in cells (must be ≥ 2)                                           |
| `height`      | int  | `1~INT_MAX`                 | 1080     | Grid height in cells (must be ≥ 2)                                          |
| `ini_sand_num`| int  | `1~INT_MAX`             | 12000000   | Initial sand grains placed at center cell                                   |

### Pile Parameter (sim_param.json)

| JSON Key         | Type       | Allowed Values / Constraints                      | Example           | Description                                                                 |
|------------------|------------|--------------------------------------------------|-------------------|-----------------------------------------------------------------------------|
| `max_threads`  | integer    | `1` to `hw_allowed`                      | `20`           | Maximum thread number for postprocess          |
| `max_itr_steps`  | integer    | `0` to `INT_MAX`                      | `20000`           | Maximum simulation iterations (0 = run until stable)                        |
| `data_path`      | string     | Valid directory path                             | `"C:/.../data/"`  | Output directory for binary data files                                      |
| `video_path`     | string     | Valid file path with extension                   | `"C:/.../output.mp4"` | Output video file path                                                     |
| `sp_rate`        | integer    | `1~INT_MAX`                                 | `10000`           | Frame sampling interval (capture every N iterations)                        |
| `visualize_cuda` | boolean    | `true`/`false`                                   | `true`            | Enable real-time CUDA visualization during simulation                       |
| `save_bin`       | boolean    | `true`/`false`                                   | `false`            | Save raw binary simulation data                                             |
| `outseq_format`  | integer    | `0`-`4` (0=JPEG,1=PNG,2=TIFF,3=BMP,4=JPEG2000)   | `1`               | Output image format for frame sequences                                     |
| `save_seq`       | boolean    | `true`/`false`                                   | `true`            | Save frame sequence as individual images                                    |
| `lut_r1`         | int[3]     | `0-255` for each element                         | `[182,240,247]`   | Triangular grid color LUT (R components, 3 stops)                      |
| `lut_g1`         | int[3]     | `0-255` for each element                         | `[78,120,235]`    | Triangular grid color LUT (G components)                               |
| `lut_b1`         | int[3]     | `0-255` for each element                         | `[62,61,141]`     | Triangular grid color LUT (B components)                               |
| `lut_r2`         | int[4]     | `0-255` for each element                         | `[182,255,161,240]`| Quadrilateral grid color LUT (R components, 4 stops)                   |
| `lut_g2`         | int[4]     | `0-255` for each element                         | `[78,255,193,193]`| Quadrilateral grid color LUT (G components)                            |
| `lut_b2`         | int[4]     | `0-255` for each element                         | `[62,255,159,92]` | Quadrilateral grid color LUT (B components)                            |
| `lut_r3`         | int[6]     | `0-255` for each element                         | `[182,240,247,255,76,79]`| Hexagonal grid color LUT (R components, 6 stops)                |
| `lut_g3`         | int[6]     | `0-255` for each element                         | `[78,120,235,255,139,171]`| Hexagonal grid color LUT (G components)                         |
| `lut_b3`         | int[6]     | `0-255` for each element                         | `[62,61,141,255,60,181]`| Hexagonal grid color LUT (B components)                          |
| `fresh_rate`     | integer    | `1-FFmpeg_allowed`                                         | `60`              | Video framerate (frames per second)                                         |
| `bit_rate`       | long       | `100000` (100kbps) to `100000000` (100Mbps)       | `100000000`       | Target video bitrate (bits per second)                                      |
| `rc_max_rate`    | long       | Same as bit_rate constraints                     | `100000000`       | Maximum bitrate for rate control                                            |
| `rc_min_rate`    | long       | Same as bit_rate constraints                     | `100000000`       | Minimum bitrate for rate control                                            |
| `rc_buffer_size` | integer    | `10000` (10kbits) to `100000000` (100Mbits)      | `100000000`       | Decoder buffer size (bits)                                                  |
| `gop_size`       | integer    | `0` (keyframes only) to `INT_MAX`                | `10`              | Distance between keyframes (Group of Pictures size)                         |
| `max_b_frames`   | integer    | `0-FFmpeg_allowed`                                           | `0`               | Number of consecutive B-frames (0 = disable)                                |
| `thread_count`   | integer    | `0` (auto) or `hw_allowed`                             | `0`               | Video encoding threads (0 = auto-detect)                                    |

### Implementation Notes
1. **Color LUTs**:
   - Three sets for different grid shapes:
     - `*1`: Triangular (3-value LUT)
     - `*2`: Quadrilateral (4-value LUT)
     - `*3`: Hexagonal (6-value LUT)
   - Stored in C++ as fixed-size arrays:
     ```cpp
     int lut_r[6];  // Index 0-2: TRI, 0-3: QUAD, 0-5: HEX
     int lut_g[6];
     int lut_b[6];
     ```
   - Unused array elements ignored for smaller gradients

2. **Video Encoding**:
   - Recommended settings:
     ```json
     {
       "fresh_rate": 60,
       "bit_rate": 100000000,
       "gop_size": 10,
       "max_b_frames": 0
     }
     ```
   - Set `thread_count=0` for optimal performance
   - The format of the video is decided by the `video_path` in sim_param.json

3. **Path Specifications**:
   - Use absolute paths with forward slashes `/`
   - Include trailing slash for directories:
     ```json
     "data_path": "C:/any_path/data/",
     "video_path": "C:/any_path/output.mp4"
     ```

4. **Format Mapping**:
   ```cpp
   outseq_format = 0 → SEQUENCE_JPEG
   outseq_format = 1 → SEQUENCE_PNG   // Default
   outseq_format = 2 → SEQUENCE_TIFF
   outseq_format = 3 → SEQUENCE_BMP
   outseq_format = 4 → SEQUENCE_JPEG2000
   ```
