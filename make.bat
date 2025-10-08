::+FHDR//////////////////////////////////////////////////////////////////////////////
:: Shanghai Jiao Tong University, Department of Electronic Engineering, SMIL Lab
:: Author: Yu Huang
:: Coding: UTF-8
:: Create Date: 2025.3.19
:: Description: 
:: Bat for compilation for Sandpile fractal
::
:: Revision:
:: ---------------------------------------------------------------------------------
:: [Date]         [By]         [Version]         [Change Log]
:: ---------------------------------------------------------------------------------
:: 2025/03/19     Yu Huang     1.0               First implementation
:: 2025/10/08     Yu Huang     1.1               Static link update
:: ---------------------------------------------------------------------------------
::
::-FHDR//////////////////////////////////////////////////////////////////////////////
@echo off

SETLOCAL

if "%2" == "1" (
    set LINK_TYPE="Dynamic"
    set CUDA_LIB_FILE=cudart.lib
) else (
    set LINK_TYPE="Static"
    set CUDA_LIB_FILE=cudart_static.lib
)

if %LINK_TYPE% == "Static" (
    set CL_L_FLAGr=/MT
    set CL_L_FLAGd=/MTd
    echo [make]: Compilation link type is [static link]
) else if %LINK_TYPE% == "Dynamic" (
    set CL_L_FLAGr=/MD
    set CL_L_FLAGd=/MDd
    echo [make]: Compilation link type is [dynamic link]
) else (
    echo [make]: Invalid compilation link type!
    goto :end
)

set COMPILE_THD_NUM=8
set CPP_SOURCE_FILE=.\src\*.cpp .\spdlog\*.cpp
set CUDA_SOURCE_FILE=.\cu\pile_kernel.cu
set OUTPUT_NAME=.\sandpile.exe
set SPDLOG_INCLUDE=C:\Users\12416\Desktop\C++File\Libs\spdlog\include
set SPDLOG_LIB=C:\Users\12416\Desktop\C++File\Libs\spdlog\build\Release
set NLOHMANNJSON_INCLUDE=C:\Users\12416\Desktop\C++File\Libs\nlohmannjson\single_include
set FFMPEG_INCLUDE=C:\Users\12416\Desktop\C++File\Libs\ffmpeg\include
set FFMPEG_LIB=C:\Users\12416\Desktop\C++File\Libs\ffmpeg\lib
set EIGEN_INCLUDE=C:\Users\12416\Desktop\C++File\Libs\eigen
set MIMALLOC_INCLUDE=C:\Users\12416\Desktop\C++File\Libs\mimalloc\include
set MIMALLOC_LIB=C:\Users\12416\Desktop\C++File\Libs\mimalloc\out\msvc-x64\Release
set MKL_COMPILER_LIB=C:\Users\12416\Desktop\C++File\Libs\Intel\oneAPI\compiler\2025.1\lib
set CUDA_INCLUDE=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.3\include
set CUDA_LIB=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.3\lib\x64

:cuda_build
    echo [make]: msvc cuda version build start
    if "%1" == "1" (
        set C_FLAG=/EHsc /utf-8 %CL_L_FLAGr% /DNDEBUG /GL /arch:AVX2 /Ox /openmp /DSPDLOG_COMPILED_LIB /DUSE_MIMALLOC /DMI_OVERRIDE=ON
        set NVCC_FLAG=%CL_L_FLAGr%
        echo [make]: Build mode is [Release]
    ) else (
        set C_FLAG=/EHsc /utf-8 %CL_L_FLAGd% /O2 /openmp /DSPDLOG_COMPILED_LIB /DUSE_MIMALLOC /DMI_OVERRIDE=ON
        set NVCC_FLAG=%CL_L_FLAGd%
        echo [make]: Build mode is [Debug]
    )
    nvcc -c -Xcompiler "%NVCC_FLAG%" -Xptxas -O3 %CUDA_SOURCE_FILE%
    echo [make]: cuda objects build done

    cl /MP%COMPILE_THD_NUM% %C_FLAG% %CPP_SOURCE_FILE% %CFLAG% pile_kernel.obj /Fe:"%OUTPUT_NAME%" ^
    /I"%MIMALLOC_INCLUDE%" /I"%SPDLOG_INCLUDE%" /I"%NLOHMANNJSON_INCLUDE%" /I"%FFMPEG_INCLUDE%" /I"%EIGEN_INCLUDE%" /I"%CUDA_INCLUDE%" ^
    /link -libpath:"%MIMALLOC_LIB%" mimalloc.dll.lib ^
    -libpath:"%SPDLOG_LIB%" spdlog.lib ^
    -libpath:"%FFMPEG_LIB%" avcodec.lib avdevice.lib avfilter.lib avformat.lib avutil.lib swresample.lib swscale.lib ^
    :: delete this ↓ line if you don't have Intel oneAPI installed or don't want to fully static link
    -libpath:"%MKL_COMPILER_LIB%" libiomp5md.lib ^
    -libpath:"%CUDA_LIB%" %CUDA_LIB_FILE%
    echo [make]: msvc cuda version build done
    goto :clean

:clean
    del /q *.i
    del /q *.s
    del /q *.o
    del /q *.obj
    echo [make]: clean files generated during compilation done
    goto :end

:end
    echo [make]: all jobs done, bat exits
    ENDLOCAL
