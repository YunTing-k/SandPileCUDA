::+FHDR//////////////////////////////////////////////////////////////////////////////
:: Shanghai Jiao Tong University, Department of Electronic Engineering, SMIL Lab
:: Author: Yu Huang
:: Coding: UTF-8
:: Create Date: 2025.5.28
:: Description: 
:: Bat for clear data for SandPile
::
:: Revision:
:: ---------------------------------------------------------------------------------
:: [Date]         [By]         [Version]         [Change Log]
:: ---------------------------------------------------------------------------------
:: 2025/05/28     Yu Huang     1.0               First implementation
:: ---------------------------------------------------------------------------------
::
::-FHDR//////////////////////////////////////////////////////////////////////////////
@echo off

SETLOCAL

set DATA_PATH=.\data
set VIDEO_PATH=.\video

if "%1" == "1" (
    goto :clean
) else (
    goto :end
)

:clean
    del /s /q %DATA_PATH%\*.bin
    del /s /q %DATA_PATH%\*.jpg
    del /s /q %DATA_PATH%\*.png
    del /s /q %DATA_PATH%\*.tiff
    del /s /q %DATA_PATH%\*.bmp
    del /s /q %DATA_PATH%\*.jp2
    del /s /q %VIDEO_PATH%\

    echo [clean]: all files clean done
    goto :end

:end
    echo [clean]: all jobs done, bat exits
    ENDLOCAL
