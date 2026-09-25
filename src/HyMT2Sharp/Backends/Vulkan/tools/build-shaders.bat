@echo off
rem Dev-time only: compile .comp GLSL -> .spv next to each source (committed).
rem Not part of `dotnet build` — SPIR-V is embedded as a resource.
setlocal
set "GLSLC=C:\VulkanSDK\1.4.357.0\Bin\glslc.exe"
if not exist "%GLSLC%" for %%i in (glslc.exe) do set "GLSLC=%%~$PATH:i"
if not exist "%GLSLC%" (
    where glslc >nul 2>nul && set "GLSLC=glslc"
)
if not exist "%GLSLC%" (
    echo glslc not found — install Vulkan SDK or put glslc.exe on PATH
    exit /b 1
)
set "DIR=%~dp0..\Shaders"
for %%f in ("%DIR%\*.comp") do (
    echo glslc %%~nxf
    "%GLSLC%" -O --target-env=vulkan1.1 "%%f" -o "%%~dpnf.spv" || exit /b 1
)
rem Parametric variants: pf_gemm_cm_g tuneables via -D (TSM/TSN/SGX/SGY/NTHR).
rem Default variant HYMT_VK_GEMMV=g28 = tile 512x64, TSM=8 TSN=2, 2x8 subgroups of 16.
"%GLSLC%" -O --target-env=vulkan1.1 -DTSM=8u -DTSN=2u -DSGX=2u -DSGY=8u -DNTHR=256 "%DIR%\pf_gemm_cm_g.comp" -o "%DIR%\pf_gemm_cm_g28.spv" || exit /b 1
echo done.
