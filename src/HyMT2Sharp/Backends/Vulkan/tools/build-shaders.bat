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
    if /i not "%%~nxf"=="pf_gemm_cm_sg32.comp" (
        echo glslc %%~nxf
        "%GLSLC%" -O --target-env=vulkan1.1 "%%f" -o "%%~dpnf.spv" || exit /b 1
    )
)
rem Parametric variants: pf_gemm_cm_g tuneables via -D (TSM/TSN/SGX/SGY/NTHR).
rem Default variant HYMT_VK_GEMMV=g28 = tile 512x64, TSM=8 TSN=2, 2x8 subgroups of 16.
"%GLSLC%" -O --target-env=vulkan1.1 -DTSM=8u -DTSN=2u -DSGX=2u -DSGY=8u -DNTHR=256 "%DIR%\pf_gemm_cm_g.comp" -o "%DIR%\pf_gemm_cm_g28.spv" || exit /b 1

rem sg32 (NVIDIA/AMD) variants — opt-in via HYMT_VK_CM32=1. These MUST be
rem built with glslang, not glslc: glslc's fp16+subgroup output produces no
rem output on NVIDIA (pf_rms16 writes all-zero rstd). pf_gemm_cm_sg32 is a
rem separate shared-staged double-buffered kernel using 16x8 acc (NVIDIA has
rem no M8 coopmat shape); the *_sg32 aux shaders are the same .comp sources.
set "GLSLANG=glslang"
if exist "C:\glslang\bin\glslang.exe" set "GLSLANG=C:\glslang\bin\glslang.exe"
"%GLSLANG%" -V --target-env vulkan1.1 -S comp "%DIR%\pf_gemm_cm_sg32.comp" -o "%DIR%\pf_gemm_cm_sg32.spv" || exit /b 1
for %%f in (pf_rms16 pf_kvprep pf_attn16 pf_silumul16) do (
    echo glslang %%~nxf_sg32
    "%GLSLANG%" -V --target-env vulkan1.1 -S comp "%DIR%\%%f.comp" -o "%DIR%\%%f_sg32.spv" || exit /b 1
)
echo done.
