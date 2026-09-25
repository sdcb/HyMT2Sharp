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
    set "SKIP="
    for %%s in (pf_gemm_cm_sg32.comp pf_gemm_t32.comp pf_fa32.comp pf_addrms16.comp) do if /i "%%~nxf"=="%%s" set "SKIP=1"
    if not defined SKIP (
        echo glslc %%~nxf
        "%GLSLC%" -O --target-env=vulkan1.1 "%%f" -o "%%~dpnf.spv" || exit /b 1
    )
)
rem Parametric variants: pf_gemm_cm_g tuneables via -D (TSM/TSN/SGX/SGY/NTHR).
rem Default variant HYMT_VK_GEMMV=g28 = tile 512x64, TSM=8 TSN=2, 2x8 subgroups of 16.
"%GLSLC%" -O --target-env=vulkan1.1 -DTSM=8u -DTSN=2u -DSGX=2u -DSGY=8u -DNTHR=256 "%DIR%\pf_gemm_cm_g.comp" -o "%DIR%\pf_gemm_cm_g28.spv" || exit /b 1

rem sg32 (NVIDIA/AMD) variants. These MUST be built with glslang, not glslc:
rem glslc's fp16+subgroup output produces no output on NVIDIA (pf_rms16
rem writes all-zero rstd). pf_gemm_cm_sg32 is a separate shared-staged
rem double-buffered kernel using 16x8 acc (NVIDIA has no M8 coopmat shape);
rem tuned defaults BM=BN=64 KB=16 TSM=2 TSN=4 = 64x64 tile, 8 subgroups.
rem The *_sg32 aux shaders are the same .comp sources.
set "GLSLANG=glslang"
if exist "C:\glslang\bin\glslang.exe" set "GLSLANG=C:\glslang\bin\glslang.exe"
"%GLSLANG%" -V --target-env vulkan1.1 -S comp -DBM=64 -DBN=64 -DKB=16 -DTSM=2 -DTSN=4 -DNTHR=256 "%DIR%\pf_gemm_cm_sg32.comp" -o "%DIR%\pf_gemm_cm_sg32.spv" || exit /b 1
for %%f in (pf_rms16 pf_kvprep pf_attn16 pf_silumul16) do (
    echo glslang %%~nxf_sg32
    "%GLSLANG%" -V --target-env vulkan1.1 -S comp "%DIR%\%%f.comp" -o "%DIR%\%%f_sg32.spv" || exit /b 1
)
rem sg32 tensor-core prefill (NVIDIA): pf_gemm_t32 = 16x16x16 coopmat GEMM with
rem double-buffered shared tiles, SwiGLU epilogue and split-K partials; variant
rem name pf_gemm_t32_{BM}x{BN}_{WM}x{WN}[_k{BK}] (HYMT_VK_T32[_QKV|_WO|_GU|_DOWN]).
rem pf_fa32 = GQA-grouped two-pass causal attention; pf_addrms16 = split-K
rem partial fold + rmsnorm. glslang only (same fp16+subgroup glslc issue).
"%GLSLANG%" -V --target-env vulkan1.1 -S comp -DBM=64 -DBN=64 -DWM=32 -DWN=32 -DBK=32 "%DIR%\pf_gemm_t32.comp" -o "%DIR%\pf_gemm_t32_64x64_32x32.spv" || exit /b 1
"%GLSLANG%" -V --target-env vulkan1.1 -S comp "%DIR%\pf_fa32.comp" -o "%DIR%\pf_fa32.spv" || exit /b 1
"%GLSLANG%" -V --target-env vulkan1.1 -S comp "%DIR%\pf_addrms16.comp" -o "%DIR%\pf_addrms16.spv" || exit /b 1
echo done.
