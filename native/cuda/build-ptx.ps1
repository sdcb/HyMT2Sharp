# Regenerates the committed PTX under native/ptx:
#   native/ptx/hymt_kernels.ptx         (native/cuda/hymt_kernels.cu)
#   native/ptx/tensorsharp_kernels.ptx  (native/cuda/vendor/tensorsharp_kernels.cu)
#
# The PTX loads via cuModuleLoadData through the CUDA driver API at runtime —
# consumers need only nvcuda (the display driver), not the CUDA toolkit.
# compute_80 is the lowest arch that covers every feature the kernels use
# (cp.async MMQ staging, mma.m16n8k32 tensor cores) and JITs forward onto any
# newer GPU. Override with -Arch when targeting older hardware (kernels then
# lose the >=sm_80 fast paths via their __CUDA_ARCH__ guards).
#
# Requires nvcc on PATH (Windows also needs cl.exe on PATH, e.g. from a
# "x64 Native Tools Command Prompt" or by prepending the MSVC bin dir).
param(
    [string]$Arch = "compute_80",
    [string]$Nvcc = "nvcc",
    # nvcc stamps the toolkit's own PTX ISA in .version (e.g. 9.4 for CUDA 13.4);
    # the driver JIT rejects an ISA newer than itself (CUDA error 222). For
    # compute_80 targets the emitted code uses nothing newer than ISA 8.x, so
    # the .version line is rewritten to -PtxVersion (default 8.7) to keep the
    # committed PTX loadable on CUDA 12.8+ drivers. Pass "" to skip the patch.
    [string]$PtxVersion = "8.7"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$outDir = Join-Path $root "native\ptx"
New-Item -ItemType Directory -Force $outDir | Out-Null

function Build-Ptx([string]$src, [string]$dst) {
    & $Nvcc -ptx "-arch=$Arch" -o $dst $src
    if ($LASTEXITCODE -ne 0) { throw "nvcc failed for $src" }
    if ($PtxVersion -ne "") {
        (Get-Content $dst) -replace '^\.version \d+\.\d+', ".version $PtxVersion" |
            Set-Content $dst
    }
}

Build-Ptx (Join-Path $PSScriptRoot "hymt_kernels.cu") (Join-Path $outDir "hymt_kernels.ptx")
Build-Ptx (Join-Path $PSScriptRoot "vendor\tensorsharp_kernels.cu") (Join-Path $outDir "tensorsharp_kernels.ptx")
