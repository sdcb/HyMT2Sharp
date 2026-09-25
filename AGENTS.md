# AGENTS.md

## Build / test
- `dotnet build HyMT2Sharp.slnx -c Release` (net10.0; assemblies are named `Sdcb.*`, e.g. `src/HyMT2Sharp.Benchmark/bin/Release/net10.0/Sdcb.HyMT2Sharp.Benchmark.exe`).
- `dotnet test tests/HyMT2Sharp.Tests -c Release`. Run it four ways when touching kernels:
  default (AVX-512/AVX2), `HYMT2SHARP_FORCE_PORTABLE=1` (Vector<T> tier, 256-bit),
  `DOTNET_EnableAVX=0` (128-bit Vector<T>, same width as ARM64/NEON, no FMA), and
  `DOTNET_EnableAVX512=0` (forces the AVX2/VNNI tier on AVX-512 machines —
  note `DOTNET_EnableAVX512F` is NOT honored on .NET 10, use `DOTNET_EnableAVX512`).

## Benchmarks (models in `D:\_\model\Hy-MT2-1.8B-*.gguf`)
- End to end: `Sdcb.HyMT2Sharp.Benchmark.exe --model <gguf> --bench-prefill 512 --bench-decode 128 --threads 8` (add `--profile` for per-op split; never put profile numbers in docs).
- Portable Q4_K micro: `--micro-vec-q4 [--micro-in N --micro-out N --micro-tokens N --threads N]` → GEMM GFLOP/s and GEMV GB/s.
- JIT codegen check: `DOTNET_JitDisasm="<MethodName>"` on the Release exe.
- Vulkan prefill per-op GPU split: `HYMT_VK_PFPROF=1` (timestamps after every prefill dispatch, aggregated per kernel). NVIDIA (sg32) prefill uses `pf_gemm_t32` / `pf_fa32` / `pf_addrms16` (glslang-built, see `Backends/Vulkan/tools/build-shaders.bat`); knobs `HYMT_VK_NOT32`, `HYMT_VK_NOFA`, `HYMT_VK_NOSWIGLU`, `HYMT_VK_T32[_QKV|_WO|_GU|_DOWN]`, `HYMT_VK_SPLITK_WO/DOWN`. RTX 3080 Ti GPU clock swings 1695–1980 MHz, so compare min-of-reps.
- The dev box has background load; 8-thread numbers swing ±10%. Compare A/B interleaved, or build the baseline in a `git worktree` and alternate runs. Decode after `--bench-prefill 512` runs at a longer context than decode alone, so only compare like with like.
