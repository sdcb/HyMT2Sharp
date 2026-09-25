# HyMT2Sharp 性能：Ryzen 7 5800X 实测

测量日期：2026-09-23（UTC+8）。主角是 **HyMT2Sharp**（纯 C# CPU 推理），llama.cpp 纯 CPU 构建作为同窗口对照。本页统一 **prefill 512 / decode 128**，不计模型加载与 warmup。

与其他文档的关系：[engine-benchmark.md](engine-benchmark.md) 是同一台 5800X 上的早期跨引擎大表（含 TensorSharp 与 GPU 对照）；[perf-B.md](perf-B.md) 是另一台开发机（hybrid CPU）上 Q6/Q8 的验收数据，本页第 4 节引用其数字做跨机器对比。

## 1. 环境

| 项         | 值                                                                   |
| ---------- | -------------------------------------------------------------------- |
| CPU        | AMD Ryzen 7 5800X，8 核 / 16 线程，Zen 3                              |
| GPU        | NVIDIA GeForce RTX 3080 Ti 12GB（sm_86，Vulkan 1.1，驱动 581.80）       |
| OS         | Windows，Release 构建，`net10.0`                                      |
| 线程       | 8（worker 绑在 8 个物理核上，不占 SMT）                                |
| SIMD       | AVX2=True，AVX-VNNI=False，无 AVX-512                                 |
| HyMT2Sharp | git `a22b0f8`（2026-09-23）                                           |
| llama.cpp  | build **10941**，commit `4a8993735`，纯 CPU（`-dev none -ngl 0`）      |

| 量化   | 文件                            |     大小 |
| ------ | ------------------------------- | -------: |
| STQ1_0 | `Hy-MT2-1.8B-1.25Bit.gguf`      | 441 MiB  |
| Q2_0C  | `Hy-MT2-1.8B-2Bit.gguf`         | 573 MiB  |
| Q4_K_M | `Hy-MT2-1.8B-Q4_K_M.gguf`       | 1.05 GiB |
| Q6_K   | `Hy-MT2-1.8B-Q6_K.gguf`         | 1.37 GiB |
| Q8_0   | `Hy-MT2-1.8B-Q8_0.gguf`         | 1.77 GiB |

## 2. 主表：HyMT2Sharp @ 5800X

| 量化              |      prefill 512 |      decode 128 | prefill 三次          |
| ----------------- | ---------------: | --------------: | --------------------- |
| Q1.25 / STQ1_0    | **564.66 tok/s** | **47.75 tok/s** | 573.5 / 558.6 / 562.1 |
| Q2_0C             |     488.95 tok/s |     47.53 tok/s | 494.6 / 492.0 / 480.5 |
| Q4_K_M            |     423.99 tok/s |     26.55 tok/s | 426.8 / 419.8 / 425.5 |
| Q6_K              |     403.25 tok/s |     22.96 tok/s | 410.0 / 391.5 / 408.9 |
| Q8_0              |     319.81 tok/s |     16.60 tok/s | 318.5 / 319.9 / 321.0 |

- Q1.25 与 Q2 decode 基本持平且明显快于高比特量化；相比 Q4，Q1.25 prefill 快约 33%、decode 快约 80%。
- Q8 prefill 明显低于 Q4/Q6：本机无 AVX-VNNI，Q8 内积退回 `vpmaddubsw`，且 Q8 每字节权重的 unpack 成本更高。
- decode 随权重大小递减，全部贴着内存带宽墙（见第 5 节）。

## 3. 同窗口 llama.cpp 对照（同机、纯 CPU）

`llama-bench -m <same.gguf> -p 512 -n 128 -t 8 -ngl 0 -dev none -r 3`。llama.cpp 加载不了 Q2_0C / STQ1_0，只对照三种 K 量化。

| 量化   | HyMT2Sharp pp / tg | llama.cpp pp / tg      | prefill 倍率 | decode 比值 |
| ------ | -----------------: | ---------------------: | -----------: | ----------: |
| Q4_K_M |      423.99 / 26.55 | 246.24±0.82 / 31.57±0.17 |     **1.72×** |       0.84× |
| Q6_K   |      403.25 / 22.96 | 139.21±0.80 / 24.56±0.09 |     **2.90×** |       0.93× |
| Q8_0   |      319.81 / 16.60 | 130.70±0.03 / 19.05±0.19 |     **2.45×** |       0.87× |

Prefill 领先 llama.cpp 约 1.7–2.9 倍；decode 同量级、慢约 7–16%。注意这份 llama.cpp 加载了 `ggml-cuda.dll` 时 backend 一栏会误标成 CUDA，`dev=none` 且 pp512 仅 130–246 tok/s 可确认是纯 CPU 运行。

## 4. 跨机器对比：5800X vs 另一台开发机

另一台开发机数据引自 [perf-B.md](perf-B.md)：hybrid CPU（8 性能核 + 16 能效核），双通道 DDR4-3200，AVX-VNNI=True，同样 8 线程绑性能核。

| 量化   | 5800X prefill | 开发机B prefill | 比值   | 5800X decode | 开发机B decode | 比值   |
| ------ | ------------: | --------------: | -----: | -----------: | -------------: | -----: |
| Q4_K_M |        423.99 |          462.00 |  −8%   |        26.55 |          24.25 |  +9%   |
| Q6_K   |        403.25 |          463.70 | −13%   |        22.96 |          20.63 | +11%   |
| Q8_0   |        319.81 |          548.82 | −42%   |        16.60 |          15.96 |  +4%   |

- **Prefill**：Q4/Q6 两边都走 AVX2 `vpmaddubsw`，5800X 落后约 8–13%，主要是频率与微架构差。Q8 差距拉大到 −42%：开发机B 上 Q8 走 AVX-VNNI `vpdpbusd`，5800X 只能退回 `vpmaddubsw`——VNNI 缺失的影响集中在 Q8。
- **Decode**：5800X 反而小幅领先。decode 是内存带宽墙（第 6 节），两台机器有效带宽同量级，差异落在 perf-B 自报的噪声范围内。

## 5. portable SIMD（`Vector<T>`）档实测

运行时分发顺序是 AVX-VNNI → AVX2 → `Vector<T>` portable。portable 档面向 ARM64/NEON 与不支持 AVX2 的 x64，在本机可用 `HYMT2SHARP_FORCE_PORTABLE=1` 强制启用做验证。此时 `Vector<float>.Count=8`（256-bit lowering）；ARM64 上同一份代码 Count=4（128-bit NEON lowering），prefill 数字会相应更低，decode 依旧以带宽墙为主。

| 量化           | portable prefill 512 | portable decode 128 | 上一版 portable | AVX2 对照（第 2 节） |
| -------------- | -------------------: | ------------------: | --------------: | -------------------: |
| Q1.25 / STQ1_0 |          221.41 tok/s |          16.18 tok/s |   86.74 / 17.14 |      564.66 / 47.75 |
| Q2_0C          |          211.10 tok/s |          25.55 tok/s |   92.69 / 26.02 |      488.95 / 47.53 |
| Q4_K_M         |          225.24 tok/s |          27.00 tok/s |   89.56 / 17.82 |      423.99 / 26.55 |
| Q6_K           |          211.33 tok/s |          22.13 tok/s |   85.06 / 20.78 |      403.25 / 22.96 |
| Q8_0           |          231.04 tok/s |          16.85 tok/s |   87.15 / 17.16 |      319.81 / 16.60 |

- Portable prefill 走 `VecGemmF`：BLIS 式外积微内核。每个 256 宽 strip 把 2V 行（AVX2 16 行，NEON 8 行）反量化（`DequantizeBlockVec`）并转置成 `[k][2V]` panel，激活按 6 token 打包成 `[k][6]`，6×2V 个累加器整条 strip 驻留寄存器，`Vector.MultiplyAddEstimate` 生成 FMA；两个 panel 共享一次 x tile 读。单核约 84% FP32 FMA 峰值，8 线程约 800 GFLOP/s。各量化 prefill 仍收敛到同一水平（~210–230 tok/s），因为瓶颈是 float FMA 本身；再往上要走 int8 点积（ARM64 的 SDOT），`Vector<T>` 表达不了。
- Portable decode：Q4_K/Q6_K 用 `DotAct`。每次 GEMV 先把 Q8_K 激活重排成奇偶分离的 i16（`BlockQ8KAct`），权重按 u16 lane 读入，用 and/shift 直接拆出 i16 的 nibble/6-bit 值，不做跨 lane 加宽；i16 乘积对用 lane 内移位折叠成 i32，在向量里乘 scale，每行只做一次横向求和。Q4_K_M 的 decode 已追平 AVX2，两者都卡在内存带宽上。QKV 和 gate/up 在 `KQuantGemv.Multi` 里合并成一次并行分发并动态切块（支持 Q4/Q6 混合），AVX2 档同样受益。STQ/Q2/Q8 的 decode 仍是原来的 `DotVec`。
- Attention portable 路径补齐了 `Dot2x4Vec`（两行 q 共享 K 读）和 `AxpyRegVec`（输出驻留寄存器），prefill 的 QK/AV 耗时与 AVX2 相当。
- 同一时段 A/B（与上一提交相同负载，Q4_K_M）：portable 65–77 → 216–218 tok/s prefill，15.7–16.0 → 26.7–27.8 tok/s decode；AVX2 prefill 持平，decode 25.8–26.5 → 28.3–28.5 tok/s（合并的混合 GEMV）。
- `DOTNET_EnableAVX=0`（128-bit `Vector<T>`，与 NEON 同宽且无 FMA）的测试和端到端输出也已验证。
- 不设 portable 档时标量地板约为 prefill 4 / decode 2.5 tok/s（Q4_K_M，128/16 窗口实测）。

## 6. 带宽与读数注意

按 decode tok/s × 权重大小折算 HyMT2Sharp 的有效权重带宽：

| 量化   | 权重    | decode   | 有效带宽     |
| ------ | ------- | -------: | -----------: |
| Q1.25  | 441 MiB |    47.75 | ~20.6 GB/s   |
| Q2_0C  | 573 MiB |    47.53 | ~26.6 GB/s   |
| Q4_K_M | 1.05 GiB |   26.55 | ~27.9 GB/s   |
| Q6_K   | 1.37 GiB |   22.96 | ~31.4 GB/s   |
| Q8_0   | 1.77 GiB |   16.60 | ~29.4 GB/s   |

- Decode 吞吐近似严格反比于权重大小，是纯带宽限制，不是算力限制。
- 5800X 连续满载时频率与温度会漂；相邻两次运行 decode 可差 10–20%，上表是单次代表性值，排序与倍数关系在多次复测中稳定。
- 不要把带 `--profile` 的结果写进这些表。

## 7. GPU：Vulkan 后端 @ RTX 3080 Ti

同一台机器（5800X + 3080 Ti）上 Vulkan 后端的实测，第 1 节环境表新增 GPU 行。`--backend vulkan`，同样 8 线程（host 侧），prefill 走 cooperative matrix tensor core（16x16x16 fp16→fp32，subgroup 32），decode 走直接读量化块的 GEMV。权重支持 Q4_K/Q6_K/Q8_0/STQ1_0/Q2_0C（+F32）；embedding/logits 为 Q6_K 路径。

| 量化   | Vulkan pp512 / tg128 | llama.cpp CUDA pp / tg | prefill 比值 | decode 比值 | 同机 CPU（第 2 节） |
| ------ | -------------------: | ---------------------: | -----------: | ----------: | ------------------: |
| Q4_K_M |    **15,834 / 277**  |      14,378 / 336      |    **1.10×** |      0.82×  |      423.99 / 26.55 |
| Q6_K   |    **15,830 / 260**  |      13,421 / 278      |    **1.18×** |      0.94×  |      403.25 / 22.96 |
| Q8_0   |    **15,504 / 229**  |      14,928 / 255      |    **1.04×** |      0.90×  |      319.81 / 16.60 |
| STQ1_0 |    **15,958 / 180**  |          — / —         |    vs CPU 28× |  vs CPU 3.8× |      564.66 / 47.75 |
| Q2_0C  |    **16,107 / 201**  |          — / —         |    vs CPU 33× |  vs CPU 4.2× |      488.95 / 47.53 |

- Prefill 已反超同机 llama.cpp CUDA（int8 MMQ 路线）：sg32 专用 tensor-core 管线（`pf_gemm_t32` 寄存器预取流水 + `pf_fa32` 融合 flash attention + SwiGLU/残差/rmsnorm epilogue 融合 + split-K），gu GEMM 单步约 63 TFLOPS ≈ fp32-累加峰值的 80–90%。相对 CPU 是 ~37× prefill、~10× decode。
- Decode 为量化 GEMV（读各量化原始块，fp16 KV）；与 llama.cpp 的差距 6–18%，其 int8 点积 kernel 更贴近 roofline，列为后续调优项。STQ1_0/Q2_0C decode（180/201）低于 Q8（229）——每 token kernel 数恒定，小权重的带宽优势被 launch/dequant 开销吃掉了。
- STQ1_0/Q2_0C 是独家路径（llama.cpp 加载不了这两种 GGUF）：prefill 走 pf_deq 量化直读 → 同一 tensor-core GEMM 管线，所以 prefill 与 Q4–Q8 同档 ~16k；decode 用各自量化块的 GEMV。
- 正确性：Q4_K_M/STQ1_0/Q2_0C verify-decode 8/8 与 CPU top1 一致，verify-prefill 批式与逐行 max-abs=0。
- GPU 时钟在 1695–1980 MHz 间抖动（同 spv 两次可差 ±10%），对比时取 reps 最小值或同刻 A/B 交替。
- 对比用的 llama.cpp 是 CUDA 后端（`llama-bench -m <gguf> -p 512 -n 128 -t 8 -ngl 99`），第 3 节的纯 CPU 对照不受影响。

## 8. 复现

```powershell
dotnet run --project src/HyMT2Sharp.Benchmark -c Release -- `
  --model "D:\_\model\Hy-MT2-1.8B-Q4_K_M.gguf" --bench-prefill 512 --bench-decode 128 --threads 8

llama-bench.exe -m "D:\_\model\Hy-MT2-1.8B-Q4_K_M.gguf" -p 512 -n 128 -t 8 -ngl 0 -dev none -r 3

# portable（Vector<T>）档强制启用
$env:HYMT2SHARP_FORCE_PORTABLE = "1"
dotnet run --project src/HyMT2Sharp.Benchmark -c Release -- `
  --model "D:\_\model\Hy-MT2-1.8B-Q4_K_M.gguf" --bench-prefill 512 --bench-decode 128 --threads 8
```

```powershell
# Vulkan GPU 后端（3080 Ti）
dotnet run --project src/HyMT2Sharp.Benchmark -c Release -- `
  --model "D:\_\model\Hy-MT2-1.8B-Q4_K_M.gguf" --backend vulkan --bench-prefill 512 --bench-decode 128 --threads 8

llama-bench.exe -m "D:\_\model\Hy-MT2-1.8B-Q4_K_M.gguf" -p 512 -n 128 -t 8 -ngl 99
```

其余量化换 `--model` 路径即可。`--micro-q8` 输出 packed GEMV 与只读带宽微基准；`--micro-vec-q4`（可配 `--micro-in/--micro-out/--micro-tokens/--threads`）输出 portable 档 Q4_K prefill GEMM 的 GFLOP/s 与 decode GEMV 的 GB/s；`--profile` 输出逐 op 分项耗时。
