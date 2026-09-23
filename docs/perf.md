# HyMT2Sharp 性能：Ryzen 7 5800X 实测

测量日期：2026-09-23（UTC+8）。主角是 **HyMT2Sharp**（纯 C# CPU 推理），llama.cpp 纯 CPU 构建作为同窗口对照。本页统一 **prefill 512 / decode 128**，不计模型加载与 warmup。

与其他文档的关系：[engine-benchmark.md](engine-benchmark.md) 是同一台 5800X 上的早期跨引擎大表（含 TensorSharp 与 GPU 对照）；[perf-B.md](perf-B.md) 是另一台开发机（hybrid CPU）上 Q6/Q8 的验收数据，本页第 4 节引用其数字做跨机器对比。

## 1. 环境

| 项         | 值                                                                   |
| ---------- | -------------------------------------------------------------------- |
| CPU        | AMD Ryzen 7 5800X，8 核 / 16 线程，Zen 3                              |
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
- **Decode**：5800X 反而小幅领先。decode 是内存带宽墙（第 5 节），两台机器有效带宽同量级，差异落在 perf-B 自报的噪声范围内。

## 5. 带宽与读数注意

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

## 6. 复现

```powershell
dotnet run --project src/HyMT2Sharp.Benchmark -c Release -- `
  --model "D:\_\model\Hy-MT2-1.8B-Q4_K_M.gguf" --bench-prefill 512 --bench-decode 128 --threads 8

llama-bench.exe -m "D:\_\model\Hy-MT2-1.8B-Q4_K_M.gguf" -p 512 -n 128 -t 8 -ngl 0 -dev none -r 3
```

其余量化换 `--model` 路径即可。`--micro-q8` 输出 packed GEMV 与只读带宽微基准；`--profile` 输出逐 op 分项耗时。
