# HyMT2Sharp 性能：Apple M4（虚拟机）ARM64 实测

测量日期：2026-09-24。主角是 **HyMT2Sharp** 新增的 **AdvSimd+SDOT 档**（本 PR），`Vector<T>` portable 档与 llama.cpp 纯 CPU 构建作为同窗口对照。本页统一 **prefill 512 / decode 128、6 线程**，不计模型加载与 warmup。

与其他文档的关系：[perf.md](perf.md) 是 5800X 上的 x64 数据（含 AVX2 与 portable 档细节），本页是 ARM64 侧的对应实测。

## 1. 环境

| 项         | 值                                                              |
| ---------- | --------------------------------------------------------------- |
| CPU        | Apple M4 (Virtual)，8 个同质 vCPU（`hw.perflevel0`=Standard，无 P/E 分级） |
| OS         | macOS 26.5.2 arm64，Release 构建，`net10.0`，.NET SDK 10.0.401  |
| 内存       | 16 GiB                                                          |
| 线程       | **6**（8 线程在本 VM 上是悬崖，见第 4 节）                        |
| SIMD       | AdvSimd=True，Dp/SDOT=True，无 SVE                                |
| HyMT2Sharp | git `65f9897`（本 PR 分支）                                      |
| llama.cpp  | brew stable 0.4.1，纯 CPU（`-ngl 0`）                             |

| 量化   | 文件                            |     大小 |
| ------ | ------------------------------- | -------: |
| STQ1_0 | `Hy-MT2-1.8B-1.25Bit.gguf`      | 441 MiB  |
| Q2_0C  | `Hy-MT2-1.8B-2Bit.gguf`         | 573 MiB  |
| Q4_K_M | `Hy-MT2-1.8B-Q4_K_M.gguf`       | 1.05 GiB |
| Q6_K   | `Hy-MT2-1.8B-Q6_K.gguf`         | 1.37 GiB |
| Q8_0   | `Hy-MT2-1.8B-Q8_0.gguf`         | 1.77 GiB |

## 2. 主表：三路径对比（同机同窗口，6 线程）

HyMT2Sharp 分发顺序 **AVX-VNNI → AVX2 → AdvSimd/SDOT → `Vector<float>`**。ARM64 上 SDOT（`Dp`，`vdotq_s32`）同时打开 int8 panel 预打包（`UsePanels`）；`Vector<float>` 档用 `HYMT2SHARP_FORCE_PORTABLE=1` 强制测得。llama.cpp 加载不了 STQ1_0/Q2_0C，只对照三种 K 量化。

### prefill 512（tok/s）

| 量化   | Vector\<float\> | AdvSimd+SDOT | llama.cpp | SDOT/Vector | SDOT/llama |
| ------ | --------------: | -----------: | --------: | ----------: | ---------: |
| STQ1_0 |           80.77 |   **153.16** |       n/a |      1.90×  |        n/a |
| Q2_0C  |           82.21 |   **158.91** |       n/a |      1.93×  |        n/a |
| Q4_K_M |           82.43 |   **171.23** |      86.4 |      2.08×  |   **1.98×** |
| Q6_K   |           82.49 |   **227.23** |    120.27 |      2.75×  |   **1.89×** |
| Q8_0   |           82.10 |   **215.50** |    208.24 |      2.62×  |   **1.04×** |

### decode 128（tok/s）

| 量化   | Vector\<float\> | AdvSimd+SDOT | llama.cpp | SDOT/Vector | SDOT/llama |
| ------ | --------------: | -----------: | --------: | ----------: | ---------: |
| STQ1_0 |            4.52 |    **55.88** |       n/a |     12.4×   |        n/a |
| Q2_0C  |           15.27 |    **63.85** |       n/a |      4.18×  |        n/a |
| Q4_K_M |           27.79 |    **46.32** |     51.26 |      1.67×  |      0.90× |
| Q6_K   |           21.83 |    **44.68** |     44.69 |      2.05×  |      1.00× |
| Q8_0   |           19.37 |    **39.17** |     47.55 |      2.02×  |      0.82× |

- **Prefill**：SDOT 一条指令做 16 个 int8 乘加（float FMA 只有 4 个），所有量化收敛在 150–230 tok/s，为 portable float 档的 1.9–2.8×、llama.cpp 的 1.0–2.0×。Q6/Q8 收益最大（权重读带宽低、纯 SDOT 路径无校正项）。
- **Decode**：瓶颈从 portable 的 ALU-bound（`Vector<T>` 表达不了 sdot，只能 u16 lane 拆位）翻转为带宽/权重流式读取。STQ 提升 12 倍最夸张（portable 下 stride-16 布局接近半标量）。Q4_K_M decode 46 tok/s ≈ 30 GB/s 有效带宽，Q8_0 39 tok/s ≈ 45 GB/s，已贴 VM 实际带宽下限（第 4 节）；llama.cpp 的 Q8 decode 仍快 ~20%，差距在其 GEMV 的 load 排布。

## 3. 实现要点

- **Decode GEMV**：5 种量化全部 SDOT。Q6_K 在 `RepackQ6K.RowsNeon` 里把码值预偏移成有符号（code−32），GEMV 免掉 bsum 校正；Q2_0C 为 `2·dot − 3·Σa`，STQ1_0 为 `dot − Σa`（`TotalSums` 的三重加只用于 Q2）。
- **Prefill GEMM**：4 token × 8 列 int8 SDOT 面板内核，激活复用 `BlockQ8Kx4`；`PairAdd` 输出即自然列序 `[c0..c3]`，不需要 AVX2 侧的 `colPerm` 修正。Q4_K 的 scale/min 向量在 `RepackQ4K.BuildMetaNeon` 里按 NEON lane 序存放（canonical 布局保留给 `GemmScalar` 对照测试）。
- **动态领取**：全部 prefill quantize+GEMM 与 packed GEMV 切分从静态 `worker/workers` 改为 `Interlocked` 游标领取——单一优化在本 VM 上值 **prefill +18%**（145.7→172.4 tok/s @Q4_K_M），原因是 vCPU 被宿主机不均时限流时，静态切分会让最慢的核拖住整个 tile。

## 4. 本机特性：虚拟化 M4 的限流证据

直接 C+NEON intrinsic 微基准（`microbench/bench.c`）：

| 指标 | 1 线程 | 6 线程 | 8 线程 | 真机 M4 参考 |
| ---- | -----: | -----: | -----: | -----------: |
| NEON FMA 峰值 (GFLOP/s) |   69.4 |  336.9 |  413.8 | ~140/核      |
| 只读带宽 (GB/s)         |  254.0 |  134.5 |  103.3 | ~120+        |
| triad 带宽 (GB/s)       |  696.9 |  123.1 |   93.8 | —            |

- **vCPU 算力约是真机一半**：单核 FMA 69 GFLOP/s，真机 M4 P 核约 140——宿主机大概率超售。prefill 的绝对数字因此低估真机 M4 水平。
- **带宽"线程越多越慢"**：只读带宽从 1 线程 254 GB/s 跌到 8 线程 103 GB/s，说明宿主通道被共享/挤兑，非 M4 内存系统特性。
- **8 线程悬崖**：`CpuThreadPool` 的 barrier 纯自旋无让出，8 vCPU 一旦有一个被换出，其余核烧时间片空转——8 线程 decode 曾掉到 2.8 tok/s。**6 线程是甜点**；动态领取（第 3 节）进一步缓解了不均时限流。
- 结论：本页数字是"虚拟化 M4 下限"。SDOT/Vector 倍率（prefill ~2×、decode ~2×）对真机 M4 有参考价值；绝对 tok/s 在真机 M4（~120 GB/s、无超售核）上应显著更高，尤其 decode。

## 5. 复现

```bash
# HyMT2Sharp SDOT 档（默认分发，ARM64 自动命中）
./Sdcb.HyMT2Sharp.Benchmark --model <gguf> --bench-prefill 512 --bench-decode 128 --threads 6

# Vector<float> portable 档
HYMT2SHARP_FORCE_PORTABLE=1 ./Sdcb.HyMT2Sharp.Benchmark --model <gguf> --bench-prefill 512 --bench-decode 128 --threads 6

# llama.cpp 对照
llama-bench -m <same.gguf> -p 512 -n 128 -t 6 -ngl 0

# 正确性（三种模式）
dotnet test tests/HyMT2Sharp.Tests -c Release                    # 90/90
HYMT2SHARP_FORCE_PORTABLE=1 dotnet test tests/HyMT2Sharp.Tests -c Release   # 90/90
DOTNET_EnableAVX=0 dotnet test tests/HyMT2Sharp.Tests -c Release            # 90/90
```
