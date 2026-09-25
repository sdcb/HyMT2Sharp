# HyMT2Sharp 性能：Apple M4（虚拟机）ARM64 实测

测量日期：2026-09-24。主角是 **HyMT2Sharp** 的 **AdvSimd+SDOT 档**与 **bf16 KV cache**（K/V 以 bf16 存储，64KB/token 上下文，加载时拓宽为 fp32 参与计算），`Vector<T>` portable 档与 llama.cpp 纯 CPU 构建作为同窗口对照。本页统一 **prefill 512 / decode 128、7 线程**（本机自动校准的最优点，见第 4 节），不计模型加载与 warmup。

与其他文档的关系：[perf.md](perf.md) 是 5800X 上的 x64 数据（含 AVX2 与 portable 档细节），本页是 ARM64 侧的对应实测。

## 1. 环境

| 项         | 值                                                              |
| ---------- | --------------------------------------------------------------- |
| CPU        | Apple M4 (Virtual)，8 个同质 vCPU（`hw.perflevel0`=Standard，无 P/E 分级） |
| OS         | macOS 26.5.2 arm64，Release 构建，`net10.0`，.NET SDK 10.0.401  |
| 内存       | 16 GiB                                                          |
| 线程       | **7**（启动校准选出；旧版 8 线程悬崖已由主线程参与修复，见第 4 节）  |
| SIMD       | AdvSimd=True，Dp/SDOT=True，无 SVE                                |
| KV cache   | bf16（K/V 各 64KB/token；fp32 需 128KB）                          |
| HyMT2Sharp | git `2b07aab`（bf16 KV cache 分支）                              |
| llama.cpp  | brew stable 0.4.1，纯 CPU（`-ngl 0`，KV=f16 默认）                |

| 量化   | 文件                            |     大小 |
| ------ | ------------------------------- | -------: |
| STQ1_0 | `Hy-MT2-1.8B-1.25Bit.gguf`      | 441 MiB  |
| Q2_0C  | `Hy-MT2-1.8B-2Bit.gguf`         | 573 MiB  |
| Q4_K_M | `Hy-MT2-1.8B-Q4_K_M.gguf`       | 1.05 GiB |
| Q6_K   | `Hy-MT2-1.8B-Q6_K.gguf`         | 1.37 GiB |
| Q8_0   | `Hy-MT2-1.8B-Q8_0.gguf`         | 1.77 GiB |

## 2. 主表：三路径对比（同机同窗口，7 线程）

HyMT2Sharp 分发顺序 **AVX-VNNI → AVX2 → AdvSimd/SDOT → `Vector<float>`**。ARM64 上 SDOT（`Dp`，`vdotq_s32`）同时打开 int8 panel 预打包（`UsePanels`）；`Vector<float>` 档用 `HYMT2SHARP_FORCE_PORTABLE=1` 强制测得。llama.cpp 加载不了 STQ1_0/Q2_0C，只对照三种 K 量化。

### prefill 512（tok/s）

| 量化   | Vector\<float\> | AdvSimd+SDOT | llama.cpp | SDOT/Vector | SDOT/llama |
| ------ | --------------: | -----------: | --------: | ----------: | ---------: |
| STQ1_0 |           79.45 |   **140.00** |       n/a |      1.76×  |        n/a |
| Q2_0C  |           74.82 |   **137.89** |       n/a |      1.84×  |        n/a |
| Q4_K_M |           80.38 |   **147.70** |     84.60 |      1.84×  |   **1.75×** |
| Q6_K   |           68.68 |   **194.86** |    109.57 |      2.84×  |   **1.78×** |
| Q8_0   |           77.47 |   **207.81** |    205.15 |      2.68×  |   **1.01×** |

### decode 128（tok/s）

| 量化   | Vector\<float\> | AdvSimd+SDOT | llama.cpp | SDOT/Vector | SDOT/llama |
| ------ | --------------: | -----------: | --------: | ----------: | ---------: |
| STQ1_0 |            4.13 |    **38.54** |       n/a |      9.3×   |        n/a |
| Q2_0C  |           15.00 |    **46.46** |       n/a |      3.10×  |        n/a |
| Q4_K_M |           26.99 |    **40.83** |     34.79 |      1.51×  |      1.17× |
| Q6_K   |           17.47 |    **35.87** |     33.43 |      2.05×  |      1.07× |
| Q8_0   |           19.06 |    **34.41** |     37.30 |      1.81×  |      0.92× |

- **Prefill**：SDOT 一条指令做 16 个 int8 乘加（float FMA 只有 4 个），所有量化收敛在 138–208 tok/s，为 portable float 档的 1.8–2.8×、llama.cpp 的 1.0–1.8×。Q6/Q8 收益最大（权重读带宽低、纯 SDOT 路径无校正项）。
- **Decode**：瓶颈从 portable 的 ALU-bound（`Vector<T>` 表达不了 sdot，只能 u16 lane 拆位）翻转为带宽/权重流式读取。STQ 提升 9 倍最夸张（portable 下 stride-16 布局接近半标量）。llama.cpp 的 Q8 decode 仍快 ~8%，差距在其 GEMV 的 load 排布；HyMT2Sharp 已在 Q4/Q6 反超。
- **bf16 KV 的长上下文收益**（同一构建 A/B，pp2048+decode128 @2K ctx）：AdvSimd 档 decode fp32-KV 33.10 → bf16-KV **41.21** tok/s（+24.5%），Vector 档 22.65 → **27.23**（+20.2%）；prefill 同步 +2~5%。KV 读取带宽减半，收益随上下文长度继续放大（2K ctx 时 KV 读约占 decode 时间的 1/3）；同时 2048 ctx 的 KV 内存从 512MB 降到 256MB。

### Metal GPU 后端（纯 net10.0 + libobjc P/Invoke，`--backend metal`，同日补测）

同一 VM 的 Apple M4 paravirtual GPU（Metal only，无 bfloat/simdgroup 特性，走 portable 档 kernel）。decode 为 M2 融合图（embed_gather 去 host 化，~165 dispatch/token），prefill 为两段式 fp32 GEMM（量化权重一次性反量化进 `_wfp32` 缓存 + xT 转置 + 64tok×64col/256线程 `sgemm4x4`）+ 融合 causal attention；KV 为 bf16 + 64-token 块表分页（device 侧 KvBlockStore）。支持全部 5 种量化（Q2_0C/STQ1_0 的 embd 为 Q6_K，output.weight 与其绑定）。

| 量化   | Metal prefill 512 | SDOT prefill | Metal decode 128 | SDOT decode | Metal/SDOT decode |
| ------ | ----------------: | -----------: | ---------------: | ----------: | ----------------: |
| STQ1_0 |             646.4 |     140.00   |            58.59 |      38.54  |           1.52×   |
| Q2_0C  |             650.0 |     137.89   |            59.59 |      46.46  |           1.28×   |
| Q4_K_M |             649.6 |     147.70   |            56.30 |      40.83  |           1.38×   |
| Q6_K   |             650.2 |     194.86   |            51.15 |      35.87  |           1.43×   |
| Q8_0   |             646.2 |     207.81   |            49.96 |      34.41  |           1.45×   |

- **decode 全面反超 SDOT**（1.26–1.55×），但各量化收敛在 44–60 tok/s 而非随位宽下降等比变快——瓶颈已移到与量化无关的部分：vocab=120818 的 logits gemv（embd/output 恒为 Q6_K，STQ/Q2 文件里它独占 ~44% 字节）加固定 dispatch 序列；低位量化的权重字节优势被淹没。
- **prefill 改造后全量化 ~647–650 tok/s**（原 90–211）：权重反量化从 hot loop 挪进一次性 `_wfp32` 缓存（fp32 副本，代价 ~6GB 常驻显存/共享内存），hot loop 只剩一个量化无关的 fp32 `sgemm4x4`——各量化收敛到同一吞吐即 GEMM 已到该 GPU 的算力墙（实测 fp32 FMA 峰值 ~3.9 TFLOP/s，prefill 512 约耗 ~2.3 TFLOP/s ≈ 60% 峰值；对比 MLX 同机 767 tok/s）。原"逐量化 GEMM+TILE_T=8"形态下 X 被每 4 列重读一遍、in_dim=2048 的 K-quant GEMM 只有 8/64 线程有活，是此前 ~200 的主因。
- **STQ kernel 曾撞 paravirt GPU watchdog**（~10s command buffer 上限）：初版逐 lane 标量 gather + 函数内 `const uchar CB[32]` 被编译器 spill 到私有显存，512-token prefill 25s 触发 timeout（`status=5,error=2`），之后所有 command buffer 被秒拒（`error=4`）表现为 decode "1368 tok/s" 假象。改法：CB 提为文件作用域 `constant`、内层按 4 组一批做 `float4` 向量化读 x——25s→5.7s，回到阈值内。**对低位量化 kernel 的教训：paravirt GPU 上私有地址空间的动态索引小表是性能杀手。**
- 复现：`./Sdcb.HyMT2Sharp.Benchmark --model <gguf> --bench-prefill 512 --bench-decode 128 --backend metal`；`HYMT_METAL_TIMING=1` 可看每 token 的 encode/GPU 分解。

## 3. 实现要点

- **Decode GEMV**：5 种量化全部 SDOT。Q6_K 在 `RepackQ6K.RowsNeon` 里把码值预偏移成有符号（code−32），GEMV 免掉 bsum 校正；Q2_0C 为 `2·dot − 3·Σa`，STQ1_0 为 `dot − Σa`（`TotalSums` 的三重加只用于 Q2）。
- **Prefill GEMM**：4 token × 8 列 int8 SDOT 面板内核，激活复用 `BlockQ8Kx4`；`PairAdd` 输出即自然列序 `[c0..c3]`，不需要 AVX2 侧的 `colPerm` 修正。Q4_K 的 scale/min 向量在 `RepackQ4K.BuildMetaNeon` 里按 NEON lane 序存放（canonical 布局保留给 `GemmScalar` 对照测试）。
- **动态领取**：全部 prefill quantize+GEMM 与 packed GEMV 切分从静态 `worker/workers` 改为 `Interlocked` 游标领取——单一优化在本 VM 上值 **prefill +18%**（145.7→172.4 tok/s @Q4_K_M），原因是 vCPU 被宿主机不均时限流时，静态切分会让最慢的核拖住整个 tile。
- **bf16 KV cache**：`_cacheK/_cacheV` 以 `ushort` 存 bf16（RNE 写入），attention 内核对每个 load 做 `u16→u32<<16` 拓宽成 fp32 再累加——解码几乎零成本且不可能溢出（bf16 与 fp32 同指数域）。选 bf16 而非 fp16 的原因：.NET 10 没有任何向量化 Half↔float 转换 intrinsic，fp16 只能整数位解码（~7 ops/4 lanes），实测短上下文反而更慢；bf16 只需 2 ops。
- **线程数自动校准 + 主线程参与**：`threads=0` 时启动校准探针选实测最优（本 VM 选 7）；主线程不再空转自旋等 join 而是领 slice 干活，8 线程悬崖消失（decode 3.5→43 tok/s）。

## 4. 本机特性：虚拟化 M4 的限流证据

直接 C+NEON intrinsic 微基准（`microbench/bench.c`）：

| 指标 | 1 线程 | 6 线程 | 8 线程 | 真机 M4 参考 |
| ---- | -----: | -----: | -----: | -----------: |
| NEON FMA 峰值 (GFLOP/s) |   69.4 |  336.9 |  413.8 | ~140/核      |
| 只读带宽 (GB/s)         |  254.0 |  134.5 |  103.3 | ~120+        |
| triad 带宽 (GB/s)       |  696.9 |  123.1 |   93.8 | —            |

- **vCPU 算力约是真机一半**：单核 FMA 69 GFLOP/s，真机 M4 P 核约 140——宿主机大概率超售。prefill 的绝对数字因此低估真机 M4 水平。
- **带宽"线程越多越慢"**：只读带宽从 1 线程 254 GB/s 跌到 8 线程 103 GB/s，说明宿主通道被共享/挤兑，非 M4 内存系统特性。
- **8 线程悬崖（已修复）**：根因是主线程在 `For()` 里纯自旋等 join——9 个可跑线程挤 8 个 vCPU，每轮必有一个 straggler。主线程改为参与领活后悬崖消失（8 线程 decode 2.8→43 tok/s），barrier 同时加了自旋退避兜底。**当前最优是 7 线程**（启动校准自动选出）。
- 结论：本页数字是"虚拟化 M4 下限"。SDOT/Vector 倍率（prefill ~2×、decode ~2×）对真机 M4 有参考价值；绝对 tok/s 在真机 M4（~120 GB/s、无超售核）上应显著更高，尤其 decode。

## 5. 复现

```bash
# HyMT2Sharp SDOT 档（默认分发，ARM64 自动命中）
./Sdcb.HyMT2Sharp.Benchmark --model <gguf> --bench-prefill 512 --bench-decode 128 --threads 7

# Vector<float> portable 档
HYMT2SHARP_FORCE_PORTABLE=1 ./Sdcb.HyMT2Sharp.Benchmark --model <gguf> --bench-prefill 512 --bench-decode 128 --threads 7

# llama.cpp 对照
llama-bench -m <same.gguf> -p 512 -n 128 -t 7 -ngl 0

# 正确性（三种模式）
dotnet test tests/HyMT2Sharp.Tests -c Release                    # 90/90
HYMT2SHARP_FORCE_PORTABLE=1 dotnet test tests/HyMT2Sharp.Tests -c Release   # 90/90
DOTNET_EnableAVX=0 dotnet test tests/HyMT2Sharp.Tests -c Release            # 90/90
```
