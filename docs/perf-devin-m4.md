# HyMT2Sharp 性能：Apple M4（虚拟机）ARM64 实测

测量日期：2026-09-24（Metal 行 2026-09-25 复测）。主角是 **HyMT2Sharp** 的 **AdvSimd+SDOT 档**与 **bf16 KV cache**（K/V 以 bf16 存储，64KB/token 上下文，加载时拓宽为 fp32 参与计算），`Vector<T>` portable 档与 llama.cpp 纯 CPU 构建作为同窗口对照。本页统一 **prefill 512 / decode 128、7 线程**（本机自动校准的最优点，见第 4 节），不计模型加载与 warmup。

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
| HyMT2Sharp | git `ebea3e5`（Metal fp16 MMA 分支；CPU 档与 `2b07aab` 相同）  |
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

同一 VM 的 Apple M4 paravirtual GPU（Metal only，无 bfloat，simdgroup MMA 对 float 与 half fragment 均可用）。decode 为融合图（每层 8 次 dispatch：rms 融合进 qkv 三合一 gemv→rope+rms(qnorm)+kv_append 融合→split-K attention→acc-gemv→rms→gate/up 对 gemv→acc-gemv），prefill 为 fp16 两段式 GEMM（量化权重现场反量化进共享 fp16 scratch + xT 转置 + **simdgroup half8x8 MMA `mma_gemm`** 主格 + `sgemm4x4` 补边）+ **MMA 流水化 causal attention**（K 先 `k2h` 转 fp16 行、scores=qT·Kᵀ 走 GEMM、`softmaxP` 直写 fp16 P、outᵀ=Vᵀ·P 第二个 GEMM 直写 xT3——标量 attention 与 ao+Xt 转置整步消失）；KV 为 bf16 + 64-token 块表分页（device 侧 KvBlockStore）。支持全部 5 种量化（Q2_0C/STQ1_0 的 embd 为 Q6_K，output.weight 与其绑定）。

| 量化   | Metal prefill 512 | SDOT prefill | Metal decode 128 | SDOT decode | Metal/SDOT decode |
| ------ | ----------------: | -----------: | ---------------: | ----------: | ----------------: |
| STQ1_0 |            1302.8 |     140.00   |            70.34 |      38.54  |           1.82×   |
| Q2_0C  |            1312.2 |     137.89   |            71.84 |      46.46  |           1.55×   |
| Q4_K_M |            1281.1 |     147.70   |            75.67 |      40.83  |           1.85×   |
| Q6_K   |            1292.6 |     194.86   |            71.43 |      35.87  |           1.99×   |
| Q8_0   |            1287.3 |     207.81   |            71.32 |      34.41  |           2.07×   |

- **decode ~70–76 tok/s**（与上一版融合图持平）：kernel 已贴近虚拟化 GPU 带宽墙（每 token 权重流 ~0.78GB，有效 ~55–70GB/s），本轮的 rms→qkv gemv 融合、`mv_dot` 宽读都在噪声带内。瓶颈仍是 vocab=120818 的 logits gemv（恒 Q6_K ~95GB/s、~2.6ms/token 以上）加每层 8 dispatch 序列。
- **prefill 全量化 ~1281–1312 tok/s**（fp16 改版前 893–904，+42–45%；对比最初 fp32 GEMM 版 647–650 近 2×）：两处叠加——①dequant 目标从 fp32 缓存改共享 fp16 scratch，`mma_gemm` 换 simdgroup half8x8 fragment（A 侧 fp32→fp16 暂存同时减半 threadgroup 流量），满格区 ~4.4 TFLOP/s fp16 MMA；②causal attention 整条搬上 MMA 流水线——`k2h` 把 K 转 fp16 行喂 scores GEMM、`softmaxP` 输出 zero-pad 到 kvPad 的 fp16 P、`v2f` 把 V 转 fp32 行后 outᵀ=Vᵀ·P 第二个 GEMM 直写 xT3 head 行——**ao buffer 与 Xt(ao) 转置整步省掉**。旧标量 attention 在长上下文占 ~55% 墙钟：**pp2048 Q4_K_M 1206.9 tok/s（旧路径 ~412 → 2.93×）**。寄存器内 dequant 的 mmq 式 GEMM 也实测过：融合 kernel ~3.9TF 反而低于分步 fp16 MMA 的 4.4TF，故弃用。
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
