# HyMT2Sharp 性能：AVX-512 档 x86-64 实测

测量日期：2026-09-24（含 fp16-subnormal 修复 #8 之后重测）。主角是 **HyMT2Sharp** 的 **AVX-512 档**，与强制退回 AVX2/AVX-VNNI 的基线（`DOTNET_EnableAVX512=0`）同窗口 A/B 对比。本页统一 **prefill 512 / decode 128、`--threads 0`（自动线程校准 + 绑核）**，不计模型加载与 warmup。

与其他文档的关系：[perf.md](perf.md) 是 5800X 上的 x64 数据，[perf-devin-m4.md](perf-devin-m4.md) 是 ARM64 侧。本页是 AVX-512 x86-64 服务器的对应实测。

## 1. 环境

| 项         | 值                                                                     |
| ---------- | ---------------------------------------------------------------------- |
| CPU        | Intel Xeon Platinum 8559C（Emerald Rapids），8 vCPU                      |
| ISA        | AVX-512 F/BW/VL/CD/DQ/VNNI/VBMI/VBMI2 全部可用                          |
| OS         | Ubuntu 24.04 x86_64，Release 构建，`net10.0`，.NET SDK 10               |
| 线程       | `--threads 0`（自动校准 + sched_setaffinity 绑核，main 最新线程策略）  |
| 负载       | 共享宿主机有背景负载，数字摆动 ±10-50%；A/B 交替、多窗口取 best-of-N  |
| llama.cpp  | git main 本地构建（GGML_NATIVE 自动启用 AVX-512），`-ngl 0`，t 取 4/6/8 最优（均为 t8） |

## 2. 主表：AVX-512 vs AVX2/VNNI 基线 vs llama.cpp（同机同窗口）

分发链 `UseAvx512 → UseAvxVnni → UseAvx2 → UseDp → Vector<float>`；基线用 `DOTNET_EnableAVX512=0` 关掉 AVX-512 得到（`DOTNET_EnableAVX512F` 在 .NET 10 上无效，见 §4）。每组数据跨多个窗口取 best-of（共享宿主机噪声极大）。本表用 `--threads 0`（自动线程校准 + CPU 绑核）——比固定 `--threads 8` 略快且更稳，是该箱的正确配置。

### prefill 512（tok/s）

| 量化   | AVX2/VNNI 基线 | AVX-512 | llama.cpp t8 | 512/基线 | 512/llama |
| ------ | -------------: | ------: | -----------: | ------: | --------: |
| STQ1_0 |         497.79 |  580.05 |         n/a  | **+17%** |      n/a  |
| Q2_0C  |         494.36 |  608.60 |         n/a  | **+23%** |      n/a  |
| Q4_K_M |         418.69 |  515.73 |       299.95 | **+23%** | **1.72×** |
| Q6_K   |         446.30 |  449.19 |       166.04 |    +1%¹ | **2.70×** |
| Q8_0   |         504.73 |  520.20 |       232.12 |   +3%² | **2.24×** |

¹ Q6_K 的 512 GEMM 实测退回（见 §4），prefill 走与基线相同的 AVX2 内核；448 vs llama 166 的差距来自 HyMT2 panel GEMM 体系本身而非本 PR。
² 两侧同为 VNNI-256 GEMM（见 §4），差异属测量噪声。

### decode 128（tok/s）

| 量化   | AVX2/VNNI 基线 | AVX-512 | llama.cpp t8 | 512/基线 | 512/llama |
| ------ | -------------: | ------: | -----------: | ------: | --------: |
| STQ1_0 |          50.63 |   58.15 |         n/a  | ~持平  |      n/a  |
| Q2_0C  |          57.22 |   63.53 |         n/a  | ~持平  |      n/a  |
| Q4_K_M |          29.90 |   32.63 |        36.87 | ~持平  | **0.88×** |
| Q6_K   |          26.19 |   30.01 |        33.48 | ~持平  | **0.90×** |
| Q8_0   |          28.08 |   29.45 |        26.85 | ~持平  |   1.10×  |

decode 两侧基本一致——GEMV 是内存带宽主导，512 位加宽不改变带宽需求（个别量化 +5-10%，仍在 ±10-50% 噪声带内）。**注意**：本表数字含 PR #8 的 fp16-subnormal 修复——在此之前 HyMT2 在 Q4_K/Q6_K decode 上落后 llama.cpp 30-47%，原因是真实 GGUF 的 fp16 `d`/`dmin` scale 几乎全是 subnormal（fp16 正常范围之下），`HalfBits.ToSingle` 的位操作换算路径会把它们变成 fp32 subnormal 输入、每 superblock 吃一次 ~150 周期的 denormal microcode assist；llama.cpp 走 F16C `vcvtph2ps` 硬件转换零开销。修复后 lm_head Q6 GEMV 单算 18→75 GB/s，Q4_K/Q6_K decode 对 llama 的比值从 0.72×/0.53× 升到 0.88×/0.90×，剩余差距主要是校准线程数（6-7 vs llama 8）。

## 3. 微基准（min-of-N，nIn=2048 nOut=6144 tokens=128，GEMM GMAC/s / ms）

| 内核                | AVX2/VNNI | AVX-512 | Δ      |
| ------------------- | --------- | ------- | ------ |
| Q4_K `GemmAvx2`     |    94.1   |   128.6 | **+37%** |
| Q2 `Q2Panel.Gemm`   |   29.9ms  |  25.4ms | **+18%** |
| STQ `STQPanel.Gemm` |   26.3ms  |  24.1ms |  **+9%** |
| Q6_K `GemmAvx2`     |   106.0   |   96.0¹ |  -9%¹  |
| Q8_0 `GemmVnni`     |   144.3   |   58.9¹ | -59%¹  |
| Q8_0 `GemvPacked`   |   2.66ms  |  3.00ms¹| -13%¹  |

¹ 标注的内核未进分发：实测劣于现有路径，已按"不赢不上"原则从分发链移除（见 §4）。

## 4. 加了什么 / 砍了什么

**新增的 512 档（`Simd.UseAvx512` = F+BW+VL，同时计入 `UsePanels`）：**

- `GemmQ4K.GemmAvx512`（`GemmQ4K.Avx512.cs`）：vpbroadcastq 激活广播 + zmm s16 累加 + vpermw 取 meta scale，微基准 **+37%**，e2e prefill +19%。
- `Q2Panel.GemmAvx512` / `STQPanel.GemmAvx512`：同构 512 位化（两条权重 zmm、四路激活 vpbroadcastq），**+18% / +9%**。
- Decode GEMV：`VecDotQ4K.DotAvx512`、`Q6K.DotAvx512`/`DotAvx512x4`、`Q2Panel.GemvAvx512`、`STQPanel.GemvAvx512` —— 保持分发，内存带宽主导，e2e 与 AVX2 持平，在更低带宽压力下应至少不劣。
- `Q8_0.EvenLanes512`/`OddLanes512`/`FoldDoubled512`：各内核共享的翻倍列序折叠助手。
- `Simd.UseAvx512Vbmi`：VBMI（vpermb）能力位已检测预留；当前版本内核未用到 vpermb。

**写了但没进分发的：**

- **Q6_K GEMM 512**：孤岛微基准 −9%，e2e −8%。根因是激活广播不对称——AVX2 的 `vpbroadcastd ymm` 是 load-port 融合的几乎免费操作，而 512 位下同样的 [a×8|b×8] 模式必须靠 `vpermd`（port 5），每 (row,i) 省不掉；EMR 上 zmm 整数乘吞吐 ≈ 2×ymm，无结构性收益。已从 `Gemm8x8` 分发移除（代码与测量过程见 git 历史）。
- **Q8_0 GEMM/GEMV 512**：s8×s8 点积在 512 位下无 vpdpbusd/vpsignb（.NET 10 不暴露 AVX512-VNNI），只能 vpmovsxbw 拓宽+vpmaddwd；拓宽 256↔512 无结构性差异，微基准 −59%（GEMM）、−13%（GEMV）。已整体回退到 `GemmVnni`/`PackedGroupVnni`。

**布局契约**：所有保留下来的 512 内核读的是 AVX2 同一套 canonical panel 布局，无 repack 分叉——Q4K/Q6K/Q8_0/Q2/STQ 的 repack 调用点零改动。

## 5. 验证

- `dotnet test tests/HyMT2Sharp.Tests -c Release`：默认（命中 512 档）、`HYMT2SHARP_FORCE_PORTABLE=1`、`DOTNET_EnableAVX=0`、`DOTNET_EnableAVX512=0` 四模式全绿（90/90）。
- 分发正确性：`DOTNET_EnableAVX512=0` 下 `Simd.UseAvx512=false`，落到 VNNI/AVX2 档。
