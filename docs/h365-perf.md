# HyMT2Sharp 性能：Ryzen AI 9 H 365 + Radeon 880M 实测

测量日期：2026-09-25（UTC+8）。本页统一 **prefill 512 / decode 128**，不计模型加载与 warmup。与 [perf.md](perf.md)（5800X + RTX 3080 Ti 台式机）为跨机对照关系：同样的 HyMT2Sharp commit、同样的模型文件、同样的测量口径。

## 1. 环境

| 项         | 值                                                                       |
| ---------- | ------------------------------------------------------------------------ |
| CPU        | AMD Ryzen AI 9 H 365（Strix Point，4× Zen5 + 6× Zen5c，10 核 / 20 线程） |
| GPU        | AMD Radeon 880M 核显（RDNA 3.5，共享系统内存）                            |
| 内存       | 32 GB LPDDR5X-7500（4×8 GB 板载）                                        |
| OS         | Windows，Release 构建，`net10.0`                                         |
| 线程       | 8（为与 5800X 口径一致；CPU 实测 avx2=True，vnni=True）                   |
| HyMT2Sharp | git `d74799b`（2026-09-25）                                              |
| 模型路径   | `D:\_\hy-mt2\Hy-MT2-1.8B-*.gguf`（5 个量化，同 perf.md）                  |

## 2. 主表：HyMT2Sharp CPU @ H 365

| 量化              |      prefill 512 |      decode 128 | prefill 三次          |
| ----------------- | ---------------: | --------------: | --------------------- |
| Q1.25 / STQ1_0    |     529.01 tok/s |     63.92 tok/s | 515.9 / 538.5 / 533.2 |
| Q2_0C             |     502.76 tok/s |     71.56 tok/s | 517.2 / 510.4 / 482.1 |
| Q4_K_M            |     454.24 tok/s |     41.68 tok/s | 484.0 / 443.1 / 438.3 |
| Q6_K              |     396.28 tok/s |     38.93 tok/s | 387.8 / 388.1 / 414.1 |
| Q8_0              |     467.92 tok/s |     32.32 tok/s | 465.7 / 467.1 / 470.9 |

## 3. 跨机对比：H 365 vs 5800X（perf.md 第 2 节）

| 量化              | H365 prefill | 5800X prefill | 比值  | H365 decode | 5800X decode | 比值    |
| ----------------- | -----------: | ------------: | ----: | ----------: | -----------: | ------- |
| Q1.25 / STQ1_0    |       529.01 |        564.66 |  −6%  |       63.92 |        47.75 | **+34%** |
| Q2_0C             |       502.76 |        488.95 |  +3%  |       71.56 |        47.53 | **+51%** |
| Q4_K_M            |       454.24 |        423.99 |  +7%  |       41.68 |        26.55 | **+57%** |
| Q6_K              |       396.28 |        403.25 |  −2%  |       38.93 |        22.96 | **+70%** |
| Q8_0              |       467.92 |        319.81 | **+46%** |    32.32 |        16.60 | **+95%** |

- **Prefill** 基本持平：Q4/Q6 都走 AVX2 `vpmaddubsw`，H 365 的 4 个 Zen5 满核对 5800X 的 8 核，单核 IPC/频率优势被核心数抵消。Q8 是例外（+46%）：5800X 无 AVX-VNNI 退回 `vpmaddubsw`，Zen5 有 VNNI 走 `vpdpbusd`——与 perf.md 第 4 节「VNNI 缺失的影响集中在 Q8」的预测一致。
- **Decode** 全面快 34–95%：decode 贴内存带宽墙，LPDDR5X-7500 双通道（理论 120 GB/s）对台式机 DDR4 优势明显。按 decode × 权重折算有效带宽：

  | 量化   | decode   | 有效带宽   |
  | ------ | -------: | ---------: |
  | Q1.25  |    63.92 | ~29.5 GB/s |
  | Q2_0C  |    71.56 | ~43.0 GB/s |
  | Q4_K_M |    41.68 | ~47.2 GB/s |
  | Q6_K   |    38.93 | ~57.4 GB/s |
  | Q8_0   |    32.32 | ~61.7 GB/s |

  5800X 同口径为 ~21–31 GB/s，本机约为其 1.4–2.0 倍，与 decode 增速匹配。
- Q2_0C decode（71.56）反超 Q1.25（63.92）：两者块结构差异小，5800X 上基本持平，此排序按 perf.md 第 6 节的噪声范围处理，不做过度解读。

## 4. Vulkan 后端 @ Radeon 880M

`--backend vulkan` 直接可用（`--backend` 省略时 auto 探测也会选中 Vulkan）。RDNA 3.5 subgroup=32，prefill 走 sg32 cooperative-matrix tensor-core 管线（同 NVIDIA 路径）。权重支持 Q4_K/Q6_K/Q8_0；STQ1_0/Q2_0C 无 GPU kernel，`LoadModel` 回落 CPU。

| 量化   | Vulkan pp512 / tg128 | 同机 CPU（第 2 节） | 加速比         |
| ------ | -------------------: | ------------------: | -------------- |
| Q4_K_M |    **1103.09 / 60.54** |    454.24 / 41.68   | ~2.4× / ~1.5×  |
| Q6_K   |    **1061.06 / 44.30** |    396.28 / 38.93   | ~2.7× / ~1.1×  |
| Q8_0   |     **891.02 / 40.68** |    467.92 / 32.32   | ~1.9× / ~1.3×  |

- 正确性：`--verify-decode 8`（Q4_K_M）top1 与 CPU 8/8 一致，prefill max-abs ~1e0（fp16 累加正常量级）。
- 量级参考：同代码在 RTX 3080 Ti 上 pp512 ~15.8k tok/s（perf.md 第 7 节），880M 约为 7%——核显共享内存带宽与 CU 规模所限，符合预期。Decode 侧 880M（~61 tok/s @Q4_K_M）已超 5800X 的 CPU 成绩，但相对 3080 Ti（~277）差 4.6 倍，符合显存带宽差距。
- iGPU 与 CPU 共享内存控制器：Vulkan decode 与 CPU decode 的量级接近是带宽墙在两端的同一体现。

## 5. 复现

```powershell
# CPU
src/HyMT2Sharp.Benchmark/bin/Release/net10.0/Sdcb.HyMT2Sharp.Benchmark.exe `
  --model "D:\_\hy-mt2\Hy-MT2-1.8B-Q4_K_M.gguf" --backend cpu --bench-prefill 512 --bench-decode 128 --threads 8

# Vulkan（880M）
src/HyMT2Sharp.Benchmark/bin/Release/net10.0/Sdcb.HyMT2Sharp.Benchmark.exe `
  --model "D:\_\hy-mt2\Hy-MT2-1.8B-Q4_K_M.gguf" --backend vulkan --bench-prefill 512 --bench-decode 128 --threads 8

# GPU 正确性
... --backend vulkan --verify-decode 8
```

其余量化换 `--model` 路径即可。注意 `--backend` 缺省为 auto：有可用 Vulkan 设备时会自动走 GPU，跑 CPU 对照务必显式 `--backend cpu`。
