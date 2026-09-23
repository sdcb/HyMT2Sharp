# 开发机B 性能复测：Q4_K_M / Q6_K / Q8_0 vs llama.cpp

测量日期：2026-09-16（UTC+8）。主角是 **HyMT2Sharp**（纯 C# CPU 推理），llama.cpp 纯 CPU 构建作为同窗口对照。本页统一 **prefill 512 / decode 128**，不计模型加载与 warmup。

与 [engine-benchmark.md](engine-benchmark.md) 的关系：那一页是 5800X 上的跨引擎大表（含 Q1.25 / Q2 / GPU 对照）；本页只记录开发机B 上 Q6/Q8 落地的验收数据。

## 1. 环境

| 项         | 值                                                                                       |
| ---------- | ---------------------------------------------------------------------------------------- |
| 机器       | 开发机B：hybrid CPU，8 个性能核 + 16 个能效核（24 物理核 / 32 逻辑核），双通道 DDR4-3200 |
| OS         | Windows，Release 构建，`net10.0`                                                         |
| 线程       | 8（worker 绑在 8 个独立 P-core 上）                                                      |
| SIMD       | AVX2=True，AVX-VNNI=True（`DOTNET_EnableAVXVNNI=0` 可关）                                |
| HyMT2Sharp | 基于 `60c2da3` 的工作区 + Q6/Q8 支持改动                                                 |
| llama.cpp  | build **10875**，commit `14a9d09f7`，纯 CPU（`-ngl 0`）                                  |

| 量化   | 文件                      |     大小 |
| ------ | ------------------------- | -------: |
| Q4_K_M | `Hy-MT2-1.8B-Q4_K_M.gguf` | 1.05 GiB |
| Q6_K   | `Hy-MT2-1.8B-Q6_K.gguf`   | 1.37 GiB |
| Q8_0   | `Hy-MT2-1.8B-Q8_0.gguf`   | 1.77 GiB |

## 2. 主表

| 模型              | SIMD     |      prefill 512 |      decode 128 | prefill 三次          |
| ----------------- | -------- | ---------------: | --------------: | --------------------- |
| HyMT2Sharp Q4_K_M | AVX2     | **462.00 tok/s** | **24.25 tok/s** | 450.2 / 466.8 / 469.5 |
| HyMT2Sharp Q6_K   | AVX2     | **463.70 tok/s** |     20.63 tok/s | 437.8 / 453.8 / 504.5 |
| HyMT2Sharp Q8_0   | AVX-VNNI | **548.82 tok/s** |     15.96 tok/s | 596.8 / 573.6 / 488.5 |
| llama.cpp Q4_K_M  | —        |     291.55 tok/s |     26.75 tok/s | `llama-bench` 单轮    |
| llama.cpp Q6_K    | —        |     119.08 tok/s |     21.36 tok/s | 同窗口                |
| llama.cpp Q8_0    | —        |      93.00 tok/s |     17.94 tok/s | 同窗口                |

SIMD 说明：Q4 没有 VNNI 内核；Q6 固定走 AVX2 `vpmaddubsw`（实测 `vpdpbusd` 在 Raptor Lake 同构微架构上反而更慢）；Q8 有 AVX-VNNI 时走 `vpdpbusd`。

## 3. 结论

- **Prefill 全面碾压**：HyMT2Sharp 是 llama.cpp 的约 4–5.9 倍（Q6 463.7 vs 119.1，Q8 548.8 vs 93.0）。
- **Decode 同量级、略慢**：Q6 20.6 vs 21.4（−3.5%），Q8 16.0 vs 17.9（−11%）。
- Decode 全部贴着内存带宽墙：权重大小 Q4 1.05 GiB → Q6 1.37 GiB → Q8 1.77 GiB，吞吐几乎严格反比于权重大小。

## 4. 带宽与微基准

开发机B 的实测带宽上限（8 线程只读扫描，与模型同大小的连续 buffer）：

| 测量                                               | 数值                               |
| -------------------------------------------------- | ---------------------------------- |
| 纯只读带宽                                         | ~26–37 GB/s（随后台负载波动）      |
| Q8 packed GEMV 微基准（L3 内 6144×2048）           | ~91 GB/s                           |
| Q8 packed GEMV 微基准（lm_head 2048×120816，DRAM） | ~38.7 GB/s                         |
| Q8 模拟 decode GEMV 链（micro `--micro-q8`）       | ~28–37 GB/s，约 14–17.1 tok/s 等价 |

GEMV 微基准已能达到与 llama.cpp decode 相同的有效带宽，说明纯权重流并不慢；真实 decode 的剩余差距来自逐 op 调度间隙与串行小段（RMSNorm / 激活量化 / SiLU / argmax / softmax）。

## 5. 噪声警告

这台开发机有常驻后台负载（IDE、IM、企业服务等），内存带宽在 ~26–37 GB/s 之间跳变，同一二进制相邻两次 decode 可差 20%。上表数字是多次交替采样中的代表性值；对比结论（prefill 倍数、decode 差距量级）在各窗口下稳定，绝对值会漂。

## 6. 复现

```powershell
dotnet run --project src/HyMT2Sharp.Benchmark -c Release -- `
  --model "Hy-MT2-1.8B-Q8_0.gguf" --bench --bench-prefill 512 --bench-decode 128 --threads 8

llama-bench.exe -m "Hy-MT2-1.8B-Q8_0.gguf" -p 512 -n 128 -t 8 -ngl 0
```

`--micro-q8` 输出 packed GEMV 与只读带宽微基准；`--profile` 输出逐 op 分项耗时。
