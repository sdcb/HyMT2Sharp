# Vulkan 后端性能报告 — Intel Arc B580

测试机:Windows Server 2022,Ryzen 9 5950X(8 线程),Intel Arc B580 12GB(ReBAR 开启,驱动 32.0.101.8331)。Vulkan 后端为纯 C# 实现(Backends/Vulkan,PR #20)。测试日期:2026-09-25。

## 端到端吞吐(pp512 / tg128,tok/s)

| 模型 | HyMT2Sharp Vulkan | llama.cpp Vulkan | HyMT2Sharp CPU(8T) |
|---|---:|---:|---:|
| Hy-MT2-1.8B-1.25Bit | —(不支持) | —(加载失败) | 616.0 / 62.7 |
| Hy-MT2-1.8B-2Bit | —(不支持) | —(加载失败) | 610.4 / 59.2 |
| Hy-MT2-1.8B-Q4_K_M | **10,620 / 208.4** | 7,345 / 194.0 | 402.7 / 34.8 |
| Hy-MT2-1.8B-Q6_K | **10,787 / 170.8** | 7,074 / 159.8 | 391.0 / 27.5 |
| Hy-MT2-1.8B-Q8_0 | **10,705 / 115.9** | 7,352 / 143.1 | 291.5 / 20.2 |

支持矩阵说明:

- **HyMT2Sharp Vulkan** 权重支持 Q4_K/Q6_K/Q8_0(+F32),其余类型(1.25Bit/2Bit 等自定义 STQ/Q2)`LoadModel` 显式抛 `NotSupportedException`(由 BackendFactory 自动回落 CPU,不会崩)。
- **1.25Bit / 2Bit** 是 HyMT2 自定义量化(STQ/Q2 系),连 llama.cpp 都无法加载 —— 只有 HyMT2Sharp CPU 能跑,Vulkan 列无法比较。

可比两行的倍率:

| 模型 | prefill vs llama.cpp | decode vs llama.cpp | prefill vs CPU | decode vs CPU |
|---|---:|---:|---:|---:|
| Q4_K_M | **1.45×** | **1.07×** | 26.4× | 6.0× |
| Q6_K | **1.52×** | **1.07×** | 27.6× | 6.2× |
| Q8_0 | **1.46×** | 0.81× | 36.7× | 5.7× |

Q8_0 decode 未反超(115.9 vs 143.1):两边同为权重带宽瓶颈,llama.cpp 的 Q8 GEMV 更接近上限(int8 点积路径),我们的 float-dot kernel 与自家 Q4/Q6 同处 ~77% 上限。prefill 不受影响——coopmat GEMM 吃的是预 dequant 的 fp16 权重。

decode 为带宽瓶颈:Q4_K_M 有效权重流带宽 268–278 GB/s,接近探测到的读带宽上限(~286 GB/s),即 ~95% 利用率;Q6_K decode 较低是权重更大(1.37GB vs 1.05GB)的带宽缩放,与 llama.cpp 的降幅一致。

## 关键设计点(摘要)

- **Decode**:整图单条预录 command buffer,descriptor 建一次永不动;pos/kvLen 等参数经 HOST_VISIBLE SSBO 注入;GEMV 直接读 Q4_K/Q6_K 量化块(边 dequant 边算)
- **Prefill GEMM**:VK_KHR_cooperative_matrix(8×16×16 fp16→fp32,tile 512×64,16×sg16),加载期把量化权重 dequant 成 fp16 副本供 GEMM 用(设备上量化块+fp16 并存 ~4.5GB)
- **Prefill attention**:4-kernel coopmat flash-attention(qprep→qk→soft→pv),替代标量 attention 后 attention 从 ~52ms 降到 ~7ms
- **KV**:设备端 fp16(CPU 端 bf16,UploadKv 时 host 转换)
- **回落**:无 coopmat / subgroupSizeControl 不满足时自动回落经典标量 GEMM;无真 GPU 时 BackendFactory 自动回落 CPU

## 正确性

- `--verify-decode`:GPU 与 CPU 参考逐 token argmax 对拍,seq=512 与 seq=64 均 8/8 全对(Q4_K_M);Q8_0 16/16 全对
- `dotnet test` 105/105
- Server 实测:自动选中 vulkan,流式/非流式翻译正确(194–207 tok/s)

## 理论空间与遗留

- Decode 已贴近带宽墙(剩余 ~5%)
- Prefill 距 B580 XMX 峰值(~117 TFLOPS,对应 ~20-24k t/s 现实上限)仍有 ~2× 空间;路径主要是 dequant 融合 GEMM(省 fp16 副本的 3.4GB VRAM)与 tile 形状继续调优
- 量化覆盖缺口:Q4_0 / STQ(1.25Bit)/ Q2 系尚无 GPU kernel,dequant+GEMV kernel 需逐类型补(每个类型一套 dequant+gemv,工程量按类型线性);Q8_0 已支持
- 本报告仅为 B580 实测;其他 GPU(如 RTX 3080Ti,Ada 架构)能跑但 tile/subgroup 参数非最优,需各机实测调参。macOS 走 Metal 后端。
