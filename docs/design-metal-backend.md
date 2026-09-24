# HyMT2Sharp Metal GPU 后端设计(纯 net10.0 + libobjc P/Invoke)

目标:在 Apple Silicon 上以 **零原生代码、零 cmake、零第三方依赖** 提供 GPU 推理档,性能贴近平台极限(参考口径:MLX 快路径实测 decode ~173 tok/s ≈ 175 GB/s 带宽墙,prefill ~1900 tok/s)。**注意:VM(paravirt GPU)缺 simdgroup reduce/bfloat,我们的 portable 档在 VM 上的真实上限未经实测**,§0 数字均属快路径/MLX 侧参考,M1 spike 第一要务就是测出 portable 档的 VM 实墙。CPU 路径(SDOT/AVX2/portable)保持不动,Metal 只是新的可选后端。

## 0. 已验证前提(本 VM 实测)

| 事实 | 数据 | 来源 |
|---|---|---|
| C#(net10.0-macos workload)→ Metal 绑定直通 | saxpy 151–154 GB/s = Swift 原生 153 GB/s | metalspike |
| runtime 编译 MSL → PSO → dispatch | 全托管可行,结果正确 | metalspike |
| GPU family | Apple5=True, Mac2=True → **ICB 可用** | metalspike |
| 450 个独立 encoder/token | CPU encode ~2.8–4.4ms/token | metalspike |
| MLX lazy 图 + 单 eval | decode 173 tok/s、prefill ~1900 | mlxspike |
| 带宽墙 | ~1.05 GB 权重/token × 175 GB/s ≈ 175 tok/s | 推算 |
| paravirt GPU 缺特性 | simdgroup reduce=false、bfloat=false | ioreg/ggml 崩溃日志 |
| **portable 档 VM 性能** | **未实测——M1 spike 第一个测量点** | 本设计最大未知量 |

## 1. 分层结构与依赖方向

**单程序集方案**:`Sdcb.HyMT2Sharp.Kernels`、`Sdcb.HyMT2Sharp.Gguf`、`Sdcb.HyMT2Sharp.Model` 合并为单一项目 **`Sdcb.HyMT2Sharp`**(assembly/dll 名与 NuGet 包名一致),Metal 后端同驻此 dll;源码保留文件夹逻辑层次。决策说明:NuGet 多 dll 其实同体验、惰性 DllImport 也已使非 Mac 零成本——选择合并的真实动机是**单文件分发简洁 + 零接线成本**,代价是一次 csproj/InternalsVisibleTo 调整(本仓库项目边界本就是为了分层而非独立部署)。入口项目引用关系零变化:

```
Cli / Server / Benchmark ──→ Sdcb.HyMT2Sharp(单 dll)
                                  ├── Gguf/            (原 Gguf 项目,逻辑层次不变)
                                  ├── Kernels/         (原 Kernels 项目,CPU SIMD 实现不变)
                                  ├── Backends/
                                  │    ├── IComputeBackend.cs / IDeviceTensor.cs   (~250 行)
                                  │    ├── Cpu/CpuBackend.cs   薄壳转发现有 Ops.*  (~200 行)
                                  │    └── Metal/
                                  │         ├── ObjC.cs          libobjc P/Invoke  (~400 行)
                                  │         ├── MetalWrappers.cs MTL* 薄封装       (~500 行)
                                  │         ├── MslKernels.cs    内嵌 MSL+PSO缓存  (~1500–2500 行 MSL)
                                  │         └── MetalBackend.cs  实现+权重+调度    (~800 行)
                                  └── HunyuanDenseModel(backend)  forward 经接口发 op
```

- **跨平台零成本**:Metal 代码是纯 net10.0 C#,全平台编译;`DllImport("libobjc.dylib")` 声明本身不解析,只有被调用时才加载——Windows/Linux 上 `Backends.Metal` 命名空间只是死代码。入口加 `--backend metal` 分支时以 `RuntimeInformation.IsOSPlatform(OSX)` + `MTLCreateSystemDefaultDevice()` 探测门控,探测失败给出"无 Metal 设备"错误,绝不影响 CPU 路径。
- **接口仍解耦**:`IComputeBackend` 定义在 `Backends/` 下,`MetalBackend` 与 `CpuBackend` 并列实现——与"接口独立、实现插件化"原则一致,只是插件边界从"程序集"降为"文件夹/命名空间",换来用户零包管理成本。
- **backend 选择**:`--backend cpu|metal` 或 `HunyuanDenseModel.Load(path, Backend.Create("metal"))`;Metal 探测失败显式报错而非静默回退。**量化范围:MVP 只支持 Q4_K_M**;`--backend metal` 加载 Q6_K/Q8_0/STQ/Q2 等其它量化时整体拒绝并提示改用 `--backend cpu`(不做按算子混合回退)。
- **条件编译可选**:若担心无 Mac 的开发者碰到编译告警,可用 `#if` + `SupportedOSPlatform` 属性标注;不建议拆出独立 TFM(那会回到 -macos TFM 的老问题)。

主仓库不带任何 macOS workload;NuGet 包形态不变(单 dll),用户 Mac 上 `dotnet add package` 即获得 GPU 档。

## 2. libobjc 互操作层(ObjC.cs)

只声明用到的 ~30 个 entry point,全部 `[LibraryImport("libobjc.dylib")]`:

- 类/selector:`objc_getClass`、`sel_registerName`
- `objc_msgSend` **按 arity 声明固定签名重载**:`msgSend(id,sel)`、`(id,sel,IntPtr)`、`(id,sel,nuint)`、`(id,sel,MTLSize,MTLSize)`、`(id,sel,IntPtr,nuint,nuint,IntPtr)` 等约 10 个变体;返回 `float`/`nuint`/`bool` 各一份
- 大 struct 返回值用 `objc_msgSend_stret`(compute 路径基本用不到,预留)
- `dispatch_get_global_queue`/`dispatch_async_f` 用于 MTLCommandBuffer 的异步回调可选
- `NSAutoreleasePool`:每次 encode 批外层包 `alloc/init` … `release`,防 autoreleased 对象累积

**对象所有权硬规则(Cocoa 内存约定,写错即 UAF/泄漏)**:
- `alloc`/`init`/`newXxx`/`copy`/`mutableCopy` 家族(`newCommandQueue`、`newBufferWithBytes`、`newLibraryWithSource`、`newComputePipelineState`、`MTLCreateSystemDefaultDevice` 等)→ **返回已持有对象(+1)**,封装类 `Dispose` 时必须 `release`
- 其余方法(`commandBuffer`、`computeCommandEncoder`、属性式访问器)→ **autoreleased**,只能在当前 pool 生命周期内使用;需跨 pool 持有的先 `retain`、`Dispose` 时 `release`
- 诊断:`newLibraryWithSource` 的 `NSError` 立即接 `localizedDescription` 转成托管异常——MSL 编译错误不接就是黑盒;`BOOL` 返回值注意 arm64 为 1 字节 marshal(`[MarshalAs(UnmanagedType.Bool)]` 或按 byte 收)

ABI 依据:arm64 上非变参声明的参数按 AAPCS64 常规规则传递,`objc_msgSend` 本身只是把参数透传给 IMP——SharpMetal/Metal.NET 已证明此模型在 Apple Silicon 上正确。结构体按值传参(MTLSize=24B 走栈)由 P/Invoke blittable struct 正常封送。

参考实现:`qian-o/Metal.NET`(Clang AST 生成 + 手写 objc_msgSend 封装)。`ObjectiveC.cs` 可借鉴其 NativeObject/Selector 设计;也可直接引其 NuGet 当绑定层(依赖评估后定)。

## 3. Metal 资源管理

- **权重**:加载时把 GGUF 量化块**重打包一次**进连续 MTLBuffer(`StorageModeShared`,统一内存零拷贝语义)。理由:ggml tensor 在文件内不保证 256B buffer-offset 对齐,Q4_K 的 `block_q4_K{qs,scales,d,dmin}` 布局也要按 kernel 最优遍历序重排——和 CPU panel 同一哲学,一次性成本(~1GB 拷贝,~秒级)。LM head 单独一块。
- **激活/KV**:所有中间 tensor 常驻 device buffer,生命周期由后端管理;`x` 与 `out` 双缓��,每 token 不落 host。
- **KV cache**:存储 **bf16**。M2 先用 **flat 连续 buffer**(与 CPU `KvCacheConfig` memory 模式同构,增长时重分配),attention 线性扫描——最简路径先跑通。**block-table paged attention 后置**:与 `KvBlockStore` 的块式管理对齐时,attention kernel 改按 uint32 block table 间接遍历(即 vLLM paged-attn 形态),单独立项,不堵 M2。注意 paravirt GPU **无 bfloat ALU**:kernel 按 `ushort` 读入后 `<<16` 手工转 fp32(成本~零,portable 档唯一路径);真机 M4+ 可换 MSL `bfloat` 类型走快路径。
- **token 输出环**:argmax 结果写入共享内存环形 token buffer(8 槽),host 侧等 `MTLSharedEvent`/简单轮询读取——见 §5。

## 4. MSL kernel 集(每 kernel 两档:portable + simdgroup 快路径)

load 时 `supportsFamily:`/`supportsFeatureSet:` 探测选档;paravirt GPU(Apple5 无 simdgroup reduce)命中 portable 档,真机 M4+/M5 命中快路径——这套 tier 与 CPU 的 AVX512/AVX2/AdvSimd/portable 完全同构。

| kernel | 说明 | 参考 |
|---|---|---|
| `embed_gather` | 从 device token buffer 读 id,gather embedding 行——**去 host 化关键** | 自写,~20 行 |
| `rmsnorm` | dim=2048 单 threadgroup,portable 档用 shared-mem 树形归约 | MLX `rms_norm` |
| `qmatmul_gemv` × {Q4_K,Q6_K,Q8_0} | **核心**:块内 dequant+fma,每 threadgroup 吃若干输出行,目标 ≥150 GB/s 有效带宽 | TensorSharp `MlxNative.cs` 内嵌 Q4_K MSL + MLX `qmv_fast` |
| `qmatmul_gemm` × 同上 | prefill 用,n≤512 批;portable 档先 naive-tiled,快路径 simdgroup 矩阵 | MLX `qmm`/`steel` |
| `rope_neox_qk` | fuse 进 QKV gemv 收尾,或独立小 kernel(128-dim,head 循环) | 自写 |
| `attention_decode` | GQA 16q/4kv,flash-decode:每 q-head 一 threadgroup 线性扫 flat KV(bf16→fp32 移位转换),online softmax;**paged block-table 变体后置** | MLX fast.sdpa + llama.cpp FA |
| `attention_prefill` | 512² causal tiled flash | 同上 |
| `kv_append` | K/V 行写入 cache[pos](fp32→bf16 截断存储) | 自写,~20 行 |
| `swiglu` | silu(gate)·up,可 fuse 进 up-proj gemv epilogue | 自写 |
| `residual_add` | 逐元素,可 fuse 进下一 rmsnorm/o-proj | 自写 |
| `argmax_120k` | 两阶段:block 局部 argmax → 终归约,写 token buffer | 自写 |

`embed_gather` + device 端 token 反馈是 decode 流水线的关键:下一 token 的 embedding 完全不经过 host,采样结果只被 host **读取**(用于 EOS/日志/采样参数),不再参与驱动 GPU。

## 5. Decode 调度:encode/execute 解耦(贴极限核心)

朴素方案(每 token encode 450 encoder)实测 CPU 侧 ~3ms/token,串行后必丢带宽。三级设计,逐级升级:

**Level 0 — 融合 kernel + 单 command buffer**:kernel 按 §4 融合后,每 token ~165 dispatch(QKV+rope×1、attn×1、O+res×1、gate/up+swiglu×1、down+res×1、=5/layer + 头尾),encode ~1.2ms/token。`MTLHazardTrackingModeTracked` 让 Metal 自动建立 dispatch 间依赖。

**Level 1 — 流水线化(主力方案)**:token N+1 的 command buffer **在 token N 仍在 GPU 执行时就已 encode 进队列**——embed_gather 已去 host 化,整条 token 序列可以提前 enqueue;host 只通过共享 token 环读结果做 EOS/采样检查。CPU encode(1.2ms) < GPU exec(~5.7ms) ⇒ GPU 永不饥饿,吞吐 = GPU 极限 ≈ 165–175 tok/s。

**Level 2 — ICB 选项**:`MTLIndirectCommandBuffer`(Mac2 family 已验证支持)把整个 token 图一次编码,per token 仅 patch seq-pos/KV-offset 标量后 `executeCommandsInBuffer`,CPU 成本趋零;双 ICB 轮换解决 x/out 双缓冲地址切换。作为后续优化预留,Level 1 先落地。

**同步语义**:token buffer 完成后 host 读共享内存;停止条件满足时停止 enqueue 并让队列排空。**采样复用 `Sampler`/`SamplingParams`**:非 greedy(temperature/top-k/top-p/repeat penalty)时,logits 行(120818×4B≈0.5MB)经共享内存读回、喂给现有 `Sampler`(~0.1ms/token),host 把采样结果写回 token 环供下轮 embed_gather。**注意:非 greedy 重新引入每-token host 依赖**(等观测→采样→写回,N+1 才有输入)——流水线 enqueue 深度须 ≥2(greedy 可无限深,host 永不阻塞),这是 Level 1 唯一真实时序风险,实测确认。CPU 侧采样逻辑一处维护,两后端行为天然一致。

## 6. Prefill 调度

512×2048×2048 的 fp16 GEMM 序列,单次 forward 一个大 command buffer 全量 enqueue;激活量化成 q8 面板(复用 `QuantizeQ8Kx4` 思路)或直接 f16 GEMM——portable 档先保证正确,快路径(simdgroup 矩阵,真机有)再追 ~1900 tok/s。chunked prefill 与现有 CPU 实现对齐(≥ctx 时分块)。

## 7. 正确性验证(硬门槛)

1. **算子对拍**:固定输入 dump,CPU backend vs Metal backend 每算子 L2/max-err(rmsnorm/rope/attention/qmatmul 各档)
2. **logit 对拍**:同 prompt 前向,logits cosine ≥ 0.9999、argmax 一致
3. **端到端**:greedy decode 前 N token 与 CPU 输出逐 token 相同
4. **回归**:`dotnet test` 现有套件不受影响(CPU backend 零行为变化);新增 `MetalOpsTests`(非 macOS 跳过)

## 8. 发布形态

- 单程序集 `Sdcb.HyMT2Sharp.Model`(内含 Metal 后端),NuGet 包形态不变;`--backend cpu|metal` 或 `Backend.Create("metal")` 选择
- 无 dylib、无 metallib 资源文件、无签名要求——用户 Mac 上 `dotnet add package` 即用
- 构建机无需 macOS/workload(全部源码编译期无关平台);入口项目引用关系零变化(依旧只引 Model)

## 9. 风险与工程量

| 项 | 评估 |
|---|---|
| Q4_K gemv 到 ~150 GB/s | **主要技术风险**,有 TensorSharp/MLX 成熟实现可抄,kernel 迭代 1–2 轮可达 |
| encode 流水线(Level 1) | 设计直白,Metal queue 语义支持;唯一要验证的是 hazard tracking 与共享 token 环的顺序性 |
| paravirt 特性缺失 | portable 档已覆盖,但**其在 VM 的实际上限未实测**——M1 spike 第一项就是测 portable qmatmul 带宽;若 portable 档差距大,VM 数字按 portable 口径报告,175 tok/s 仅作真机上限参考 |
| objc ABI 兼容 | libobjc = macOS 最稳 ABI(2007 至今),feature-detect 覆盖芯片差异 |
| 工程量 | binding ~400 行 + MSL ~1500–2500 行 + 集成 ~800 行 ≈ **decode MVP 2–3 天,prefill+对拍再 1–2 天**(按我的吞吐) |

## 10. 里程碑

- **M1(spike)**:ObjC.cs + qmatmul_gemv(Q4_K,portable+simdgroup 两档)+ 带宽微基准 → **先测出 portable 档在 paravirt VM 的实墙**,再对照 simdgroup 档;门槛:达到有效带宽 ≥140 GB/s 或给出 VM portable 上限数据
- **M2**:全 decode 图(embedding→argmax→token 环)Level 0/1,flat KV → 目标按 M1 实测口径定(参考 ≥150 tok/s)
- **M3**:prefill GEMM + 对拍套件 → 目标 prefill ≥1000 tok/s(VM 上)
- **M4**:block-table paged attention + `KvBlockStore` 对齐(bf16 块管理)
- **M5(可选)**:ICB 调度、Q6_K/Q8_0 kernel、STQ/Q2 自定义量化
