# HyMT2Sharp [![NuGet](https://img.shields.io/nuget/v/Sdcb.HyMT2Sharp.svg)](https://www.nuget.org/packages/Sdcb.HyMT2Sharp) [![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](https://github.com/sdcb/HyMT2Sharp/blob/main/LICENSE) [![QQ](https://img.shields.io/badge/QQ_Group-495782587-52B6EF?style=social&logo=tencent-qq&logoColor=000&logoWidth=20)](https://qm.qq.com/cgi-bin/qm/qr?_wv=1027&k=&authKey=&noverify=0&group_code=495782587)

**中文** | [English](https://github.com/sdcb/HyMT2Sharp/blob/main/README_EN.md)

纯 C# 的 [Hy-MT3](https://huggingface.co/tencent/Hy-MT3-1.8B) / [Hy-MT2](https://huggingface.co/tencent/Hy-MT2-1.8B)（`hunyuan-dense`）**非官方**推理实现。CPU 自带 AVX2 / AVX-512 / NEON SIMD 内核，另有 Vulkan 与 Metal GPU 后端；不依赖 llama.cpp 或 ONNX Runtime，面向进程内调用。

当前验证过的 GGUF 量化：**Q8_0**、**Q6_K**、**Q4_K_M**、**Q2_0C**、**1.25-bit STQ1_0**。其它格式与模型规模尚未测试。

## 模型

推理包 `Sdcb.HyMT2Sharp` 不含权重。可以从 Hugging Face 下载 GGUF，也可以直接引用 NuGet 权重包：

| 模型 | 量化 | Hugging Face | NuGet 权重包 | 分片数 | 总大小 |
| --- | --- | --- | --- | --- | --- |
| Hy-MT3-1.8B | Q4_K_M | [Hy-MT3-1.8B-GGUF](https://huggingface.co/tencent/Hy-MT3-1.8B-GGUF) | [Sdcb.GgufWeights.Hy-MT3-1.8B-Q4_K_M](https://www.nuget.org/packages/Sdcb.GgufWeights.Hy-MT3-1.8B-Q4_K_M) | 6 | 约 1.3 GB |
| Hy-MT3-1.8B | Q8_0 | [Hy-MT3-1.8B-GGUF](https://huggingface.co/tencent/Hy-MT3-1.8B-GGUF) | [Sdcb.GgufWeights.Hy-MT3-1.8B-Q8_0](https://www.nuget.org/packages/Sdcb.GgufWeights.Hy-MT3-1.8B-Q8_0) | 8 | 约 1.8 GB |
| Hy-MT2-1.8B | 1.25-bit STQ1_0 | [Hy-MT2-1.8B-1.25Bit-GGUF](https://huggingface.co/tencent/Hy-MT2-1.8B-1.25Bit-GGUF) | [Sdcb.GgufWeights.Hy-MT2-1.8B-1.25Bit](https://www.nuget.org/packages/Sdcb.GgufWeights.Hy-MT2-1.8B-1.25Bit) | 2 | 约 440 MB |
| Hy-MT2-1.8B | Q2_0C | [Hy-MT2-1.8B-2Bit-GGUF](https://huggingface.co/tencent/Hy-MT2-1.8B-2Bit-GGUF) | [Sdcb.GgufWeights.Hy-MT2-1.8B-2Bit](https://www.nuget.org/packages/Sdcb.GgufWeights.Hy-MT2-1.8B-2Bit) | 3 | 约 573 MB |
| Hy-MT2-1.8B | Q4_K_M | [Hy-MT2-1.8B-GGUF](https://huggingface.co/tencent/Hy-MT2-1.8B-GGUF) | [Sdcb.GgufWeights.Hy-MT2-1.8B-Q4_K_M](https://www.nuget.org/packages/Sdcb.GgufWeights.Hy-MT2-1.8B-Q4_K_M) | 5 | 约 1.1 GB |
| Hy-MT2-1.8B | Q6_K | [Hy-MT2-1.8B-GGUF](https://huggingface.co/tencent/Hy-MT2-1.8B-GGUF) | [Sdcb.GgufWeights.Hy-MT2-1.8B-Q6_K](https://www.nuget.org/packages/Sdcb.GgufWeights.Hy-MT2-1.8B-Q6_K) | 7 | 约 1.4 GB |
| Hy-MT2-1.8B | Q8_0 | [Hy-MT2-1.8B-GGUF](https://huggingface.co/tencent/Hy-MT2-1.8B-GGUF) | [Sdcb.GgufWeights.Hy-MT2-1.8B-Q8_0](https://www.nuget.org/packages/Sdcb.GgufWeights.Hy-MT2-1.8B-Q8_0) | 8 | 约 1.8 GB |

只需引用表中的顶层包，其余分片包（`.Part2`…）会作为依赖自动还原，分片数含顶层包本身；总大小即该模型 GGUF 的大小，也是输出目录多出的体积。包很大，首次还原可能会比较慢（2Bit 约 573 MB，本机用时约 7.5 分钟）。加载方式：`new HunyuanDenseModel(Sdcb.GgufWeights.Hy_MT2_1_8B_2Bit.Weights.Model())`。

## 快速开始

从仓库直接运行 CLI。CLI 和 Server 默认引用 NuGet 包 `Sdcb.GgufWeights.Hy-MT2-1.8B-2Bit`，不传 `--model` 即加载内置的 2Bit 权重；也可以用 `--model` 指定自己的 GGUF 文件：

```powershell
dotnet run --project src/HyMT2Sharp.Cli -c Release
dotnet run --project src/HyMT2Sharp.Cli -c Release -- --model "D:\_\model\Hy-MT2-1.8B-Q4_K_M.gguf"
```

不传 `--threads` 时按 CPU 拓扑自动绑定物理 P-core（不含 SMT 和 E-core；大小核例如 8P+8E 为 8，5800X 为 8）。加 `--prompt` 跑单轮后退出；省略则进入多轮对话。

### HTTP 服务

`HyMT2Sharp.Server` 提供 OpenAI 兼容的 `POST /v1/chat/completions`（含 SSE 流式）和内置聊天页：

```powershell
dotnet run --project src/HyMT2Sharp.Server -c Release
```

同样可用 `--model`（或环境变量 `HYMT2_MODEL`）指定 GGUF 文件。

浏览器访问 `http://127.0.0.1:8080`，或用 curl：

```powershell
curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" -d "{\"messages\":[{\"role\":\"user\",\"content\":\"Translate the following segment into Chinese, without additional explanation：SimdPaddleOCR is officially released today (NuGet: Sdcb.SimdPaddleOCR). It is a complete OCR inference engine written entirely in C#. It does not depend on Paddle Inference or ONNX Runtime, and it does not require shipping OpenCV native libraries.\"}],\"max_tokens\":128}"
```

### WinForms 示例（Windows）

`HyMT2Sharp.WinForms` 是一个桌面翻译窗口：左边输入、右边流式输出，可选引擎（自动 / CPU / Vulkan）和目标语言，底部状态栏显示加载时间与 prefill / decode 速度。权重固定用 2Bit 包，换量化请改 csproj 里的权重包引用。支持高 DPI（PerMonitorV2）。

```powershell
dotnet run --project src/HyMT2Sharp.WinForms -c Release
```

## 作为库使用

安装推理包：

```powershell
dotnet add package Sdcb.HyMT2Sharp
```

`HunyuanDenseModel` 负责加载 GGUF、分词、KV cache 和 `Forward`。库内提供 `Sampler`（`Sampler.ArgMax`，以及 temperature / top-p / top-k 等采样），但没有内置 `Generate`，文本拼接留给调用方。下面是一个最小 greedy 流式示例，权重来自 `Sdcb.GgufWeights.Hy-MT2-1.8B-2Bit` 包（见[模型](#模型)）：

```csharp
using System.Runtime.CompilerServices;
using Sdcb.GgufWeights.Hy_MT2_1_8B_2Bit;
using Sdcb.HyMT2Sharp.Backends;
using Sdcb.HyMT2Sharp.Model;

// 也可以直接传 GGUF 路径：new HunyuanDenseModel(@"D:\models\Hy-MT2-1.8B-Q4_K_M.gguf")
using HunyuanDenseModel model = new(Weights.Model(), backend: BackendFactory.Create());

await foreach (string piece in Generate(model, "Translate the following segment into Chinese, without additional explanation：SimdPaddleOCR is officially released today (NuGet: Sdcb.SimdPaddleOCR). It is a complete OCR inference engine written entirely in C#. It does not depend on Paddle Inference or ONNX Runtime, and it does not require shipping OpenCV native libraries."))
    Console.Write(piece);

// 完整字符串：string text = string.Concat(await Generate(...).ToArrayAsync());

static async IAsyncEnumerable<string> Generate(
    HunyuanDenseModel model,
    string user,
    int maxTokens = 128,
    [EnumeratorCancellation] CancellationToken cancellationToken = default)
{
    int[] prompt = model.Tokenizer.Encode(ChatTemplate.RenderHunyuanDense([new ChatMessage("user", user)]));
    float[] logits = model.Forward(model.AlignPrompt(prompt).Suffix);
    await Task.Yield();

    List<int> generated = [];
    string visible = "";
    for (int i = 0; i < maxTokens; i++)
    {
        cancellationToken.ThrowIfCancellationRequested();
        int token = Sampler.ArgMax(logits);
        if (model.Tokenizer.IsStop(token))
            break;

        generated.Add(token);
        // A trailing U+FFFD is a UTF-8 sequence split across tokens; wait for the next token.
        string next = model.Tokenizer.DecodeVisible(generated).TrimEnd('\uFFFD');
        if (next.Length > visible.Length && next.StartsWith(visible, StringComparison.Ordinal))
        {
            yield return next[visible.Length..];
            visible = next;
        }

        logits = model.Forward([token]);
        await Task.Yield();
    }
}
```

`BackendFactory.Create()` 自动选择后端：macOS 优先 Metal，其次 Vulkan（需要系统装有 Vulkan 驱动），都不可用时返回 `null`，即纯 CPU。也可以传 `"cpu"` / `"vulkan"` / `"metal"` 指定，或设置环境变量 `HYMT_BACKEND`。CLI 用 `--backend` 选择，Server 自动选择。

`threads = 0`（默认）自动绑物理 P-core，不占 SMT 和 E-core。`HunyuanDenseModel` 不是线程安全的，并发请求请串行化或各用独立实例。

## NuGet 包

| 包 | 版本 | 说明 |
| --- | --- | --- |
| `Sdcb.HyMT2Sharp` | [![NuGet](https://img.shields.io/nuget/v/Sdcb.HyMT2Sharp.svg)](https://www.nuget.org/packages/Sdcb.HyMT2Sharp) | 单 dll:GGUF、分词、KV cache、CPU SIMD kernel、Vulkan / Metal 后端、`Forward` |

`HyMT2Sharp.Cli`、`HyMT2Sharp.Server`、`HyMT2Sharp.WinForms`、`HyMT2Sharp.Benchmark` 是仓库内的示例与基准工具，不发布 NuGet。

`Sdcb.HyMT2Sharp` 还带 3 个 LINQPad 脚本，都引用 `Sdcb.GgufWeights.Hy-MT2-1.8B-2Bit` 权重包：`hello-world`（单次翻译，注释里可切换模型卡上的各类 prompt）、`cli`（交互式翻译，`/to <语言>` 切换目标语言）、`minimal-api`（ASP.NET Core minimal API，`POST /translate` 与流式 `GET /translate/stream`）。包打了 `linqpad-samples` 标签，免费版也能用。

## 性能

测试环境：Ryzen 7 5800X（Zen 3）、Windows、Release、8 线程，`avx2=True`、`vnni=False`。不计模型加载与 warmup；prefill 为 512 token 三次平均，decode 为 512 token 上下文后连续生成 128 token。完整数据（五种量化、同窗口 llama.cpp 对照、跨机器对比与带宽分析）见 [docs/perf.md](https://github.com/sdcb/HyMT2Sharp/blob/main/docs/perf.md)。

| 模型 | prefill 512 | decode 128 | prefill 三次 |
| --- | ---: | ---: | --- |
| HyMT2Sharp Q1.25 / STQ1_0 | **564.66 tok/s** | **47.75 tok/s** | 573.5 / 558.6 / 562.1 |
| HyMT2Sharp Q2_0C | 488.95 tok/s | 47.53 tok/s | 494.6 / 492.0 / 480.5 |
| HyMT2Sharp Q4_K_M | 423.99 tok/s | 26.55 tok/s | 426.8 / 419.8 / 425.5 |
| HyMT2Sharp Q6_K | 403.25 tok/s | 22.96 tok/s | 410.0 / 391.5 / 408.9 |
| HyMT2Sharp Q8_0 | 319.81 tok/s | 16.60 tok/s | 318.5 / 319.9 / 321.0 |
| llama.cpp Q4_K_M | 246.24 ± 0.82 tok/s | 31.57 ± 0.17 tok/s | `llama-bench -p 512 -n 128 -t 8 -ngl 0 -dev none` |

Q1.25 与 Q2 的 decode 基本持平；相比 Q4，Q1.25 prefill 快约 33%、decode 快约 80%。decode 全部贴着内存带宽墙，吞吐近似反比于权重大小。5800X 连续满载时频率与温度波动较大，上述数字不代表硬件上限。llama.cpp 加载不了 Q2_0C / STQ1_0，只能对照 K 系量化。

另一台**开发机B**（hybrid CPU，8 P-core + 16 E-core，DDR4-3200，AVX-VNNI）上复测过 Q4 / Q6 / Q8：**prefill 约 4–6 倍于 llama.cpp**（Q6 463.7 vs 119.1、Q8 548.8 vs 93.0 tok/s）；decode 同样贴带宽墙。完整数据、带宽微基准与噪声说明见 [docs/perf-B.md](https://github.com/sdcb/HyMT2Sharp/blob/main/docs/perf-B.md)。

复现：

```powershell
dotnet run --project src/HyMT2Sharp.Benchmark -c Release -- --model "D:\_\model\Hy-MT2-1.8B-1.25Bit.gguf" --bench-prefill 512 --bench-decode 128 --threads 8
dotnet run --project src/HyMT2Sharp.Benchmark -c Release -- --model "D:\_\model\Hy-MT2-1.8B-Q4_K_M.gguf" --bench-prefill 512 --bench-decode 128 --threads 8
dotnet run --project src/HyMT2Sharp.Benchmark -c Release -- --model "D:\_\model\Hy-MT2-1.8B-2Bit.gguf" --bench-prefill 512 --bench-decode 128 --threads 8
```

Q2 panel 默认 64 KiB tile，可用 `--q2-col-tile-kb 64` 显式指定。`--profile` 输出 STQ / Q2 / Q4 矩阵与注意力分项耗时。

## 实现要点

- **Q8_0**：34 B / 32 权重。层权重（含 token_embd，兼任 lm_head）重排成 8 列 panel；prefill 一次吃 8 个 token，decode 在 panel 上做 GEMV，权重按单条顺序流读取。有 AVX-VNNI 时内积用 `vpdpbusd`，否则用 `vpmaddubsw`。激活量化夹在 −127..127。QKV 与 gate/up 的 GEMV 各融合成一次并行调度；输出行按动态 chunk（`Interlocked.Add`，与 ggml mul_mat 相同思路）在 worker 间分配。
- **Q6_K**：沿用 `q6_Kx8 × q8_Kx4` panel。整模的 Q、K、V 和 gate/up 共用一次激活量化、一次并行调度，行分配同样是动态 chunk。内积固定 AVX2 `vpmaddubsw` + `vpmaddwd`（实测 VNNI 版在 Raptor Lake 上更慢，VNNI 函数保留供其它 CPU 与测试）。
- **decode attention**：scores → softmax → combine 按 head 融合在一次并行调度里，score 行保持在热缓存。
- **Q4_K_M**：`q4_Kx8 × q8_Kx4` AVX2 panel GEMM；decode 走 Q8 行量化 + GEMV。Q4 内核不走 VNNI。
- **Q2_0C**：8 列压缩 panel，保留 2-bit 权重，不展开为逐字节副本。
- **STQ1_0**：42 B / 256 权重的 stride-16 block，加载时重排为 8 行 panel，prefill / decode 均走 AVX2 GEMV / GEMM。
- **Q2 prefill**：QKV、gate/up、SiLU→down 复用 Q8 激活量化；2-bit 点积用 int32 归约避免 int16 溢出。
- **Portable SIMD**：分发顺序 AVX-VNNI → AVX2 → `Vector<T>`。无 AVX2 的平台（ARM64/NEON、老 x64）自动落到 portable 档：decode 的 Q4_K/Q6_K 走奇偶 i16 激活 + u16 lane 拆位的 `DotAct`（Q4_K_M 已追平 AVX2 的带宽墙），其余格式走 `DotVec`；prefill 走 BLIS 式外积 float GEMM（`VecGemmF`，寄存器驻留 6×2V 累加器 + FMA）。此时跳过 panel repack，省 ~40% 权重内存。`HYMT2SHARP_FORCE_PORTABLE=1` 可强制启用验证，实测见 [docs/perf.md](https://github.com/sdcb/HyMT2Sharp/blob/main/docs/perf.md) 第 5 节。
- Q2 尾部列与非对齐 token 路径保留。

## 开发与测试

```powershell
dotnet test tests/HyMT2Sharp.Tests -c Release
dotnet run --project src/HyMT2Sharp.Cli -c Release -- --model "D:\_\model\Hy-MT2-1.8B-2Bit.gguf" --prompt "Translate the following segment into Chinese, without additional explanation：SimdPaddleOCR is officially released today (NuGet: Sdcb.SimdPaddleOCR). It is a complete OCR inference engine written entirely in C#. It does not depend on Paddle Inference or ONNX Runtime, and it does not require shipping OpenCV native libraries." --max-tokens 128
dotnet run --project src/HyMT2Sharp.Cli -c Release -- --model "D:\_\model\Hy-MT2-1.8B-Q4_K_M.gguf" --prompt "Translate the following segment into Chinese, without additional explanation：SimdPaddleOCR is officially released today (NuGet: Sdcb.SimdPaddleOCR). It is a complete OCR inference engine written entirely in C#. It does not depend on Paddle Inference or ONNX Runtime, and it does not require shipping OpenCV native libraries." --max-tokens 128
```

## 微信群

![](https://io.starworks.cc:88/cv-public/2026/ocr-wxg-qr.png?1125)

如果微信群二维码过期了，请加入 QQ 群 [.NET骚操作 495782587](https://qm.qq.com/cgi-bin/qm/qr?_wv=1027&k=&authKey=&noverify=0&group_code=495782587)。

## 许可证

[Apache License 2.0](https://github.com/sdcb/HyMT2Sharp/blob/main/LICENSE)
