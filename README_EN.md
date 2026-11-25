# HyMT2Sharp [![NuGet](https://img.shields.io/nuget/v/Sdcb.HyMT2Sharp.svg)](https://www.nuget.org/packages/Sdcb.HyMT2Sharp) [![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](https://github.com/sdcb/HyMT2Sharp/blob/main/LICENSE) [![QQ](https://img.shields.io/badge/QQ_Group-495782587-52B6EF?style=social&logo=tencent-qq&logoColor=000&logoWidth=20)](https://qm.qq.com/cgi-bin/qm/qr?_wv=1027&k=&authKey=&noverify=0&group_code=495782587)

[中文](https://github.com/sdcb/HyMT2Sharp/blob/main/README.md) | **English**

Unofficial pure C# inference for [Hy-MT3](https://huggingface.co/tencent/Hy-MT3-1.8B) and [Hy-MT2](https://huggingface.co/tencent/Hy-MT2-1.8B) (`hunyuan-dense`). Built-in AVX2 / AVX-512 / NEON SIMD kernels on CPU, plus Vulkan and Metal GPU backends. No llama.cpp or ONNX Runtime. Intended for in-process use.

Validated GGUF quantizations: **Q8_0**, **Q6_K**, **Q4_K_M**, **Q2_0C**, and **1.25-bit STQ1_0**. Other formats and model sizes have not been tested.

## Models

The `Sdcb.HyMT2Sharp` inference package ships no weights. Download a GGUF from Hugging Face, or reference a NuGet weight package:

| Model | Quant | Hugging Face | NuGet weight package | Parts | Total size |
| --- | --- | --- | --- | --- | --- |
| Hy-MT3-1.8B | Q4_K_M | [Hy-MT3-1.8B-GGUF](https://huggingface.co/tencent/Hy-MT3-1.8B-GGUF) | [Sdcb.GgufWeights.Hy-MT3-1.8B-Q4_K_M](https://www.nuget.org/packages/Sdcb.GgufWeights.Hy-MT3-1.8B-Q4_K_M) | 6 | ~1.3 GB |
| Hy-MT3-1.8B | Q8_0 | [Hy-MT3-1.8B-GGUF](https://huggingface.co/tencent/Hy-MT3-1.8B-GGUF) | [Sdcb.GgufWeights.Hy-MT3-1.8B-Q8_0](https://www.nuget.org/packages/Sdcb.GgufWeights.Hy-MT3-1.8B-Q8_0) | 8 | ~1.8 GB |
| Hy-MT2-1.8B | 1.25-bit STQ1_0 | [Hy-MT2-1.8B-1.25Bit-GGUF](https://huggingface.co/tencent/Hy-MT2-1.8B-1.25Bit-GGUF) | [Sdcb.GgufWeights.Hy-MT2-1.8B-1.25Bit](https://www.nuget.org/packages/Sdcb.GgufWeights.Hy-MT2-1.8B-1.25Bit) | 2 | ~440 MB |
| Hy-MT2-1.8B | Q2_0C | [Hy-MT2-1.8B-2Bit-GGUF](https://huggingface.co/tencent/Hy-MT2-1.8B-2Bit-GGUF) | [Sdcb.GgufWeights.Hy-MT2-1.8B-2Bit](https://www.nuget.org/packages/Sdcb.GgufWeights.Hy-MT2-1.8B-2Bit) | 3 | ~573 MB |
| Hy-MT2-1.8B | Q4_K_M | [Hy-MT2-1.8B-GGUF](https://huggingface.co/tencent/Hy-MT2-1.8B-GGUF) | [Sdcb.GgufWeights.Hy-MT2-1.8B-Q4_K_M](https://www.nuget.org/packages/Sdcb.GgufWeights.Hy-MT2-1.8B-Q4_K_M) | 5 | ~1.1 GB |
| Hy-MT2-1.8B | Q6_K | [Hy-MT2-1.8B-GGUF](https://huggingface.co/tencent/Hy-MT2-1.8B-GGUF) | [Sdcb.GgufWeights.Hy-MT2-1.8B-Q6_K](https://www.nuget.org/packages/Sdcb.GgufWeights.Hy-MT2-1.8B-Q6_K) | 7 | ~1.4 GB |
| Hy-MT2-1.8B | Q8_0 | [Hy-MT2-1.8B-GGUF](https://huggingface.co/tencent/Hy-MT2-1.8B-GGUF) | [Sdcb.GgufWeights.Hy-MT2-1.8B-Q8_0](https://www.nuget.org/packages/Sdcb.GgufWeights.Hy-MT2-1.8B-Q8_0) | 8 | ~1.8 GB |

Reference only the top-level package; its remaining part packages (`.Part2`…) restore as dependencies, and the part count includes the top-level package. The total size equals the model's GGUF size and is also what lands in your output folder. These packages are large, so the first restore can be slow (2Bit is ~573 MB and took ~7.5 minutes here). Load with `new HunyuanDenseModel(Sdcb.GgufWeights.Hy_MT2_1_8B_2Bit.Weights.Model())`.

## Quick start

Run the CLI from this repo. The CLI and server reference the NuGet package `Sdcb.GgufWeights.Hy-MT2-1.8B-2Bit` and load those embedded 2Bit weights when `--model` is omitted; pass `--model` to use your own GGUF:

```powershell
dotnet run --project src/HyMT2Sharp.Cli -c Release
dotnet run --project src/HyMT2Sharp.Cli -c Release -- --model "D:\_\model\Hy-MT2-1.8B-Q4_K_M.gguf"
```

Omit `--threads` to bind physical P-cores from the CPU topology (8 threads on an 8P+8E hybrid or a 5800X; P-cores only, no SMT). Pass `--prompt` for a single turn; omit it for a multi-turn console chat.

### HTTP server

`HyMT2Sharp.Server` exposes OpenAI-compatible `POST /v1/chat/completions` (including SSE) and a built-in chat page:

```powershell
dotnet run --project src/HyMT2Sharp.Server -c Release
```

`--model` (or the `HYMT2_MODEL` environment variable) selects a GGUF file instead.

Open `http://127.0.0.1:8080`, or call it with curl:

```powershell
curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" -d "{\"messages\":[{\"role\":\"user\",\"content\":\"Translate the following segment into Chinese, without additional explanation：SimdPaddleOCR is officially released today (NuGet: Sdcb.SimdPaddleOCR). It is a complete OCR inference engine written entirely in C#. It does not depend on Paddle Inference or ONNX Runtime, and it does not require shipping OpenCV native libraries.\"}],\"max_tokens\":128}"
```

### WinForms sample (Windows)

`HyMT2Sharp.WinForms` is a desktop translator: input on the left, streamed output on the right, a choice of engine (auto / CPU / Vulkan) and target language, and a status bar with load time and prefill / decode speed. It always uses the 2Bit weight package; swap the weight package reference in the csproj for another quantization. High DPI (PerMonitorV2) is supported.

```powershell
dotnet run --project src/HyMT2Sharp.WinForms -c Release
```

## Use as a library

Install the inference package:

```powershell
dotnet add package Sdcb.HyMT2Sharp
```

`HunyuanDenseModel` loads the GGUF, tokenizes, owns the KV cache, and runs `Forward`. The library ships `Sampler` (`Sampler.ArgMax` plus temperature / top-p / top-k sampling) but no built-in `Generate`; string assembly is left to the caller. Minimal greedy streaming example, with weights from the `Sdcb.GgufWeights.Hy-MT2-1.8B-2Bit` package (see [Models](#models)):

```csharp
using System.Runtime.CompilerServices;
using Sdcb.GgufWeights.Hy_MT2_1_8B_2Bit;
using Sdcb.HyMT2Sharp.Backends;
using Sdcb.HyMT2Sharp.Model;

// Or pass a GGUF path: new HunyuanDenseModel(@"D:\models\Hy-MT2-1.8B-Q4_K_M.gguf")
using HunyuanDenseModel model = new(Weights.Model(), backend: BackendFactory.Create());

await foreach (string piece in Generate(model, "Translate the following segment into Chinese, without additional explanation：SimdPaddleOCR is officially released today (NuGet: Sdcb.SimdPaddleOCR). It is a complete OCR inference engine written entirely in C#. It does not depend on Paddle Inference or ONNX Runtime, and it does not require shipping OpenCV native libraries."))
    Console.Write(piece);

// Full string: string text = string.Concat(await Generate(...).ToArrayAsync());

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

`BackendFactory.Create()` picks a backend automatically: Metal first on macOS, then Vulkan (needs a Vulkan driver on the system); if neither is usable it returns `null`, meaning plain CPU. Pass `"cpu"` / `"vulkan"` / `"metal"` to force one, or set the `HYMT_BACKEND` environment variable. The CLI selects with `--backend`; the server picks automatically.

`threads = 0` (the default) binds physical P-cores and skips SMT and E-cores. `HunyuanDenseModel` is not thread-safe; serialize requests or use one instance per caller.

## NuGet packages

| Package | Version | Notes |
| --- | --- | --- |
| `Sdcb.HyMT2Sharp` | [![NuGet](https://img.shields.io/nuget/v/Sdcb.HyMT2Sharp.svg)](https://www.nuget.org/packages/Sdcb.HyMT2Sharp) | Single dll: GGUF, tokenizer, KV cache, CPU SIMD kernels, Vulkan / Metal backends, `Forward` |

`HyMT2Sharp.Cli`, `HyMT2Sharp.Server`, `HyMT2Sharp.WinForms`, and `HyMT2Sharp.Benchmark` live in this repo as samples and tooling. They are not published to NuGet.

`Sdcb.HyMT2Sharp` also ships 3 LINQPad scripts, all referencing the `Sdcb.GgufWeights.Hy-MT2-1.8B-2Bit` weight package: `hello-world` (one translation; switch between the model card's prompt types by uncommenting), `cli` (interactive translator; `/to <language>` switches the target), and `minimal-api` (ASP.NET Core minimal API with `POST /translate` and streaming `GET /translate/stream`). The package is tagged `linqpad-samples`, so the free edition can use them.

## Performance

Environment: Ryzen 7 5800X (Zen 3), Windows, Release, 8 threads, `avx2=True`, `vnni=False`. Model load and warmup are excluded. Prefill is the mean of three 512-token runs. Decode is 128 tokens after a 512-token context. Full data (all five quantizations, same-window llama.cpp comparison, cross-machine numbers and bandwidth analysis): [docs/perf.md](https://github.com/sdcb/HyMT2Sharp/blob/main/docs/perf.md) (Chinese).

| Model | prefill 512 | decode 128 | prefill reps |
| --- | ---: | ---: | --- |
| HyMT2Sharp Q1.25 / STQ1_0 | **564.66 tok/s** | **47.75 tok/s** | 573.5 / 558.6 / 562.1 |
| HyMT2Sharp Q2_0C | 488.95 tok/s | 47.53 tok/s | 494.6 / 492.0 / 480.5 |
| HyMT2Sharp Q4_K_M | 423.99 tok/s | 26.55 tok/s | 426.8 / 419.8 / 425.5 |
| HyMT2Sharp Q6_K | 403.25 tok/s | 22.96 tok/s | 410.0 / 391.5 / 408.9 |
| HyMT2Sharp Q8_0 | 319.81 tok/s | 16.60 tok/s | 318.5 / 319.9 / 321.0 |
| llama.cpp Q4_K_M | 246.24 ± 0.82 tok/s | 31.57 ± 0.17 tok/s | `llama-bench -p 512 -n 128 -t 8 -ngl 0 -dev none` |

Q1.25 and Q2 decode at about the same rate. Versus Q4, Q1.25 prefill is ~33% faster and decode ~80% faster. Decode sits at the memory-bandwidth wall, so throughput falls off almost exactly in inverse proportion to weight size. A 5800X under sustained load will wander with clocks and temperature; these numbers are not a hardware ceiling. llama.cpp cannot load Q2_0C / STQ1_0, so only the K-quants can be compared.

A second box (**dev machine B**: hybrid CPU, 8 P-cores + 16 E-cores, DDR4-3200, AVX-VNNI) re-measured Q4 / Q6 / Q8 against CPU-only llama.cpp in the same window: **prefill is ~4–6× llama.cpp** (Q6 463.7 vs 119.1, Q8 548.8 vs 93.0 tok/s); decode sits at the same bandwidth wall. Full data, bandwidth microbenchmarks, and noise caveats: [docs/perf-B.md](https://github.com/sdcb/HyMT2Sharp/blob/main/docs/perf-B.md) (Chinese).

Reproduce:

```powershell
dotnet run --project src/HyMT2Sharp.Benchmark -c Release -- --model "D:\_\model\Hy-MT2-1.8B-1.25Bit.gguf" --bench-prefill 512 --bench-decode 128 --threads 8
dotnet run --project src/HyMT2Sharp.Benchmark -c Release -- --model "D:\_\model\Hy-MT2-1.8B-Q4_K_M.gguf" --bench-prefill 512 --bench-decode 128 --threads 8
dotnet run --project src/HyMT2Sharp.Benchmark -c Release -- --model "D:\_\model\Hy-MT2-1.8B-2Bit.gguf" --bench-prefill 512 --bench-decode 128 --threads 8
```

Q2 panels default to a 64 KiB tile (`--q2-col-tile-kb 64`). `--profile` prints STQ / Q2 / Q4 matmul and attention breakdowns.

## Implementation notes

- **Q8_0**: 34 B / 32 weights. Layer weights (including `token_embd`, which doubles as `lm_head`) are repacked into 8-column panels; prefill consumes 8 tokens per pass and decode runs GEMV directly on the panel so weights stream as one sequential read. Inner products use `vpdpbusd` with AVX-VNNI, else `vpmaddubsw`. Activations quantize to ±127. The QKV and gate/up GEMVs each fuse into a single parallel dispatch; output rows are claimed by workers in dynamic chunks (`Interlocked.Add`, same idea as ggml `mul_mat`).
- **Q6_K**: keeps the `q6_Kx8 × q8_Kx4` panel. Q, K, V and gate/up share one activation quantization and one parallel dispatch, with the same dynamic chunking. The inner product is fixed to AVX2 `vpmaddubsw` + `vpmaddwd` (the VNNI variant measured slower on Raptor Lake; VNNI code remains for other CPUs and tests).
- **Decode attention**: scores → softmax → combine are fused per head inside a single parallel dispatch so the score row stays cache-hot.
- **Q4_K_M**: `q4_Kx8 × q8_Kx4` AVX2 panel GEMM; decode uses Q8 row quant + GEMV.
- **Q2_0C**: compressed 8-column panels; 2-bit weights stay packed (no per-weight byte expansion).
- **STQ1_0**: 42 B / 256-weight stride-16 blocks, repacked to 8-row panels; prefill and decode both use AVX2 GEMV / GEMM.
- **Q2 prefill**: QKV, gate/up, and SiLU→down reuse Q8 activation quant; 2-bit dots reduce in int32 to avoid int16 overflow.
- **Portable SIMD**: dispatch order is AVX-VNNI → AVX2 → `Vector<T>`. Platforms without AVX2 (ARM64/NEON, older x64) fall to the portable tier automatically: Q4_K/Q6_K decode runs `DotAct` (even/odd i16 activations, weights split from u16 lanes; Q4_K_M reaches the same bandwidth wall as AVX2), other formats run `DotVec`; prefill runs a BLIS-style outer-product float GEMM (`VecGemmF`, 6×2V register-resident accumulators with FMA). Panel repack is skipped in this mode, saving ~40% weight memory. `HYMT2SHARP_FORCE_PORTABLE=1` forces it for verification; measured numbers are in [docs/perf.md](https://github.com/sdcb/HyMT2Sharp/blob/main/docs/perf.md) section 5.
- Q2 tail columns and non-aligned token paths remain.

## Development

```powershell
dotnet test tests/HyMT2Sharp.Tests -c Release
dotnet run --project src/HyMT2Sharp.Cli -c Release -- --model "D:\_\model\Hy-MT2-1.8B-2Bit.gguf" --prompt "Translate the following segment into Chinese, without additional explanation：SimdPaddleOCR is officially released today (NuGet: Sdcb.SimdPaddleOCR). It is a complete OCR inference engine written entirely in C#. It does not depend on Paddle Inference or ONNX Runtime, and it does not require shipping OpenCV native libraries." --max-tokens 128
dotnet run --project src/HyMT2Sharp.Cli -c Release -- --model "D:\_\model\Hy-MT2-1.8B-Q4_K_M.gguf" --prompt "Translate the following segment into Chinese, without additional explanation：SimdPaddleOCR is officially released today (NuGet: Sdcb.SimdPaddleOCR). It is a complete OCR inference engine written entirely in C#. It does not depend on Paddle Inference or ONNX Runtime, and it does not require shipping OpenCV native libraries." --max-tokens 128
```

## WeChat group

![](https://io.starworks.cc:88/cv-public/2026/ocr-wxg-qr.png?1125)

If the WeChat QR code has expired, join the QQ group [.NET骚操作 495782587](https://qm.qq.com/cgi-bin/qm/qr?_wv=1027&k=&authKey=&noverify=0&group_code=495782587).

## License

[Apache License 2.0](https://github.com/sdcb/HyMT2Sharp/blob/main/LICENSE)
