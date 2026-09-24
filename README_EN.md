# HyMT2Sharp [![NuGet](https://img.shields.io/nuget/v/Sdcb.HyMT2Sharp.svg)](https://www.nuget.org/packages/Sdcb.HyMT2Sharp) [![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE) [![QQ](https://img.shields.io/badge/QQ_Group-495782587-52B6EF?style=social&logo=tencent-qq&logoColor=000&logoWidth=20)](https://qm.qq.com/cgi-bin/qm/qr?_wv=1027&k=&authKey=&noverify=0&group_code=495782587)

[中文](README.md) | **English**

Unofficial pure C# CPU inference for [Hy-MT2](https://huggingface.co/tencent/Hy-MT2-1.8B) (`hunyuan-dense`). No llama.cpp or ONNX Runtime; AVX2 kernels are built in. Intended for in-process use.

Validated GGUF quantizations: **Q8_0**, **Q6_K**, **Q4_K_M**, **Q2_0C**, and **1.25-bit STQ1_0**. Other formats and model sizes have not been tested.

## Models

Weights are not shipped in the NuGet packages. Download a GGUF yourself:

| Quant | Hugging Face |
| --- | --- |
| 1.25-bit STQ1_0 | [Hy-MT2-1.8B-1.25Bit-GGUF](https://huggingface.co/tencent/Hy-MT2-1.8B-1.25Bit-GGUF) |
| Q2_0C | [Hy-MT2-1.8B-2Bit-GGUF](https://huggingface.co/tencent/Hy-MT2-1.8B-2Bit-GGUF) |
| Q8_0 | [Hy-MT2-1.8B-GGUF](https://huggingface.co/tencent/Hy-MT2-1.8B-GGUF) |
| Q6_K | [Hy-MT2-1.8B-GGUF](https://huggingface.co/tencent/Hy-MT2-1.8B-GGUF) |
| Q4_K_M | [Hy-MT2-1.8B-GGUF](https://huggingface.co/tencent/Hy-MT2-1.8B-GGUF) |

## Quick start

Run the CLI from this repo (point `--model` at your GGUF):

```powershell
dotnet run --project src/HyMT2Sharp.Cli -c Release -- --model "D:\_\model\Hy-MT2-1.8B-Q4_K_M.gguf"
```

Omit `--threads` to bind physical P-cores from the CPU topology (8 threads on an 8P+8E hybrid or a 5800X; P-cores only, no SMT). Pass `--prompt` for a single turn; omit it for a multi-turn console chat.

### HTTP server

`HyMT2Sharp.Server` exposes OpenAI-compatible `POST /v1/chat/completions` (including SSE) and a built-in chat page:

```powershell
dotnet run --project src/HyMT2Sharp.Server -c Release -- --model "D:\_\model\Hy-MT2-1.8B-Q4_K_M.gguf"
```

Open `http://127.0.0.1:8080`, or call it with curl:

```powershell
curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" -d "{\"messages\":[{\"role\":\"user\",\"content\":\"Translate the following segment into Chinese, without additional explanation：SimdPaddleOCR is officially released today (NuGet: Sdcb.SimdPaddleOCR). It is a complete OCR inference engine written entirely in C#. It does not depend on Paddle Inference or ONNX Runtime, and it does not require shipping OpenCV native libraries.\"}],\"max_tokens\":128}"
```

## Use as a library

Install the inference package (Gguf / Kernels / backends are merged into a single assembly):

```powershell
dotnet add package Sdcb.HyMT2Sharp
```

`HunyuanDenseModel` loads the GGUF, tokenizes, owns the KV cache, and runs `Forward`. Sampling and string assembly are left to the caller — there is no built-in `Generate` / `ArgMax`. Minimal greedy streaming example:

```csharp
using Sdcb.HyMT2Sharp.Model;

using HunyuanDenseModel model = new(@"D:\_\model\Hy-MT2-1.8B-1.25Bit.gguf");

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
        int token = ArgMax(logits);
        if (model.Tokenizer.IsStop(token))
            break;

        generated.Add(token);
        string next = model.Tokenizer.DecodeVisible(generated);
        if (next.Length > visible.Length && next.StartsWith(visible, StringComparison.Ordinal))
            yield return next[visible.Length..];
        visible = next;

        logits = model.Forward([token]);
        await Task.Yield();
    }
}

static int ArgMax(float[] logits)
{
    int best = 0;
    for (int i = 1; i < logits.Length; i++)
        if (logits[i] > logits[best])
            best = i;
    return best;
}
```

`threads = 0` (the default) binds physical P-cores and skips SMT and E-cores. `HunyuanDenseModel` is not thread-safe; serialize requests or use one instance per caller.

## NuGet packages

| Package | Version | Notes |
| --- | --- | --- |
| `Sdcb.HyMT2Sharp` | [![NuGet](https://img.shields.io/nuget/v/Sdcb.HyMT2Sharp.svg)](https://www.nuget.org/packages/Sdcb.HyMT2Sharp) | Single dll: GGUF, tokenizer, KV cache, CPU SIMD kernels, `Forward` |

`HyMT2Sharp.Cli`, `HyMT2Sharp.Server`, and `HyMT2Sharp.Benchmark` live in this repo as samples and tooling. They are not published to NuGet.

## Performance

Environment: Ryzen 7 5800X (Zen 3), Windows, Release, 8 threads, `avx2=True`, `vnni=False`. Model load and warmup are excluded. Prefill is the mean of three 512-token runs. Decode is 128 tokens after a 512-token context. Full data (all five quantizations, same-window llama.cpp comparison, cross-machine numbers and bandwidth analysis): [docs/perf.md](docs/perf.md) (Chinese).

| Model | prefill 512 | decode 128 | prefill reps |
| --- | ---: | ---: | --- |
| HyMT2Sharp Q1.25 / STQ1_0 | **564.66 tok/s** | **47.75 tok/s** | 573.5 / 558.6 / 562.1 |
| HyMT2Sharp Q2_0C | 488.95 tok/s | 47.53 tok/s | 494.6 / 492.0 / 480.5 |
| HyMT2Sharp Q4_K_M | 423.99 tok/s | 26.55 tok/s | 426.8 / 419.8 / 425.5 |
| HyMT2Sharp Q6_K | 403.25 tok/s | 22.96 tok/s | 410.0 / 391.5 / 408.9 |
| HyMT2Sharp Q8_0 | 319.81 tok/s | 16.60 tok/s | 318.5 / 319.9 / 321.0 |
| llama.cpp Q4_K_M | 246.24 ± 0.82 tok/s | 31.57 ± 0.17 tok/s | `llama-bench -p 512 -n 128 -t 8 -ngl 0 -dev none` |

Q1.25 and Q2 decode at about the same rate. Versus Q4, Q1.25 prefill is ~33% faster and decode ~80% faster. Decode sits at the memory-bandwidth wall, so throughput falls off almost exactly in inverse proportion to weight size. A 5800X under sustained load will wander with clocks and temperature; these numbers are not a hardware ceiling. llama.cpp cannot load Q2_0C / STQ1_0, so only the K-quants can be compared.

A second box (**dev machine B**: hybrid CPU, 8 P-cores + 16 E-cores, DDR4-3200, AVX-VNNI) re-measured Q4 / Q6 / Q8 against CPU-only llama.cpp in the same window: **prefill is ~4–6× llama.cpp** (Q6 463.7 vs 119.1, Q8 548.8 vs 93.0 tok/s); decode sits at the same bandwidth wall. Full data, bandwidth microbenchmarks, and noise caveats: [docs/perf-B.md](docs/perf-B.md) (Chinese).

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
- **Portable SIMD**: dispatch order is AVX-VNNI → AVX2 → `Vector<T>`. Platforms without AVX2 (ARM64/NEON, older x64) fall to the portable tier automatically: Q4_K/Q6_K decode runs `DotAct` (even/odd i16 activations, weights split from u16 lanes; Q4_K_M reaches the same bandwidth wall as AVX2), other formats run `DotVec`; prefill runs a BLIS-style outer-product float GEMM (`VecGemmF`, 6×2V register-resident accumulators with FMA). Panel repack is skipped in this mode, saving ~40% weight memory. `HYMT2SHARP_FORCE_PORTABLE=1` forces it for verification; measured numbers are in [docs/perf.md](docs/perf.md) section 5.
- Q2 tail columns and non-aligned token paths remain.

## Development

```powershell
dotnet test tests/HyMT2Sharp.Tests -c Release
dotnet run --project src/HyMT2Sharp.Cli -c Release -- --model "D:\_\model\Hy-MT2-1.8B-2Bit.gguf" --prompt "Translate the following segment into Chinese, without additional explanation：SimdPaddleOCR is officially released today (NuGet: Sdcb.SimdPaddleOCR). It is a complete OCR inference engine written entirely in C#. It does not depend on Paddle Inference or ONNX Runtime, and it does not require shipping OpenCV native libraries." --max-tokens 128
dotnet run --project src/HyMT2Sharp.Cli -c Release -- --model "D:\_\model\Hy-MT2-1.8B-Q4_K_M.gguf" --prompt "Translate the following segment into Chinese, without additional explanation：SimdPaddleOCR is officially released today (NuGet: Sdcb.SimdPaddleOCR). It is a complete OCR inference engine written entirely in C#. It does not depend on Paddle Inference or ONNX Runtime, and it does not require shipping OpenCV native libraries." --max-tokens 128
```

## WeChat group

![](https://io.starworks.cc:88/cv-public/2026/ocr-wxg-qr.png)

If the WeChat QR code has expired, join the QQ group [.NET骚操作 495782587](https://qm.qq.com/cgi-bin/qm/qr?_wv=1027&k=&authKey=&noverify=0&group_code=495782587).

## License

[Apache License 2.0](LICENSE)
