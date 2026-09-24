---
name: testing-hymt2sharp-server
description: How to launch and end-to-end test the HyMT2Sharp.Server HTTP API and bundled Web UI (ports, cwd quirk for wwwroot, request shapes, determinism checks)
---

# Testing HyMT2Sharp.Server

## Devin Secrets Needed
None — everything is local.

## Launch
- Build: `dotnet build HyMT2Sharp.slnx -c Release` (needs `export DOTNET_ROOT=/Users/devin/.dotnet PATH=/Users/devin/.dotnet:$PATH`).
- Binary: `src/HyMT2Sharp.Server/bin/Release/net10.0/Sdcb.HyMT2Sharp.Server --model <gguf> --urls http://127.0.0.1:<PORT>` (~10s model load; prints `listening` then ASP.NET `Now listening on`).
- Models in `/Users/devin/models/` (e.g. `Hy-MT2-1.8B-Q4_K_M.gguf`).
- **wwwroot quirk**: `wwwroot/` is NOT copied to build output. Launch with cwd=`src/HyMT2Sharp.Server` or the Web UI won't serve (`WebRootPath was not found` warning; API still works). `kill` the nohup wrapper PID may leave the real `Sdcb.HyMT2Sharp.Server` child alive — check `lsof -i :<PORT>` before relaunching.

## API shape
- `POST /v1/chat/completions` (also `/chat/completions`), `GET /health`, `GET /v1/models`, `GET /props`.
- JSON uses `SnakeCaseLower` naming + case-insensitive reads + ignore-null writes. Sampling fields: `temperature`, `top_p`, `top_k`, `min_p`, `repetition_penalty`, `seed`. `stream:true` → SSE `data: {chunk}\n\n` … `data: [DONE]`.
- Response text at `choices[0].message.content`; `id`/`created` are random per request — compare only `content` for determinism checks.
- Model is Hy-MT2 (translation). Good prompts: `Translate to English: <中文>` or `翻译成英文：<text>`.

## Determinism testing notes
- Greedy (no `temperature`, or `temperature:0`) is deterministic across repeats, including across KV prefix-cache hits (full-prefix hit replays the last token — logits stay identical).
- `top_k:1` with any seed must equal the greedy output (sole survivor = argmax).
- Same `seed` + same request → byte-identical `content`; different seed should diverge at temperature ≥ ~0.8.
- `repetition_penalty` A/B: hold `seed` constant, vary only the penalty; on a loop-inducing prompt (e.g. many `哈`) penalty=1 produces verbatim repeats and penalty≥4 visibly mutates/short-circuits the loop. Warm the KV cache with an identical throwaway request first so `FeedHistory` sees the same suffix in both arms.
- Malformed field types (e.g. `"temperature":"hot"`) → 400 from ASP.NET binding.
- Web UI at `/` exercises `stream:true` SSE and shows prefill/decode timings — good visual proof of the whole stack.
