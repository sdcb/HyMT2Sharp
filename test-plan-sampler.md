# Test plan — Sampler end-to-end over HTTP (devin/1790246352-sampler, 1f18f8e)

## Setup (done)
- Built `dotnet build HyMT2Sharp.slnx -c Release` — clean, 0 warnings.
- Server: `src/HyMT2Sharp.Server/bin/Release/net10.0/Sdcb.HyMT2Sharp.Server --model /Users/devin/models/Hy-MT2-1.8B-Q4_K_M.gguf --urls http://127.0.0.1:8399`
- Endpoint: `POST /v1/chat/completions`; JSON `SnakeCaseLower` naming → `temperature`, `top_p`, `top_k`, `min_p`, `repetition_penalty`, `seed`.
- SSE: `data: <json>\n\n` chunks, terminal `data: [DONE]`.
- Model is a translation model (Hy-MT2): prompts are translation requests.
- KV-cache note: identical repeated prompts hit full-prefix reuse (replays last token) — logits deterministic, so same-seed comparisons across repeats are valid. `FeedHistory` only sees the uncached prompt suffix; penalty A/B uses generated-token divergence with identical cache state (warm both arms with an identical throwaway request first).

Prompts used:
- P_translate: `Translate to English: 今天天气很好，我们一起去公园散步吧。`
- P_loop: `翻译成英文：哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈` (induces repeated tokens)

All requests use `max_tokens` 48–96 to keep decode fast. Compare `choices[0].message.content` (id/created differ by design — do NOT compare whole bodies).

## Tests

### T1 — Greedy baseline unchanged (no sampling params)
- POST P_translate, max_tokens=64, no sampling fields → save content `G1`.
- Repeat identical request → `G2`.
- PASS: HTTP 200 both; `G1 == G2` exactly (greedy determinism incl. warm-cache path); content non-empty.

### T2 — Sampling params accepted
- POST P_translate + `temperature:0.8, top_p:0.9, top_k:40` → PASS: HTTP 200, non-empty content, valid JSON (no 400 binding error).

### T3 — Seed determinism / wiring
- POST P_translate + `temperature:1.0, top_p:0.95, seed:42`, max_tokens=64 → `S1`.
- Identical request again → `S2`. PASS: `S1 == S2` byte-identical.
- Same but `seed:43` → `S3`. PASS: `S3 != S1` (proves seed actually wired; if identical after 2 alt seeds, mark FAILED/inconclusive).

### T4 — temperature=0 equals historical greedy
- POST P_translate + `temperature:0` → `T0`. PASS: `T0 == G1` exactly. (Adversarial: proves temp=0 maps to argmax AND the full greedy pipeline matches no-params behavior.)

### T5 — top_k=1 equals greedy
- POST P_translate + `temperature:1.0, top_k:1, seed:999` → `K1`. PASS: `K1 == G1` exactly. (Sole surviving candidate = argmax; proves top_k truncation actually selects the max.)

### T6 — repetition_penalty changes behavior
- Warm cache: POST P_loop once (throwaway, temp 1.0 seed 7).
- `repetition_penalty:1.0, temperature:1.0, top_p:0.95, seed:7`, max_tokens=96 → `P1`.
- Same but `repetition_penalty:8` → `P2`.
- PASS: `P1 != P2` (penalty provably applied; if identical → penalty ignored → FAIL). Secondary: dominant-token frequency in `P2` ≤ `P1` on a loopy output — report counts; divergence alone is the hard criterion.

### T7 — min_p alone
- POST P_translate + `temperature:1.0, min_p:0.1` → PASS: 200, non-empty content.

### T8 — Streaming with sampling params
- POST P_translate + `stream:true, temperature:0.8, top_p:0.9, max_tokens:32, seed:5` via `curl -N`.
- PASS: multiple `data:` chunks, ≥1 chunk with `delta.content`, a chunk with `finish_reason`, terminal `data: [DONE]`.

### T9 — Malformed field type → 400 (negative)
- POST P_translate + `temperature:"hot"` → PASS: HTTP 400 (JSON binding rejection), not a 200/crash.

### T10 — Web UI smoke (visual regression)
- Open `http://127.0.0.1:8399/` in browser; send `Translate to English: 你好世界`; PASS: streaming reply text appears in the chat UI. Recorded.
