using System.Diagnostics;
using System.Runtime.Intrinsics.X86;
using Sdcb.HyMT2Sharp.Gguf;
using Sdcb.HyMT2Sharp.Kernels;

namespace Sdcb.HyMT2Sharp.Model;

public sealed unsafe partial class HunyuanDenseModel
{
    /// <summary>
    /// Prefill-only block. May use pair GEMM, SiLU→q8, and thread-pool Barriers.
    /// Decode must not enter this method.
    /// </summary>
    private void PrefillBlock(float* hidden, int layer, int seq, int start)
    {
        int hiddenSize = Config.HiddenSize;
        float* n1 = (float*)Bump((nuint)((long)seq * hiddenSize * sizeof(float)));
        float* attn = (float*)Bump((nuint)((long)seq * hiddenSize * sizeof(float)));
        Rms($"blk.{layer}.attn_norm.weight", hidden, n1, seq, hiddenSize);
        PrefillAttention(n1, attn, layer, seq, start);
        Ops.AddInPlace(hidden, attn, seq * hiddenSize, _pool);

        float* n2 = (float*)Bump((nuint)((long)seq * hiddenSize * sizeof(float)));
        Rms($"blk.{layer}.ffn_norm.weight", hidden, n2, seq, hiddenSize);
        float* gate = (float*)Bump((nuint)((long)seq * Config.FfnSize * sizeof(float)));
        float* up = (float*)Bump((nuint)((long)seq * Config.FfnSize * sizeof(float)));
        float* down = (float*)Bump((nuint)((long)seq * hiddenSize * sizeof(float)));
        PrefillGateUp(n2, gate, up, layer, seq);
        PrefillDown(gate, up, down, layer, seq);
        Ops.AddInPlace(hidden, down, seq * hiddenSize, _pool);
    }

    private void PrefillAttention(float* input, float* output, int layer, int seq, int start)
    {
        int heads = Config.NumHeads;
        int kvHeads = Config.NumKvHeads;
        int dim = Config.HeadDim;
        int qDim = heads * dim;
        int kDim = kvHeads * dim;
        float* q = (float*)Bump((nuint)((long)seq * qDim * sizeof(float)));
        float* k = _cacheK[layer] + start * kDim;
        float* v = _cacheV[layer] + start * kDim;
        PrefillQkv(input, q, k, v, layer, seq);

        long tRope = ProfileEnabled ? Stopwatch.GetTimestamp() : 0;
        Ops.NeoXRoPE(q, seq, heads, dim, Config.RopeDim, start, Config.RopeBase, _pool);
        Ops.NeoXRoPE(k, seq, kvHeads, dim, Config.RopeDim, start, Config.RopeBase, _pool);
        if (ProfileEnabled)
            TicksRope += Stopwatch.GetTimestamp() - tRope;
        Rms($"blk.{layer}.attn_q_norm.weight", q, q, seq * heads, dim);
        Rms($"blk.{layer}.attn_k_norm.weight", k, k, seq * kvHeads, dim);

        int kvStride = kDim;
        int kvLen = start + seq;
        float scale = 1f / MathF.Sqrt(dim);
        float* sc = (float*)Bump((nuint)((long)heads * seq * kvLen * sizeof(float)));
        long t0 = ProfileEnabled ? Stopwatch.GetTimestamp() : 0;
        Ops.AttentionScores(q, _cacheK[layer], sc, heads, kvHeads, dim, seq, kvLen, qDim, kvStride, scale, start, _pool);
        if (ProfileEnabled)
        {
            TicksAttnScore += Stopwatch.GetTimestamp() - t0;
            t0 = Stopwatch.GetTimestamp();
        }

        Ops.SoftmaxCausal(sc, heads, seq, kvLen, start, _pool);
        if (ProfileEnabled)
        {
            TicksSoftmax += Stopwatch.GetTimestamp() - t0;
            t0 = Stopwatch.GetTimestamp();
        }

        float* ao = (float*)Bump((nuint)((long)seq * qDim * sizeof(float)));
        Ops.AttentionCombine(_cacheV[layer], sc, ao, heads, kvHeads, dim, seq, kvLen, qDim, kvStride, start, _pool);
        if (ProfileEnabled)
            TicksAttnCombine += Stopwatch.GetTimestamp() - t0;

        Linear(ao, $"blk.{layer}.attn_output.weight", output, qDim, Config.HiddenSize, seq);
    }

    private void PrefillQkv(float* input, float* q, float* k, float* v, int layer, int seq)
    {
        Weight wq = _weights[$"blk.{layer}.attn_q.weight"];
        Weight wk = _weights[$"blk.{layer}.attn_k.weight"];
        Weight wv = _weights[$"blk.{layer}.attn_v.weight"];
        int hidden = Config.HiddenSize;
        int qDim = Config.NumHeads * Config.HeadDim;
        int kDim = Config.NumKvHeads * Config.HeadDim;
        if (Simd.UseAvx2 && (seq & 3) == 0 &&
            wq.Type == GgmlTensorType.STQ1_0 && wk.Type == GgmlTensorType.STQ1_0 && wv.Type == GgmlTensorType.STQ1_0 &&
            wq.PackedSTQ != null && wk.PackedSTQ != null && wv.PackedSTQ != null &&
            (qDim & 7) == 0 && (kDim & 7) == 0)
        {
            long t2 = ProfileEnabled ? Stopwatch.GetTimestamp() : 0;
            BlockQ8Kx4* q8 = STQPrefillBuffer(hidden, seq);
            MulMatSTQ.QuantizeAndGemm(input, q8, hidden, seq, _pool,
                new STQPanelWeight(wq.PackedSTQ, q, qDim),
                new STQPanelWeight(wk.PackedSTQ, k, kDim),
                new STQPanelWeight(wv.PackedSTQ, v, kDim));
            if (ProfileEnabled) TicksSTQ += Stopwatch.GetTimestamp() - t2;
            return;
        }
        if (Simd.UseAvx2 && (seq & 3) == 0 &&
            wq.Type == GgmlTensorType.Q2_0C && wk.Type == GgmlTensorType.Q2_0C && wv.Type == GgmlTensorType.Q2_0C &&
            wq.Packed2 != null && wk.Packed2 != null && wv.Packed2 != null &&
            (qDim & 7) == 0 && (kDim & 7) == 0)
        {
            long t2 = ProfileEnabled ? Stopwatch.GetTimestamp() : 0;
            BlockQ8Kx4* q8 = Q2PrefillBuffer(hidden, seq);
            MulMatQ2.QuantizeAndGemm(input, q8, hidden, seq, _pool,
                new Q2PanelWeight(wq.Packed2, q, qDim),
                new Q2PanelWeight(wk.Packed2, k, kDim),
                new Q2PanelWeight(wv.Packed2, v, kDim));
            if (ProfileEnabled) TicksQ2 += Stopwatch.GetTimestamp() - t2;
            return;
        }
        if (Simd.UseAvx2 && (seq & 3) == 0 &&
            wq.Type == GgmlTensorType.Q8_0 && wk.Type == GgmlTensorType.Q8_0 && wv.Type == GgmlTensorType.Q8_0 &&
            wq.Packed8 != null && wk.Packed8 != null && wv.Packed8 != null &&
            (qDim & 7) == 0 && (kDim & 7) == 0)
        {
            long t8 = ProfileEnabled ? Stopwatch.GetTimestamp() : 0;
            BlockQ8_0x4* q8 = Q8PrefillBuffer(hidden, seq);
            MulMatQ8_0.QuantizeAndGemm(input, q8, hidden, seq, _pool,
                new Q8Panel(wq.Packed8, q, qDim),
                new Q8Panel(wk.Packed8, k, kDim),
                new Q8Panel(wv.Packed8, v, kDim));
            if (ProfileEnabled) TicksQ8 += Stopwatch.GetTimestamp() - t8;
            return;
        }
        if (Simd.UseAvx2 && (seq & 3) == 0 && wq.Packed6 != null && wk.Packed6 != null && wv.Packed6 != null &&
            (qDim & 7) == 0 && (kDim & 7) == 0)
        {
            int nb6 = hidden / Qk.SuperBlock;
            nuint actBytes6 = (nuint)((seq / 4) * nb6 * Qk.Q8Kx4Size);
            BlockQ8Kx4* q8 = (BlockQ8Kx4*)_gemmScratch.E(actBytes6);
            long t6 = ProfileEnabled ? Stopwatch.GetTimestamp() : 0;
            MulMatPanel.QuantizeAndGemm(
                input, q8, hidden, seq, _pool,
                PanelWeight.ForQ6(wq.Packed6, q, qDim),
                PanelWeight.ForQ6(wk.Packed6, k, kDim),
                PanelWeight.ForQ6(wv.Packed6, v, kDim));
            if (ProfileEnabled)
                TicksQ6 += Stopwatch.GetTimestamp() - t6;
            return;
        }
        if (Simd.UseAvx2 && (seq & 3) == 0 && wq.Packed != null && wk.Packed != null && (wv.Packed != null || wv.Packed6 != null))
        {
            int nb = hidden / Qk.SuperBlock;
            nuint actBytes = (nuint)((seq / 4) * nb * Qk.Q8Kx4Size);
            BlockQ8Kx4* q8 = (BlockQ8Kx4*)_gemmScratch.E(actBytes);
            long t0 = ProfileEnabled ? Stopwatch.GetTimestamp() : 0;
            MulMatPanel.QuantizeAndGemm(
                input, q8, hidden, seq, _pool,
                PanelWeight.ForQ4(wq.Packed, wq.Meta, q, qDim),
                PanelWeight.ForQ4(wk.Packed, wk.Meta, k, kDim),
                Panel(wv, v, kDim));
            if (ProfileEnabled)
                TicksQ4 += Stopwatch.GetTimestamp() - t0;
            return;
        }

        Linear(input, $"blk.{layer}.attn_q.weight", q, hidden, qDim, seq);
        Linear(input, $"blk.{layer}.attn_k.weight", k, hidden, kDim, seq);
        Linear(input, $"blk.{layer}.attn_v.weight", v, hidden, kDim, seq);
    }

    private static PanelWeight Panel(Weight w, float* dst, int nOut) =>
        w.Packed6 != null ? PanelWeight.ForQ6(w.Packed6, dst, nOut) : PanelWeight.ForQ4(w.Packed, w.Meta, dst, nOut);

    private void PrefillGateUp(float* input, float* gate, float* up, int layer, int seq)
    {
        Weight wg = _weights[$"blk.{layer}.ffn_gate.weight"];
        Weight wu = _weights[$"blk.{layer}.ffn_up.weight"];
        int hidden = Config.HiddenSize;
        int ffn = Config.FfnSize;
        if (Simd.UseAvx2 && (seq & 3) == 0 &&
            wg.Type == GgmlTensorType.STQ1_0 && wu.Type == GgmlTensorType.STQ1_0 &&
            wg.PackedSTQ != null && wu.PackedSTQ != null && (ffn & 7) == 0)
        {
            long t2 = ProfileEnabled ? Stopwatch.GetTimestamp() : 0;
            BlockQ8Kx4* q8 = STQPrefillBuffer(hidden, seq);
            MulMatSTQ.QuantizeAndGemm(input, q8, hidden, seq, _pool,
                new STQPanelWeight(wg.PackedSTQ, gate, ffn),
                new STQPanelWeight(wu.PackedSTQ, up, ffn));
            if (ProfileEnabled) TicksSTQ += Stopwatch.GetTimestamp() - t2;
            return;
        }
        if (Simd.UseAvx2 && (seq & 3) == 0 && wg.Packed6 != null && wu.Packed6 != null && (ffn & 7) == 0)
        {
            int nb6 = hidden / Qk.SuperBlock;
            nuint actBytes6 = (nuint)((seq / 4) * nb6 * Qk.Q8Kx4Size);
            BlockQ8Kx4* q8 = (BlockQ8Kx4*)_gemmScratch.E(actBytes6);
            long t6 = ProfileEnabled ? Stopwatch.GetTimestamp() : 0;
            MulMatPanel.QuantizeAndGemm(
                input, q8, hidden, seq, _pool,
                PanelWeight.ForQ6(wg.Packed6, gate, ffn),
                PanelWeight.ForQ6(wu.Packed6, up, ffn));
            if (ProfileEnabled)
                TicksQ6 += Stopwatch.GetTimestamp() - t6;
            return;
        }
        if (Simd.UseAvx2 && (seq & 3) == 0 &&
            wg.Type == GgmlTensorType.Q8_0 && wu.Type == GgmlTensorType.Q8_0 &&
            wg.Packed8 != null && wu.Packed8 != null && (ffn & 7) == 0)
        {
            long t8 = ProfileEnabled ? Stopwatch.GetTimestamp() : 0;
            BlockQ8_0x4* q8 = Q8PrefillBuffer(hidden, seq);
            MulMatQ8_0.QuantizeAndGemm(input, q8, hidden, seq, _pool,
                new Q8Panel(wg.Packed8, gate, ffn),
                new Q8Panel(wu.Packed8, up, ffn));
            if (ProfileEnabled) TicksQ8 += Stopwatch.GetTimestamp() - t8;
            return;
        }
        if (Simd.UseAvx2 && (seq & 3) == 0 && wg.Packed != null && wu.Packed != null)
        {
            int nb = hidden / Qk.SuperBlock;
            nuint actBytes = (nuint)((seq / 4) * nb * Qk.Q8Kx4Size);
            BlockQ8Kx4* q8 = (BlockQ8Kx4*)_gemmScratch.E(actBytes);
            long t0 = ProfileEnabled ? Stopwatch.GetTimestamp() : 0;
            MulMatPanel.QuantizeAndGemm(
                input, q8, hidden, seq, _pool,
                PanelWeight.ForQ4(wg.Packed, wg.Meta, gate, ffn),
                PanelWeight.ForQ4(wu.Packed, wu.Meta, up, ffn));
            if (ProfileEnabled)
                TicksQ4 += Stopwatch.GetTimestamp() - t0;
            return;
        }

        if (Simd.UseAvx2 && (seq & 3) == 0 &&
            wg.Type == GgmlTensorType.Q2_0C && wu.Type == GgmlTensorType.Q2_0C &&
            wg.Packed2 != null && wu.Packed2 != null && (ffn & 7) == 0)
        {
            long t2 = ProfileEnabled ? Stopwatch.GetTimestamp() : 0;
            BlockQ8Kx4* q8 = Q2PrefillBuffer(hidden, seq);
            MulMatQ2.QuantizeAndGemm(input, q8, hidden, seq, _pool,
                new Q2PanelWeight(wg.Packed2, gate, ffn),
                new Q2PanelWeight(wu.Packed2, up, ffn));
            if (ProfileEnabled) TicksQ2 += Stopwatch.GetTimestamp() - t2;
            return;
        }

        Linear(input, $"blk.{layer}.ffn_gate.weight", gate, hidden, ffn, seq);
        Linear(input, $"blk.{layer}.ffn_up.weight", up, hidden, ffn, seq);
    }

    private void PrefillDown(float* gate, float* up, float* down, int layer, int seq)
    {
        Weight wd = _weights[$"blk.{layer}.ffn_down.weight"];
        int hidden = Config.HiddenSize;
        int ffn = Config.FfnSize;
        if (Simd.UseAvx2 && (seq & 3) == 0 && wd.Type == GgmlTensorType.STQ1_0 &&
            wd.PackedSTQ != null && (hidden & 7) == 0)
        {
            long t2 = ProfileEnabled ? Stopwatch.GetTimestamp() : 0;
            BlockQ8Kx4* q8 = STQPrefillBuffer(ffn, seq);
            MulMatSTQ.SiluQuantizeAndGemm(gate, up, q8, ffn, seq, _pool,
                new STQPanelWeight(wd.PackedSTQ, down, hidden));
            if (ProfileEnabled) TicksSTQ += Stopwatch.GetTimestamp() - t2;
            return;
        }
        if (Simd.UseAvx2 && (seq & 3) == 0 && wd.Type == GgmlTensorType.Q8_0 &&
            wd.Packed8 != null && (hidden & 7) == 0)
        {
            long t8 = ProfileEnabled ? Stopwatch.GetTimestamp() : 0;
            BlockQ8_0x4* q8 = Q8PrefillBuffer(ffn, seq);
            MulMatQ8_0.SiluQuantizeAndGemm(gate, up, q8, ffn, seq, _pool,
                new Q8Panel(wd.Packed8, down, hidden));
            if (ProfileEnabled) TicksQ8 += Stopwatch.GetTimestamp() - t8;
            return;
        }
        if (Simd.UseAvx2 && (seq & 3) == 0 && (wd.Packed != null || wd.Packed6 != null))
        {
            int nb = ffn / Qk.SuperBlock;
            nuint actBytes = (nuint)((seq / 4) * nb * Qk.Q8Kx4Size);
            BlockQ8Kx4* q8 = (BlockQ8Kx4*)_gemmScratch.A(actBytes);
            long t0 = ProfileEnabled ? Stopwatch.GetTimestamp() : 0;
            MulMatPanel.SiluQuantizeAndGemm(gate, up, q8, ffn, seq, _pool, Panel(wd, down, hidden));
            if (ProfileEnabled)
            {
                if (wd.Packed6 != null)
                    TicksQ6 += Stopwatch.GetTimestamp() - t0;
                else
                    TicksQ4 += Stopwatch.GetTimestamp() - t0;
            }

            return;
        }

        if (Simd.UseAvx2 && (seq & 3) == 0 && wd.Type == GgmlTensorType.Q2_0C &&
            wd.Packed2 != null && (hidden & 7) == 0)
        {
            long t2 = ProfileEnabled ? Stopwatch.GetTimestamp() : 0;
            BlockQ8Kx4* q8 = Q2PrefillBuffer(ffn, seq);
            MulMatQ2.SiluQuantizeAndGemm(gate, up, q8, ffn, seq, _pool,
                new Q2PanelWeight(wd.Packed2, down, hidden));
            if (ProfileEnabled) TicksQ2 += Stopwatch.GetTimestamp() - t2;
            return;
        }

        long tSilu = ProfileEnabled ? Stopwatch.GetTimestamp() : 0;
        Ops.SiLUMul(gate, up, seq * ffn, _pool);
        if (ProfileEnabled)
            TicksSilu += Stopwatch.GetTimestamp() - tSilu;
        Linear(gate, $"blk.{layer}.ffn_down.weight", down, ffn, hidden, seq);
    }

    private BlockQ8Kx4* Q2PrefillBuffer(int nIn, int tokens)
    {
        int q8Blocks = nIn / Qk.SuperBlock;
        return (BlockQ8Kx4*)_gemmScratch.E((nuint)((long)(tokens / 4) * q8Blocks * Qk.Q8Kx4Size));
    }

    private BlockQ8Kx4* STQPrefillBuffer(int nIn, int tokens)
    {
        int q8Blocks = nIn / STQ1_0.BlockLength;
        return (BlockQ8Kx4*)_gemmScratch.E((nuint)((long)(tokens / 4) * q8Blocks * Qk.Q8Kx4Size));
    }

    private BlockQ8_0x4* Q8PrefillBuffer(int nIn, int tokens)
    {
        int blocks = nIn / Qk.Q8_0Block;
        return (BlockQ8_0x4*)_gemmScratch.E((nuint)((long)(tokens / 4) * blocks * Qk.Q8_0x4Size));
    }
}
