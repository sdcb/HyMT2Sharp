using System.Runtime.InteropServices;
using Sdcb.HyMT2Sharp.Kernels;

namespace Sdcb.HyMT2Sharp.Tests;

public sealed class STQ1_0Tests
{
    [Fact]
    public unsafe void BlockSizeAndStrideLayoutMatchSherry()
    {
        Assert.Equal(42, sizeof(BlockSTQ1_0));
        Assert.Equal(42, Qk.STQ1_0Size);

        using NativeBuffer w = new((nuint)sizeof(BlockSTQ1_0));
        using NativeBuffer outBuf = new((nuint)(STQ1_0.BlockLength * sizeof(float)));
        BlockSTQ1_0* block = (BlockSTQ1_0*)w.Pointer;
        block->D = HalfBits.FromSingle(2);
        // slot 0/sign 0 is (0,+1,+1,+1).  It should land at offsets
        // g, g+16, g+32, g+48 for the first 64-value chunk.
        block->Qs[0] = 0;
        STQ1_0.DequantizeRow(block, (float*)outBuf.Pointer, STQ1_0.BlockLength);
        float* dst = (float*)outBuf.Pointer;
        Assert.Equal(0, dst[0]);
        Assert.Equal(2, dst[16]);
        Assert.Equal(2, dst[32]);
        Assert.Equal(2, dst[48]);
    }

    [Fact]
    public unsafe void DotMatchesDequantizedReference()
    {
        const int n = 512;
        Random rng = new(123);
        using NativeBuffer w = new((nuint)(n / STQ1_0.BlockLength * sizeof(BlockSTQ1_0)));
        using NativeBuffer x = new((nuint)Q8K.RowBytes(n));
        using NativeBuffer dequant = new((nuint)(n * sizeof(float)));
        byte* raw = (byte*)w.Pointer;
        for (int i = 0; i < (int)w.Bytes; i++)
            raw[i] = (byte)rng.Next(256);
        BlockSTQ1_0* weights = (BlockSTQ1_0*)w.Pointer;
        for (int b = 0; b < n / STQ1_0.BlockLength; b++)
            weights[b].D = HalfBits.FromSingle(0.05f + b * 0.01f);
        BlockQ8K* q8 = (BlockQ8K*)x.Pointer;
        q8[0].D = 0.02f;
        q8[1].D = 0.03f;
        for (int b = 0; b < n / Qk.SuperBlock; b++)
        {
            for (int i = 0; i < Qk.SuperBlock; i++)
                q8[b].Qs[i] = (sbyte)rng.Next(-127, 128);
            int sum = 0;
            for (int i = 0; i < Qk.SuperBlock; i++) sum += q8[b].Qs[i];
            for (int i = 0; i < Qk.SuperBlock / 16; i++)
            {
                int s = 0;
                for (int j = 0; j < 16; j++) s += q8[b].Qs[i * 16 + j];
                q8[b].Bsums[i] = (short)s;
            }
        }

        STQ1_0.DequantizeRow(weights, (float*)dequant.Pointer, n);
        float expected = 0;
        float* dw = (float*)dequant.Pointer;
        for (int i = 0; i < n; i++)
            expected += dw[i] * q8[i / Qk.SuperBlock].Qs[i % Qk.SuperBlock] * q8[i / Qk.SuperBlock].D;

        float scalar = STQ1_0.DotScalar(weights, q8, n);
        float dot = STQ1_0.Dot(weights, q8, n);
        Assert.True(MathF.Abs(scalar - expected) < 1e-3f * n, $"scalar={scalar} expected={expected}");
        Assert.True(MathF.Abs(dot - scalar) < 1e-3f * n, $"dot={dot} scalar={scalar}");
    }

    [Fact]
    public unsafe void GemmMatchesIndependentRows()
    {
        const int nIn = 512, nOut = 11, tokens = 3;
        Random rng = new(456);
        using NativeBuffer w = new((nuint)((long)nOut * (nIn / STQ1_0.BlockLength) * sizeof(BlockSTQ1_0)));
        using NativeBuffer input = new((nuint)((long)tokens * nIn * sizeof(float)));
        using NativeBuffer output = new((nuint)((long)tokens * nOut * sizeof(float)));
        float* ip = (float*)input.Pointer;
        for (int i = 0; i < tokens * nIn; i++) ip[i] = (float)(rng.NextDouble() * 2 - 1);
        BlockSTQ1_0* rows = (BlockSTQ1_0*)w.Pointer;
        for (int i = 0; i < (int)w.Bytes; i++) ((byte*)w.Pointer)[i] = (byte)rng.Next(256);
        for (int r = 0; r < nOut; r++)
            for (int b = 0; b < nIn / STQ1_0.BlockLength; b++)
                rows[r * (nIn / STQ1_0.BlockLength) + b].D = HalfBits.FromSingle(0.02f + r * 0.001f);

        using ScratchArena scratch = new();
        MulMatSTQ.Gemm(rows, ip, (float*)output.Pointer, nIn, nOut, tokens, null, scratch);
        // Portable GEMM dequantizes weights and consumes raw f32 input.
        using NativeBuffer dw = new((nuint)(nIn * sizeof(float)));
        float* op = (float*)output.Pointer;
        int nb = nIn / STQ1_0.BlockLength;
        for (int t = 0; t < tokens; t++)
        {
            for (int r = 0; r < nOut; r++)
            {
                STQ1_0.DequantizeRow(rows + r * nb, (float*)dw.Pointer, nIn);
                float expected = 0;
                float* dwp = (float*)dw.Pointer;
                for (int i = 0; i < nIn; i++)
                    expected += dwp[i] * ip[t * nIn + i];
                Assert.True(MathF.Abs(expected - op[t * nOut + r]) < 1e-3f,
                    $"t={t} r={r} expected={expected} got={op[t * nOut + r]}");
            }
        }
    }

    [Fact]
    public unsafe void PackedPanelMatchesScalarGemvAndGemm()
    {
        if (!Simd.UsePanels)
            return;
        const int nIn = 512, nOut = 16, tokens = 4;
        Random rng = new(789);
        int nb = nIn / STQ1_0.BlockLength;
        using NativeBuffer rawBuf = new((nuint)((long)nOut * nb * sizeof(BlockSTQ1_0)));
        using NativeBuffer panelBuf = new((nuint)((long)(nOut / 8) * nb * Qk.STQ1_0x8Size));
        using NativeBuffer inputBuf = new((nuint)((long)tokens * nIn * sizeof(float)));
        using NativeBuffer outputBuf = new((nuint)((long)tokens * nOut * sizeof(float)));
        using NativeBuffer refBuf = new((nuint)((long)tokens * nOut * sizeof(float)));
        using ScratchArena scratch = new();
        BlockSTQ1_0* raw = (BlockSTQ1_0*)rawBuf.Pointer;
        for (int i = 0; i < (int)rawBuf.Bytes; i++) ((byte*)rawBuf.Pointer)[i] = (byte)rng.Next(256);
        for (int r = 0; r < nOut; r++)
            for (int b = 0; b < nb; b++)
                raw[r * nb + b].D = HalfBits.FromSingle(0.01f + 0.002f * r + 0.001f * b);
        RepackSTQ.Rows(raw, (BlockSTQ1_0x8*)panelBuf.Pointer, nIn, nOut);

        float* input = (float*)inputBuf.Pointer;
        for (int i = 0; i < tokens * nIn; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);
        BlockQ8K* q8 = (BlockQ8K*)scratch.D((nuint)Q8K.RowBytes(nIn));
        Q8K.QuantizeRow(input, q8, nIn);
        float* panelGemv = (float*)refBuf.Pointer;
        for (int g = 0; g < nOut / 8; g++)
            STQPanel.Gemv((BlockSTQ1_0x8*)panelBuf.Pointer + g * nb, q8, panelGemv + g * 8, nIn);
        for (int r = 0; r < nOut; r++)
            Assert.True(MathF.Abs(panelGemv[r] - STQ1_0.DotScalar(raw + r * nb, q8, nIn)) < 1e-4f,
                $"gemv r={r} got={panelGemv[r]}");

        BlockQ8Kx4* q84 = (BlockQ8Kx4*)scratch.E((nuint)(nb * Qk.Q8Kx4Size));
        MulMatSTQ.QuantizeAndGemm(input, q84, nIn, tokens, null,
            new STQPanelWeight((BlockSTQ1_0x8*)panelBuf.Pointer, (float*)outputBuf.Pointer, nOut));
        float* output = (float*)outputBuf.Pointer;
        float* expected = (float*)refBuf.Pointer;
        for (int t = 0; t < tokens; t++)
        {
            Q8K.QuantizeRow(input + t * nIn, q8, nIn);
            for (int r = 0; r < nOut; r++)
                expected[t * nOut + r] = STQ1_0.DotScalar(raw + r * nb, q8, nIn);
        }
        for (int i = 0; i < tokens * nOut; i++)
            Assert.True(MathF.Abs(output[i] - expected[i]) < 1e-4f, $"gemm i={i} got={output[i]} expected={expected[i]}");
    }

    [Fact]
    public unsafe void PackedPanelParallelMatchesScalar()
    {
        if (!Simd.UsePanels)
            return;
        const int nIn = 1024, nOut = 64, tokens = 8;
        Random rng = new(901);
        int nb = nIn / STQ1_0.BlockLength;
        using NativeBuffer rawBuf = new((nuint)((long)nOut * nb * sizeof(BlockSTQ1_0)));
        using NativeBuffer panelBuf = new((nuint)((long)(nOut / 8) * nb * Qk.STQ1_0x8Size));
        using NativeBuffer inputBuf = new((nuint)((long)tokens * nIn * sizeof(float)));
        using NativeBuffer outputBuf = new((nuint)((long)tokens * nOut * sizeof(float)));
        using NativeBuffer expectedBuf = new((nuint)((long)tokens * nOut * sizeof(float)));
        using ScratchArena scratch = new();
        using CpuThreadPool pool = new(4);
        BlockSTQ1_0* raw = (BlockSTQ1_0*)rawBuf.Pointer;
        for (int i = 0; i < (int)rawBuf.Bytes; i++) ((byte*)rawBuf.Pointer)[i] = (byte)rng.Next(256);
        for (int r = 0; r < nOut; r++)
            for (int b = 0; b < nb; b++) raw[r * nb + b].D = HalfBits.FromSingle(0.02f + r * 0.0007f + b * 0.001f);
        RepackSTQ.Rows(raw, (BlockSTQ1_0x8*)panelBuf.Pointer, nIn, nOut);
        float* input = (float*)inputBuf.Pointer;
        for (int i = 0; i < tokens * nIn; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);
        BlockQ8K* q8 = (BlockQ8K*)scratch.D((nuint)Q8K.RowBytes(nIn));
        float* expected = (float*)expectedBuf.Pointer;
        for (int t = 0; t < tokens; t++)
        {
            Q8K.QuantizeRow(input + t * nIn, q8, nIn);
            for (int r = 0; r < nOut; r++) expected[t * nOut + r] = STQ1_0.DotScalar(raw + r * nb, q8, nIn);
        }
        BlockQ8Kx4* q84 = (BlockQ8Kx4*)scratch.E((nuint)((tokens / 4) * nb * Qk.Q8Kx4Size));
        MulMatSTQ.QuantizeAndGemm(input, q84, nIn, tokens, pool,
            new STQPanelWeight((BlockSTQ1_0x8*)panelBuf.Pointer, (float*)outputBuf.Pointer, nOut));
        float* output = (float*)outputBuf.Pointer;
        for (int i = 0; i < tokens * nOut; i++)
            Assert.True(MathF.Abs(output[i] - expected[i]) < 1e-4f, $"i={i} got={output[i]} expected={expected[i]}");
    }
}
