using System.Runtime.InteropServices;
using Sdcb.HyMT2Sharp.Kernels;

namespace Sdcb.HyMT2Sharp.Tests;

public sealed class Q4KKernelTests
{
    [Fact]
    public unsafe void Q2_0C_DotAvxMatchesScalar()
    {
        if (!System.Runtime.Intrinsics.X86.Avx2.IsSupported)
            return;
        const int n = 1024;
        using NativeBuffer w = new((nuint)(2 * sizeof(BlockQ2_0C)));
        using NativeBuffer x = new((nuint)(4 * sizeof(BlockQ8K)));
        Random r = new(7);
        BlockQ2_0C* q = (BlockQ2_0C*)w.Pointer;
        BlockQ8K* a = (BlockQ8K*)x.Pointer;
        for (int b = 0; b < 2; b++) { q[b].D = HalfBits.FromSingle(0.125f); for (int j = 0; j < 128; j++) q[b].Qs[j] = (byte)r.Next(256); }
        for (int b = 0; b < 4; b++) { a[b].D = 0.02f; for (int j = 0; j < 256; j++) a[b].Qs[j] = (sbyte)r.Next(-127, 128); }
        float s = Q2_0C.DotScalarForTest(q, a, n);
        float v = Q2_0C.DotAvx2(q, a, n);
        using NativeBuffer e = new((nuint)n);
        Q2_0C.Expand(q, (sbyte*)e.Pointer, n);
        float z = Q2_0C.DotExpanded(q, (sbyte*)e.Pointer, a, n);
        Assert.True(MathF.Abs(s - v) < 1e-3f * n, $"scalar {s} avx {v}");
        Assert.True(MathF.Abs(s - z) < 1e-3f * n, $"scalar {s} expanded {z}");
    }

    [Fact]
    public unsafe void Q2_PackedPanelMatchesRows()
    {
        const int nIn = 1024;
        const int nOut = 16;
        const int tokens = 4;
        Assert.Equal(1056, sizeof(BlockQ2x8));
        Random rng = new(31);
        using NativeBuffer raw = new((nuint)((long)nOut * (nIn / Q2_0C.BlockLength) * sizeof(BlockQ2_0C)));
        using NativeBuffer panel = new((nuint)((long)(nOut / 8) * (nIn / Q2_0C.BlockLength) * sizeof(BlockQ2x8)));
        using NativeBuffer input = new((nuint)((long)tokens * nIn * sizeof(float)));
        using NativeBuffer output = new((nuint)((long)tokens * nOut * sizeof(float)));
        float* ip = (float*)input.Pointer;
        for (int i = 0; i < tokens * nIn; i++) ip[i] = (float)(rng.NextDouble() * 2 - 1);
        BlockQ2_0C* rows = (BlockQ2_0C*)raw.Pointer;
        for (int r = 0; r < nOut; r++)
        {
            for (int b = 0; b < nIn / Q2_0C.BlockLength; b++)
            {
                rows[r * 2 + b].D = HalfBits.FromSingle(0.01f + r * 0.001f + b * 0.003f);
                for (int j = 0; j < 128; j++) rows[r * 2 + b].Qs[j] = (byte)rng.Next(256);
            }
        }
        RepackQ2.Rows(rows, (BlockQ2x8*)panel.Pointer, nIn, nOut);
        using ScratchArena scratch = new();
        MulMatQ2.Gemm((BlockQ2x8*)panel.Pointer, rows, ip, (float*)output.Pointer, nIn, nOut, tokens, null, scratch);
        using NativeBuffer dw = new((nuint)(nIn * sizeof(float)));
        for (int t = 0; t < tokens; t++)
            for (int r = 0; r < nOut; r++)
            {
                // Portable dispatch runs an f32 dequant GEMM; AVX2 runs the
                // Q8-quantized packed panel.
                float tol = Simd.UsePanels ? 1e-4f : 1e-3f;
                float expected;
                if (Simd.UsePanels)
                {
                    using NativeBuffer rowQ8 = new((nuint)Q8K.RowBytes(nIn));
                    Q8K.QuantizeRow(ip + t * nIn, (BlockQ8K*)rowQ8.Pointer, nIn);
                    expected = Q2_0C.DotScalarForTest(rows + r * 2, (BlockQ8K*)rowQ8.Pointer, nIn);
                }
                else
                {
                    Q2_0C.DequantizeRow(rows + r * 2, (float*)dw.Pointer, nIn);
                    float* dwp = (float*)dw.Pointer;
                    expected = 0;
                    for (int i = 0; i < nIn; i++)
                        expected += dwp[i] * ip[t * nIn + i];
                }
                Assert.True(MathF.Abs(expected - ((float*)output.Pointer)[t * nOut + r]) < tol, $"t={t} r={r} exp={expected} got={((float*)output.Pointer)[t * nOut + r]}");
            }
    }

    [Fact]
    public unsafe void Q2_PackedPanelDoesNotOverflowInt16()
    {
        const int nIn = 512;
        const int nOut = 8;
        const int tokens = 4;
        using NativeBuffer raw = new((nuint)(nOut * sizeof(BlockQ2_0C)));
        using NativeBuffer panel = new((nuint)sizeof(BlockQ2x8));
        using NativeBuffer input = new((nuint)(tokens * nIn * sizeof(float)));
        using NativeBuffer output = new((nuint)(tokens * nOut * sizeof(float)));
        for (int i = 0; i < tokens * nIn; i++) ((float*)input.Pointer)[i] = 1;
        for (int r = 0; r < nOut; r++)
        {
            ((BlockQ2_0C*)raw.Pointer)[r].D = HalfBits.FromSingle(1);
            for (int j = 0; j < 128; j++) ((BlockQ2_0C*)raw.Pointer)[r].Qs[j] = 0xFF;
        }
        RepackQ2.Rows((BlockQ2_0C*)raw.Pointer, (BlockQ2x8*)panel.Pointer, nIn, nOut);
        using ScratchArena scratch = new();
        MulMatQ2.Gemm((BlockQ2x8*)panel.Pointer, (BlockQ2_0C*)raw.Pointer, (float*)input.Pointer, (float*)output.Pointer, nIn, nOut, tokens, null, scratch);
        for (int i = 0; i < tokens * nOut; i++) Assert.InRange(((float*)output.Pointer)[i], 1535.9f, 1536.1f);
    }

    [Fact]
    public unsafe void Q2_GemmHandlesTokenAndColumnTails()
    {
        const int nIn = 512, nOut = 10, tokens = 3;
        using NativeBuffer raw = new((nuint)(nOut * sizeof(BlockQ2_0C)));
        using NativeBuffer panel = new((nuint)sizeof(BlockQ2x8));
        using NativeBuffer input = new((nuint)(tokens * nIn * sizeof(float)));
        using NativeBuffer output = new((nuint)(tokens * nOut * sizeof(float)));
        Random rng = new(77);
        for (int i = 0; i < tokens * nIn; i++) ((float*)input.Pointer)[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int r = 0; r < nOut; r++)
        {
            BlockQ2_0C* row = (BlockQ2_0C*)raw.Pointer + r;
            row->D = HalfBits.FromSingle(0.01f + r * 0.002f);
            for (int j = 0; j < 128; j++) row->Qs[j] = (byte)rng.Next(256);
        }
        RepackQ2.Rows((BlockQ2_0C*)raw.Pointer, (BlockQ2x8*)panel.Pointer, nIn, 8);
        using ScratchArena scratch = new();
        MulMatQ2.Gemm((BlockQ2x8*)panel.Pointer, (BlockQ2_0C*)raw.Pointer, (float*)input.Pointer, (float*)output.Pointer, nIn, nOut, tokens, null, scratch);
        using NativeBuffer dw = new((nuint)(nIn * sizeof(float)));
        float* ip2 = (float*)input.Pointer;
        float tol2 = Simd.UsePanels ? 1e-4f : 1e-3f;
        for (int t = 0; t < tokens; t++)
        {
            for (int r = 0; r < nOut; r++)
            {
                float expected;
                if (Simd.UsePanels)
                {
                    using NativeBuffer q8 = new((nuint)Q8K.RowBytes(nIn));
                    Q8K.QuantizeRow(ip2 + t * nIn, (BlockQ8K*)q8.Pointer, nIn);
                    expected = Q2_0C.DotScalarForTest((BlockQ2_0C*)raw.Pointer + r, (BlockQ8K*)q8.Pointer, nIn);
                }
                else
                {
                    Q2_0C.DequantizeRow((BlockQ2_0C*)raw.Pointer + r, (float*)dw.Pointer, nIn);
                    float* dwp = (float*)dw.Pointer;
                    expected = 0;
                    for (int i = 0; i < nIn; i++)
                        expected += dwp[i] * ip2[t * nIn + i];
                }
                Assert.True(MathF.Abs(expected - ((float*)output.Pointer)[t * nOut + r]) < tol2);
            }
        }
    }

    [Fact]
    public unsafe void Q2_QuantizeAndGemmMatchesRows()
    {
        if (!Simd.UsePanels)
            return;
        const int nIn = 1024, nOut = 8, tokens = 4;
        Random rng = new(91);
        using NativeBuffer raw0 = new((nuint)(nOut * (nIn / Q2_0C.BlockLength) * sizeof(BlockQ2_0C)));
        using NativeBuffer raw1 = new((nuint)(nOut * (nIn / Q2_0C.BlockLength) * sizeof(BlockQ2_0C)));
        using NativeBuffer panel0 = new((nuint)((nIn / Q2_0C.BlockLength) * sizeof(BlockQ2x8)));
        using NativeBuffer panel1 = new((nuint)((nIn / Q2_0C.BlockLength) * sizeof(BlockQ2x8)));
        using NativeBuffer input = new((nuint)((long)tokens * nIn * sizeof(float)));
        using NativeBuffer output0 = new((nuint)((long)tokens * nOut * sizeof(float)));
        using NativeBuffer output1 = new((nuint)((long)tokens * nOut * sizeof(float)));
        using NativeBuffer q8 = new((nuint)((tokens / 4) * (nIn / Qk.SuperBlock) * Qk.Q8Kx4Size));
        float* ip = (float*)input.Pointer;
        for (int i = 0; i < tokens * nIn; i++) ip[i] = (float)(rng.NextDouble() * 2 - 1);
        BlockQ2_0C* rows0 = (BlockQ2_0C*)raw0.Pointer;
        BlockQ2_0C* rows1 = (BlockQ2_0C*)raw1.Pointer;
        for (int r = 0; r < nOut; r++)
            for (int b = 0; b < nIn / Q2_0C.BlockLength; b++)
            {
                rows0[r * 2 + b].D = HalfBits.FromSingle(0.01f + r * 0.001f + b * 0.002f);
                rows1[r * 2 + b].D = HalfBits.FromSingle(0.02f + r * 0.001f + b * 0.003f);
                for (int j = 0; j < 128; j++)
                {
                    rows0[r * 2 + b].Qs[j] = (byte)rng.Next(256);
                    rows1[r * 2 + b].Qs[j] = (byte)rng.Next(256);
                }
            }
        RepackQ2.Rows(rows0, (BlockQ2x8*)panel0.Pointer, nIn, nOut);
        RepackQ2.Rows(rows1, (BlockQ2x8*)panel1.Pointer, nIn, nOut);
        MulMatQ2.QuantizeAndGemm(ip, (BlockQ8Kx4*)q8.Pointer, nIn, tokens, null,
            new Q2PanelWeight((BlockQ2x8*)panel0.Pointer, (float*)output0.Pointer, nOut),
            new Q2PanelWeight((BlockQ2x8*)panel1.Pointer, (float*)output1.Pointer, nOut));

        for (int t = 0; t < tokens; t++)
        {
            using NativeBuffer rowQ8 = new((nuint)Q8K.RowBytes(nIn));
            Q8K.QuantizeRow(ip + t * nIn, (BlockQ8K*)rowQ8.Pointer, nIn);
            for (int r = 0; r < nOut; r++)
            {
                float expected0 = Q2_0C.DotScalarForTest(rows0 + r * 2, (BlockQ8K*)rowQ8.Pointer, nIn);
                float expected1 = Q2_0C.DotScalarForTest(rows1 + r * 2, (BlockQ8K*)rowQ8.Pointer, nIn);
                Assert.True(MathF.Abs(expected0 - ((float*)output0.Pointer)[t * nOut + r]) < 1e-4f);
                Assert.True(MathF.Abs(expected1 - ((float*)output1.Pointer)[t * nOut + r]) < 1e-4f);
            }
        }
    }

    [Fact]
    public unsafe void BlockSizes_MatchGgml()
    {
        Assert.Equal(144, sizeof(BlockQ4K));
        Assert.Equal(176, sizeof(BlockQ5K));
        Assert.Equal(210, sizeof(BlockQ6K));
        Assert.Equal(292, sizeof(BlockQ8K));
        Assert.Equal(1152, sizeof(BlockQ4Kx8));
        Assert.Equal(1168, sizeof(BlockQ8Kx4));
        Assert.Equal(2848, sizeof(BlockQ6Kx8));
        Assert.Equal(Qk.Q2_0CSize, sizeof(BlockQ2_0C));
        Assert.Equal(Qk.Q2x8Size, sizeof(BlockQ2x8));
        Assert.Equal(Qk.Q6Kx8Size, sizeof(BlockQ6Kx8));
        Assert.Equal(448, sizeof(BlockQ4Kx8Meta));
        Assert.Equal(Qk.Q4Kx8MetaSize, sizeof(BlockQ4Kx8Meta));
        Assert.Equal(Qk.Q4KSize, sizeof(BlockQ4K));
        Assert.Equal(Qk.Q8KSize, sizeof(BlockQ8K));
    }

    [Fact]
    public unsafe void VecDot_MatchesDequantDot()
    {
        const int nIn = 256 * 4;
        float[] weights = RandomRow(nIn, 1);
        float[] input = RandomRow(nIn, 2);
        using NativeBuffer wBuf = new((nuint)(sizeof(BlockQ4K)));
        // one superblock-group of 4, actually nIn/256 blocks
        using NativeBuffer packed = new((nuint)((nIn / Qk.SuperBlock) * Qk.Q4KSize));
        using NativeBuffer q8 = new((nuint)Q8K.RowBytes(nIn));
        using NativeBuffer dequant = new((nuint)(nIn * sizeof(float)));

        fixed (float* w = weights, x = input)
        {
            Q4K.PackSimple(w, (BlockQ4K*)packed.Pointer, nIn);
            Q8K.QuantizeRow(x, (BlockQ8K*)q8.Pointer, nIn);
            Q4K.DequantizeRow((BlockQ4K*)packed.Pointer, (float*)dequant.Pointer, nIn);

            float expected = 0;
            float* dw = (float*)dequant.Pointer;
            BlockQ8K* y = (BlockQ8K*)q8.Pointer;
            for (int i = 0; i < nIn; i++)
            {
                int block = i / Qk.SuperBlock;
                int off = i % Qk.SuperBlock;
                expected += dw[i] * (y[block].Qs[off] * y[block].D);
            }

            float scalar = VecDotQ4K.DotScalar((BlockQ4K*)packed.Pointer, y, nIn);
            Assert.True(MathF.Abs(scalar - expected) < 1e-2f * nIn, $"scalar {scalar} vs {expected}");

            if (System.Runtime.Intrinsics.X86.Avx2.IsSupported)
            {
                float avx = VecDotQ4K.DotAvx2((BlockQ4K*)packed.Pointer, y, nIn);
                Assert.True(MathF.Abs(avx - scalar) < 1e-3f * nIn, $"avx {avx} vs scalar {scalar}");
            }
        }
    }

    [Fact]
    public unsafe void PanelGemm_MatchesPerRowVecDot()
    {
        const int nIn = 256 * 3;
        const int nOut = 16;
        const int tokens = 4;
        float[] w = RandomRow(nOut * nIn, 3);
        float[] a = RandomRow(tokens * nIn, 4);

        using NativeBuffer q4 = new((nuint)((long)nOut * (nIn / Qk.SuperBlock) * Qk.Q4KSize));
        using NativeBuffer q4x8 = new((nuint)((long)(nOut / 8) * (nIn / Qk.SuperBlock) * Qk.Q4Kx8Size));
        using NativeBuffer q8x4 = new((nuint)((nIn / Qk.SuperBlock) * Qk.Q8Kx4Size));
        using NativeBuffer expected = new((nuint)((long)tokens * nOut * sizeof(float)));
        using NativeBuffer got = new((nuint)((long)tokens * nOut * sizeof(float)));
        using NativeBuffer q8 = new((nuint)Q8K.RowBytes(nIn));
        using NativeBuffer meta = new((nuint)((long)(nOut / 8) * (nIn / Qk.SuperBlock) * Qk.Q4Kx8MetaSize));

        fixed (float* wp = w, ap = a)
        {
            BlockQ4K* rows = (BlockQ4K*)q4.Pointer;
            int nb = nIn / Qk.SuperBlock;
            for (int r = 0; r < nOut; r++)
                Q4K.PackSimple(wp + r * nIn, rows + r * nb, nIn);
            RepackQ4K.Rows(rows, (BlockQ4Kx8*)q4x8.Pointer, nIn, nOut);
            RepackQ4K.BuildMeta(rows, (BlockQ4Kx8Meta*)meta.Pointer, nIn, nOut);
            QuantizeQ8Kx4.Quantize4x8(ap, (BlockQ8Kx4*)q8x4.Pointer, nIn);

            float* exp = (float*)expected.Pointer;
            for (int t = 0; t < tokens; t++)
            {
                Q8K.QuantizeRow(ap + t * nIn, (BlockQ8K*)q8.Pointer, nIn);
                for (int r = 0; r < nOut; r++)
                    exp[t * nOut + r] = VecDotQ4K.DotScalar(rows + r * nb, (BlockQ8K*)q8.Pointer, nIn);
            }

            GemmQ4K.GemmScalar(nIn, (float*)got.Pointer, nOut, (BlockQ4Kx8*)q4x8.Pointer, (BlockQ8Kx4*)q8x4.Pointer, tokens, nOut);
            AssertClose((float*)expected.Pointer, (float*)got.Pointer, tokens * nOut, 2e-2f);

            if (System.Runtime.Intrinsics.X86.Avx2.IsSupported)
            {
                NativeMemory.Clear(got.Pointer, got.Bytes);
                GemmQ4K.GemmAvx2(nIn, (float*)got.Pointer, nOut, (BlockQ4Kx8*)q4x8.Pointer, (BlockQ8Kx4*)q8x4.Pointer, tokens, nOut, (BlockQ4Kx8Meta*)meta.Pointer);
                AssertClose((float*)expected.Pointer, (float*)got.Pointer, tokens * nOut, 2e-2f);
            }
        }
    }

    [Fact]
    public unsafe void MulMatGemv_MatchesPanelForOneToken()
    {
        const int nIn = 256 * 2;
        const int nOut = 8;
        float[] w = RandomRow(nOut * nIn, 5);
        float[] a = RandomRow(nIn, 6);
        using NativeBuffer q4 = new((nuint)((long)nOut * (nIn / Qk.SuperBlock) * Qk.Q4KSize));
        using NativeBuffer q4x8 = new((nuint)((nIn / Qk.SuperBlock) * Qk.Q4Kx8Size));
        using NativeBuffer out0 = new((nuint)(nOut * sizeof(float)));
        using NativeBuffer out1 = new((nuint)(nOut * sizeof(float)));

        fixed (float* wp = w, ap = a)
        {
            BlockQ4K* rows = (BlockQ4K*)q4.Pointer;
            int nb = nIn / Qk.SuperBlock;
            for (int r = 0; r < nOut; r++)
                Q4K.PackSimple(wp + r * nIn, rows + r * nb, nIn);
            RepackQ4K.Rows(rows, (BlockQ4Kx8*)q4x8.Pointer, nIn, nOut);
            MulMatQ4K.Gemv(rows, ap, (float*)out0.Pointer, nIn, nOut);
            MulMatQ4K.Gemm((BlockQ4Kx8*)q4x8.Pointer, rows, ap, (float*)out1.Pointer, nIn, nOut, 1);
            AssertClose((float*)out0.Pointer, (float*)out1.Pointer, nOut, 1e-3f);
        }
    }

    [Fact]
    public unsafe void Q6K_Avx2MatchesScalar()
    {
        if (!System.Runtime.Intrinsics.X86.Avx2.IsSupported)
            return;

        const int nIn = 256 * 3;
        Random rng = new(9);
        using NativeBuffer q6 = new((nuint)((nIn / Qk.SuperBlock) * Qk.Q6KSize));
        using NativeBuffer q8 = new((nuint)Q8K.RowBytes(nIn));
        using NativeBuffer f32 = new((nuint)(nIn * sizeof(float)));
        byte* raw = (byte*)q6.Pointer;
        for (int i = 0; i < (int)q6.Bytes; i++)
            raw[i] = (byte)rng.Next(256);
        float[] input = RandomRow(nIn, 11);
        fixed (float* x = input)
        {
            Q8K.QuantizeRow(x, (BlockQ8K*)q8.Pointer, nIn);
            float scalar = Q6K.DotScalar((BlockQ6K*)q6.Pointer, (BlockQ8K*)q8.Pointer, nIn);
            float avx = Q6K.DotAvx2((BlockQ6K*)q6.Pointer, (BlockQ8K*)q8.Pointer, nIn);
            Assert.True(MathF.Abs(avx - scalar) < 1e-2f * nIn, $"avx {avx} vs scalar {scalar}");

            Q6K.DequantizeRow((BlockQ6K*)q6.Pointer, (float*)f32.Pointer, nIn);
            float expected = 0;
            float* dw = (float*)f32.Pointer;
            BlockQ8K* y = (BlockQ8K*)q8.Pointer;
            for (int i = 0; i < nIn; i++)
            {
                int block = i / Qk.SuperBlock;
                int off = i % Qk.SuperBlock;
                expected += dw[i] * (y[block].Qs[off] * y[block].D);
            }

            Assert.True(MathF.Abs(scalar - expected) < 1e-2f * nIn, $"scalar {scalar} vs dequant {expected}");

            float[] in0 = RandomRow(nIn, 21);
            float[] in1 = RandomRow(nIn, 22);
            float[] in2 = RandomRow(nIn, 23);
            float[] in3 = RandomRow(nIn, 24);
            using NativeBuffer y0 = new((nuint)Q8K.RowBytes(nIn));
            using NativeBuffer y1 = new((nuint)Q8K.RowBytes(nIn));
            using NativeBuffer y2 = new((nuint)Q8K.RowBytes(nIn));
            using NativeBuffer y3 = new((nuint)Q8K.RowBytes(nIn));
            float* got4 = stackalloc float[4];
            fixed (float* a0 = in0, a1 = in1, a2 = in2, a3 = in3)
            {
                Q8K.QuantizeRow(a0, (BlockQ8K*)y0.Pointer, nIn);
                Q8K.QuantizeRow(a1, (BlockQ8K*)y1.Pointer, nIn);
                Q8K.QuantizeRow(a2, (BlockQ8K*)y2.Pointer, nIn);
                Q8K.QuantizeRow(a3, (BlockQ8K*)y3.Pointer, nIn);
                Q6K.DotAvx2x4(
                    (BlockQ6K*)q6.Pointer,
                    (BlockQ8K*)y0.Pointer,
                    (BlockQ8K*)y1.Pointer,
                    (BlockQ8K*)y2.Pointer,
                    (BlockQ8K*)y3.Pointer,
                    nIn,
                    got4);
                Assert.True(MathF.Abs(got4[0] - Q6K.DotAvx2((BlockQ6K*)q6.Pointer, (BlockQ8K*)y0.Pointer, nIn)) < 1e-3f * nIn);
                Assert.True(MathF.Abs(got4[1] - Q6K.DotAvx2((BlockQ6K*)q6.Pointer, (BlockQ8K*)y1.Pointer, nIn)) < 1e-3f * nIn);
                Assert.True(MathF.Abs(got4[2] - Q6K.DotAvx2((BlockQ6K*)q6.Pointer, (BlockQ8K*)y2.Pointer, nIn)) < 1e-3f * nIn);
                Assert.True(MathF.Abs(got4[3] - Q6K.DotAvx2((BlockQ6K*)q6.Pointer, (BlockQ8K*)y3.Pointer, nIn)) < 1e-3f * nIn);
            }
        }
    }

    private static float[] RandomRow(int n, int seed)
    {
        Random rng = new(seed);
        float[] data = new float[n];
        for (int i = 0; i < n; i++)
            data[i] = (float)(rng.NextDouble() * 2 - 1);
        return data;
    }

    private static unsafe void AssertClose(float* a, float* b, int n, float rel)
    {
        float maxAbs = 0;
        float maxRel = 0;
        for (int i = 0; i < n; i++)
        {
            float d = MathF.Abs(a[i] - b[i]);
            if (d > maxAbs) maxAbs = d;
            float scale = MathF.Max(1e-3f, MathF.Max(MathF.Abs(a[i]), MathF.Abs(b[i])));
            float r = d / scale;
            if (r > maxRel) maxRel = r;
        }

        Assert.True(maxRel < rel, $"maxRel={maxRel} maxAbs={maxAbs}");
    }
}
