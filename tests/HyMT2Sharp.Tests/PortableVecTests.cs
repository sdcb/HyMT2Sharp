using System.Numerics;
using System.Runtime.InteropServices;
using Sdcb.HyMT2Sharp.Kernels;

namespace Sdcb.HyMT2Sharp.Tests;

/// <summary>
/// Vector&lt;T&gt; portable-tier tests. These call the *Vec implementations
/// directly, so they validate the fallback path on every host (default AVX2,
/// HYMT2SHARP_FORCE_PORTABLE, and DOTNET_EnableAVX=0).
/// </summary>
public sealed class PortableVecTests
{
    [Fact]
    public unsafe void DotI8_MatchesScalar()
    {
        const int n = 512;
        using NativeBuffer w = new(n);
        using NativeBuffer a = new(n);
        Random r = new(101);
        sbyte* wp = (sbyte*)w.Pointer;
        sbyte* ap = (sbyte*)a.Pointer;
        for (int i = 0; i < n; i++)
        {
            wp[i] = (sbyte)r.Next(-128, 128);
            ap[i] = (sbyte)r.Next(-128, 128);
        }

        int expected = 0;
        for (int i = 0; i < n; i++)
            expected += wp[i] * ap[i];
        Assert.Equal(expected, VecI8.DotI8(wp, ap, n));
    }

    [Fact]
    public unsafe void SumI8_And_Bsums16_MatchScalar()
    {
        const int n = 256;
        using NativeBuffer w = new(n);
        using NativeBuffer bs = new((nuint)(n / 16 * sizeof(short)));
        Random r = new(102);
        sbyte* wp = (sbyte*)w.Pointer;
        for (int i = 0; i < n; i++)
            wp[i] = (sbyte)r.Next(-128, 128);

        int expected = 0;
        for (int i = 0; i < n; i++)
            expected += wp[i];
        Assert.Equal(expected, VecI8.SumI8(wp, n));

        VecI8.Bsums16(wp, (short*)bs.Pointer, n);
        for (int g = 0; g < n / 16; g++)
        {
            int s = 0;
            for (int i = 0; i < 16; i++)
                s += wp[g * 16 + i];
            Assert.Equal((short)s, ((short*)bs.Pointer)[g]);
        }
    }

    [Fact]
    public unsafe void Dot16x2_MatchesScalar()
    {
        using NativeBuffer w = new(32);
        using NativeBuffer a = new(32);
        Random r = new(103);
        sbyte* wp = (sbyte*)w.Pointer;
        sbyte* ap = (sbyte*)a.Pointer;
        for (int i = 0; i < 32; i++)
        {
            wp[i] = (sbyte)r.Next(-128, 128);
            ap[i] = (sbyte)r.Next(-128, 128);
        }

        VecI8.Dot16x2(wp, ap, out int s0, out int s1);
        int e0 = 0, e1 = 0;
        for (int i = 0; i < 16; i++)
        {
            e0 += wp[i] * ap[i];
            e1 += wp[16 + i] * ap[16 + i];
        }

        Assert.Equal(e0, s0);
        Assert.Equal(e1, s1);
    }

    [Fact]
    public unsafe void DotNibblesI8_MatchesScalar()
    {
        using NativeBuffer p = new(32);
        using NativeBuffer a = new(64);
        Random r = new(104);
        byte* pp = (byte*)p.Pointer;
        sbyte* ap = (sbyte*)a.Pointer;
        for (int i = 0; i < 32; i++)
            pp[i] = (byte)r.Next(256);
        for (int i = 0; i < 64; i++)
            ap[i] = (sbyte)r.Next(-128, 128);

        VecI8.DotNibblesI8(pp, ap, ap + 32, out int lo, out int hi);
        int eLo = 0, eHi = 0;
        for (int i = 0; i < 32; i++)
        {
            eLo += (pp[i] & 0xF) * ap[i];
            eHi += (pp[i] >> 4) * ap[32 + i];
        }

        Assert.Equal(eLo, lo);
        Assert.Equal(eHi, hi);
    }

    [Fact]
    public unsafe void QuantizeStore_MatchesScalarRound()
    {
        const int n = 256;
        using NativeBuffer src = new((nuint)(n * sizeof(float)));
        using NativeBuffer qs = new(n);
        Random r = new(105);
        float* sp = (float*)src.Pointer;
        for (int i = 0; i < n; i++)
            sp[i] = (float)(r.NextDouble() * 4 - 2);
        const float id = 40.5f;
        VecF.QuantizeStore(sp, id, (sbyte*)qs.Pointer, n);
        for (int i = 0; i < n; i++)
        {
            int v = (int)MathF.Round(id * sp[i]);
            v = Math.Clamp(v, -127, 127);
            Assert.Equal((sbyte)v, ((sbyte*)qs.Pointer)[i]);
        }
    }

    [Fact]
    public unsafe void AbsMax_And_DequantI8_MatchScalar()
    {
        const int n = 257;
        using NativeBuffer src = new((nuint)(n * sizeof(float)));
        using NativeBuffer dst = new((nuint)(n * sizeof(float)));
        Random r = new(106);
        float* sp = (float*)src.Pointer;
        for (int i = 0; i < n; i++)
            sp[i] = (float)(r.NextDouble() * 200 - 100);
        sp[64] = -333.5f;
        Assert.Equal(333.5f, VecF.AbsMax(sp, n));

        using NativeBuffer qs = new(n);
        sbyte* qp = (sbyte*)qs.Pointer;
        for (int i = 0; i < n; i++)
            qp[i] = (sbyte)r.Next(-128, 128);
        VecF.DequantI8(qp, 0.031f, (float*)dst.Pointer, n);
        for (int i = 0; i < n; i++)
            Assert.Equal(0.031f * qp[i], ((float*)dst.Pointer)[i], 6);
    }

    [Fact]
    public unsafe void ExpVec_SiluVec_MatchScalar()
    {
        float[] xs = [-90f, -20f, -10f, -1.5f, -0.25f, 0f, 0.5f, 2.5f, 10f, 30f, 90f, 200f];
        int n = xs.Length;
        using NativeBuffer buf = new((nuint)(Math.Max(n, Vector<float>.Count) * sizeof(float)));
        float* p = (float*)buf.Pointer;
        for (int i = 0; i < n; i++)
            p[i] = xs[i];
        for (int i = n; i < Vector<float>.Count; i++)
            p[i] = 0;

        for (int i = 0; i < n; i += Vector<float>.Count)
        {
            int take = Math.Min(Vector<float>.Count, n - i);
            if (take < Vector<float>.Count)
                break;
            Vector<float> e = FastExp.ExpVec(VecF.Load(p + i));
            Vector<float> s = FastExp.SiluVec(VecF.Load(p + i));
            for (int l = 0; l < Vector<float>.Count; l++)
            {
                float x = p[i + l];
                float refE = MathF.Exp(x);
                if (float.IsInfinity(refE))
                    Assert.True(float.IsInfinity(e[l]) || e[l] > 1e30f, $"exp({x}) = {e[l]}");
                else
                    Assert.True(MathF.Abs(e[l] - refE) <= MathF.Max(1e-4f, refE * 1e-4f), $"exp({x}) = {e[l]} vs {refE}");

                float refS = x / (1f + MathF.Exp(-x));
                if (!float.IsInfinity(refS))
                    Assert.True(MathF.Abs(s[l] - refS) <= MathF.Max(1e-3f, MathF.Abs(refS) * 1e-3f), $"silu({x}) = {s[l]} vs {refS}");
            }
        }
    }

    [Fact]
    public unsafe void Q8_0_DotVec_MatchesScalar()
    {
        const int n = 512;
        int nb = n / Qk.Q8_0Block;
        using NativeBuffer w = new((nuint)(nb * sizeof(BlockQ8_0)));
        using NativeBuffer a = new((nuint)(nb * sizeof(BlockQ8_0Act)));
        Random r = new(107);
        BlockQ8_0* wp = (BlockQ8_0*)w.Pointer;
        BlockQ8_0Act* ap = (BlockQ8_0Act*)a.Pointer;
        for (int b = 0; b < nb; b++)
        {
            wp[b].D = HalfBits.FromSingle(0.01f + b * 0.001f);
            ap[b].D = 0.02f + b * 0.001f;
            for (int j = 0; j < Qk.Q8_0Block; j++)
            {
                wp[b].Qs[j] = (sbyte)r.Next(-128, 128);
                ap[b].Qs[j] = (sbyte)r.Next(-128, 128);
            }
        }

        float scalar = Q8_0.DotScalar(wp, ap, n);
        float vec = Q8_0.DotVec(wp, ap, n);
        Assert.True(MathF.Abs(scalar - vec) < 1e-3f * n, $"scalar {scalar} vs vec {vec}");
    }

    [Fact]
    public unsafe void Q4K_DotVec_MatchesScalar()
    {
        const int n = Qk.SuperBlock * 3;
        int nb = n / Qk.SuperBlock;
        using NativeBuffer w = new((nuint)(nb * sizeof(BlockQ4K)));
        using NativeBuffer a = new((nuint)(nb * sizeof(BlockQ8K)));
        FillQ4K((BlockQ4K*)w.Pointer, nb, 108);
        FillQ8K((BlockQ8K*)a.Pointer, nb, 109);
        float scalar = VecDotQ4K.DotScalar((BlockQ4K*)w.Pointer, (BlockQ8K*)a.Pointer, n);
        float vec = VecDotQ4K.DotVec((BlockQ4K*)w.Pointer, (BlockQ8K*)a.Pointer, n);
        Assert.True(MathF.Abs(scalar - vec) < 1e-3f * n, $"scalar {scalar} vs vec {vec}");
    }

    [Fact]
    public unsafe void Q5K_DotVec_MatchesScalar()
    {
        const int n = Qk.SuperBlock * 3;
        int nb = n / Qk.SuperBlock;
        using NativeBuffer w = new((nuint)(nb * sizeof(BlockQ5K)));
        using NativeBuffer a = new((nuint)(nb * sizeof(BlockQ8K)));
        Random r = new(110);
        BlockQ5K* wp = (BlockQ5K*)w.Pointer;
        for (int b = 0; b < nb; b++)
        {
            wp[b].D = HalfBits.FromSingle(0.01f + b * 0.001f);
            wp[b].Dmin = HalfBits.FromSingle(0.002f);
            for (int j = 0; j < Qk.ScaleBytes; j++)
                wp[b].Scales[j] = (byte)r.Next(256);
            for (int j = 0; j < Qk.SuperBlock / 8; j++)
                wp[b].Qh[j] = (byte)r.Next(256);
            for (int j = 0; j < Qk.SuperBlock / 2; j++)
                wp[b].Qs[j] = (byte)r.Next(256);
        }

        FillQ8K((BlockQ8K*)a.Pointer, nb, 111);
        float scalar = Q5K.DotScalar(wp, (BlockQ8K*)a.Pointer, n);
        float vec = Q5K.DotVec(wp, (BlockQ8K*)a.Pointer, n);
        Assert.True(MathF.Abs(scalar - vec) < 1e-3f * n, $"scalar {scalar} vs vec {vec}");
    }

    [Fact]
    public unsafe void Q6K_DotVec_MatchesScalar()
    {
        const int n = Qk.SuperBlock * 3;
        int nb = n / Qk.SuperBlock;
        using NativeBuffer w = new((nuint)(nb * sizeof(BlockQ6K)));
        using NativeBuffer a = new((nuint)(nb * sizeof(BlockQ8K)));
        Random r = new(112);
        BlockQ6K* wp = (BlockQ6K*)w.Pointer;
        for (int b = 0; b < nb; b++)
        {
            wp[b].D = HalfBits.FromSingle(0.01f + b * 0.001f);
            for (int j = 0; j < Qk.SuperBlock / 2; j++)
                wp[b].Ql[j] = (byte)r.Next(256);
            for (int j = 0; j < Qk.SuperBlock / 4; j++)
                wp[b].Qh[j] = (byte)r.Next(256);
            for (int j = 0; j < Qk.SuperBlock / 16; j++)
                wp[b].Scales[j] = (sbyte)r.Next(-64, 64);
        }

        FillQ8K((BlockQ8K*)a.Pointer, nb, 113);
        float scalar = Q6K.DotScalar(wp, (BlockQ8K*)a.Pointer, n);
        float vec = Q6K.DotVec(wp, (BlockQ8K*)a.Pointer, n);
        Assert.True(MathF.Abs(scalar - vec) < 1e-3f * n, $"scalar {scalar} vs vec {vec}");
    }

    [Fact]
    public unsafe void Q2_0C_DotVec_MatchesScalar()
    {
        const int n = 1024;
        using NativeBuffer w = new((nuint)(2 * sizeof(BlockQ2_0C)));
        using NativeBuffer a = new((nuint)(4 * sizeof(BlockQ8K)));
        Random r = new(114);
        BlockQ2_0C* wp = (BlockQ2_0C*)w.Pointer;
        BlockQ8K* ap = (BlockQ8K*)a.Pointer;
        for (int b = 0; b < 2; b++)
        {
            wp[b].D = HalfBits.FromSingle(0.125f);
            for (int j = 0; j < 128; j++)
                wp[b].Qs[j] = (byte)r.Next(256);
        }

        for (int b = 0; b < 4; b++)
        {
            ap[b].D = 0.02f;
            int bs = 0;
            for (int j = 0; j < 256; j++)
            {
                ap[b].Qs[j] = (sbyte)r.Next(-127, 128);
                bs += ap[b].Qs[j];
                if ((j & 15) == 15)
                {
                    ap[b].Bsums[j / 16] = (short)bs;
                    bs = 0;
                }
            }
        }

        float scalar = Q2_0C.DotScalarForTest(wp, ap, n);
        float vec = Q2_0C.DotVec(wp, ap, n);
        Assert.True(MathF.Abs(scalar - vec) < 1e-3f * n, $"scalar {scalar} vs vec {vec}");
    }

    [Fact]
    public unsafe void STQ_DotVec_MatchesScalar()
    {
        const int n = Qk.SuperBlock * 2;
        int nb = n / Qk.SuperBlock;
        using NativeBuffer w = new((nuint)(nb * sizeof(BlockSTQ1_0)));
        using NativeBuffer a = new((nuint)(nb * sizeof(BlockQ8K)));
        Random r = new(115);
        BlockSTQ1_0* wp = (BlockSTQ1_0*)w.Pointer;
        for (int b = 0; b < nb; b++)
        {
            wp[b].D = HalfBits.FromSingle(0.01f + b * 0.001f);
            for (int j = 0; j < Qk.STQ1_0BlockLength / 8; j++)
                wp[b].Qs[j] = (byte)r.Next(256);
            for (int j = 0; j < Qk.STQ1_0BlockLength / 32; j++)
                wp[b].Sign[j] = (byte)r.Next(256);
        }

        FillQ8K((BlockQ8K*)a.Pointer, nb, 116);
        float scalar = STQ1_0.DotScalar(wp, (BlockQ8K*)a.Pointer, n);
        float vec = STQ1_0.DotVec(wp, (BlockQ8K*)a.Pointer, n);
        Assert.True(MathF.Abs(scalar - vec) < 1e-3f * n, $"scalar {scalar} vs vec {vec}");
    }

    [Fact]
    public unsafe void Q8K_QuantizeRow_ProducesValidBlocks()
    {
        const int n = Qk.SuperBlock * 2;
        using NativeBuffer src = new((nuint)(n * sizeof(float)));
        using NativeBuffer y = new((nuint)Q8K.RowBytes(n));
        Random r = new(117);
        float* sp = (float*)src.Pointer;
        for (int i = 0; i < n; i++)
            sp[i] = (float)(r.NextDouble() * 6 - 3);
        sp[3] = -5.5f; // first signed extremum path
        sp[200] = 5.5f;

        Q8K.QuantizeRow(sp, (BlockQ8K*)y.Pointer, n);
        BlockQ8K* yp = (BlockQ8K*)y.Pointer;
        for (int b = 0; b < n / Qk.SuperBlock; b++)
        {
            Assert.True(yp[b].D != 0);
            for (int j = 0; j < Qk.SuperBlock; j++)
                Assert.InRange((int)yp[b].Qs[j], -127, 127);
            for (int g = 0; g < Qk.SuperBlock / 16; g++)
            {
                int s = 0;
                for (int i = 0; i < 16; i++)
                    s += yp[b].Qs[g * 16 + i];
                Assert.Equal((short)s, yp[b].Bsums[g]);
            }

            // round-trip sanity: dequantized block approximates the input.
            float worst = 0;
            for (int j = 0; j < Qk.SuperBlock; j++)
                worst = MathF.Max(worst, MathF.Abs(yp[b].Qs[j] * yp[b].D - sp[b * Qk.SuperBlock + j]));
            Assert.True(worst < 0.1f, $"block {b} worst={worst}");
        }
    }

    [Fact]
    public unsafe void Q4K_RowGemm_MatchesPerRowDots()
    {
        const int nIn = Qk.SuperBlock * 2;
        const int nOut = 10; // exercises the non-multiple-of-8 tail too
        const int tokens = 4;
        float[] w = RandomRow(nOut * nIn, 118);
        float[] a = RandomRow(tokens * nIn, 119);
        using NativeBuffer q4 = new((nuint)((long)nOut * (nIn / Qk.SuperBlock) * Qk.Q4KSize));
        using NativeBuffer got = new((nuint)((long)tokens * nOut * sizeof(float)));
        using NativeBuffer dw = new((nuint)(nIn * sizeof(float)));
        fixed (float* wp = w, ap = a)
        {
            BlockQ4K* rows = (BlockQ4K*)q4.Pointer;
            int nb = nIn / Qk.SuperBlock;
            for (int r2 = 0; r2 < nOut; r2++)
                Q4K.PackSimple(wp + r2 * nIn, rows + r2 * nb, nIn);

            // packed == null exercises the portable row-GEMM path on every host.
            MulMatQ4K.Gemm(null, rows, ap, (float*)got.Pointer, nIn, nOut, tokens);
            // The portable GEMM dequantizes weights to f32 and consumes raw
            // floats, so the reference is the exact dequant·input dot rather
            // than the Q8_K-quantized dot.
            for (int t = 0; t < tokens; t++)
            {
                for (int r2 = 0; r2 < nOut; r2++)
                {
                    float expected = 0;
                    Q4K.DequantizeRow(rows + r2 * nb, (float*)dw.Pointer, nIn);
                    float* dwp = (float*)dw.Pointer;
                    for (int i = 0; i < nIn; i++)
                        expected += dwp[i] * ap[t * nIn + i];
                    Assert.True(MathF.Abs(expected - ((float*)got.Pointer)[t * nOut + r2]) < 1e-3f,
                        $"t={t} r={r2} exp={expected} got={((float*)got.Pointer)[t * nOut + r2]}");
                }
            }
        }
    }

    private static unsafe void FillQ4K(BlockQ4K* w, int nb, int seed)
    {
        Random r = new(seed);
        for (int b = 0; b < nb; b++)
        {
            w[b].D = HalfBits.FromSingle(0.01f + b * 0.001f);
            w[b].Dmin = HalfBits.FromSingle(0.002f);
            for (int j = 0; j < Qk.ScaleBytes; j++)
                w[b].Scales[j] = (byte)r.Next(256);
            for (int j = 0; j < Qk.SuperBlock / 2; j++)
                w[b].Qs[j] = (byte)r.Next(256);
        }
    }

    private static unsafe void FillQ8K(BlockQ8K* y, int nb, int seed)
    {
        Random r = new(seed);
        for (int b = 0; b < nb; b++)
        {
            y[b].D = 0.02f + b * 0.001f;
            int bs = 0;
            for (int j = 0; j < Qk.SuperBlock; j++)
            {
                y[b].Qs[j] = (sbyte)r.Next(-127, 128);
                bs += y[b].Qs[j];
                if ((j & 15) == 15)
                {
                    y[b].Bsums[j / 16] = (short)bs;
                    bs = 0;
                }
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
}
