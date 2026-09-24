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

    [Fact]
    public void HalfBits_ToSingle_MatchesBclForEveryBitPattern()
    {
        for (int bits = 0; bits <= ushort.MaxValue; bits++)
        {
            float expected = (float)BitConverter.UInt16BitsToHalf((ushort)bits);
            float actual = HalfBits.ToSingle((ushort)bits);
            Assert.True(BitConverter.SingleToUInt32Bits(expected) == BitConverter.SingleToUInt32Bits(actual) || (float.IsNaN(expected) && float.IsNaN(actual)),
                $"0x{bits:X4}: {actual} vs {expected}");
        }
    }

    [Fact]
    public unsafe void Q4K_DequantizeBlockVec_MatchesScalar()
    {
        const int nb = 3;
        using NativeBuffer w = new((nuint)(nb * sizeof(BlockQ4K)));
        using NativeBuffer a = new((nuint)(nb * Qk.SuperBlock * sizeof(float)));
        using NativeBuffer b = new((nuint)(Qk.SuperBlock * sizeof(float)));
        FillQ4K((BlockQ4K*)w.Pointer, nb, 120);
        Q4K.DequantizeRow((BlockQ4K*)w.Pointer, (float*)a.Pointer, nb * Qk.SuperBlock);
        for (int i = 0; i < nb; i++)
        {
            Q4K.DequantizeBlockVec((BlockQ4K*)w.Pointer + i, (float*)b.Pointer);
            for (int j = 0; j < Qk.SuperBlock; j++)
            {
                float e = ((float*)a.Pointer)[i * Qk.SuperBlock + j];
                float g = ((float*)b.Pointer)[j];
                Assert.True(MathF.Abs(e - g) <= 1e-6f * MathF.Max(1f, MathF.Abs(e)), $"block {i} [{j}] {g} vs {e}");
            }
        }
    }

    [Fact]
    public unsafe void Q6K_DequantizeBlockVec_MatchesScalar()
    {
        const int nb = 3;
        using NativeBuffer w = new((nuint)(nb * sizeof(BlockQ6K)));
        using NativeBuffer a = new((nuint)(nb * Qk.SuperBlock * sizeof(float)));
        using NativeBuffer b = new((nuint)(Qk.SuperBlock * sizeof(float)));
        FillQ6K((BlockQ6K*)w.Pointer, nb, 121);
        Q6K.DequantizeRow((BlockQ6K*)w.Pointer, (float*)a.Pointer, nb * Qk.SuperBlock);
        for (int i = 0; i < nb; i++)
        {
            Q6K.DequantizeBlockVec((BlockQ6K*)w.Pointer + i, (float*)b.Pointer);
            for (int j = 0; j < Qk.SuperBlock; j++)
                Assert.Equal(((float*)a.Pointer)[i * Qk.SuperBlock + j], ((float*)b.Pointer)[j]);
        }
    }

    /// <summary>Extreme activations (±127, and the −128 the i16 bound must still tolerate) and max nibbles/6-bit codes.</summary>
    [Fact]
    public unsafe void DotAct_ExtremeValues_MatchScalar()
    {
        const int nb = 2;
        const int n = nb * Qk.SuperBlock;
        using NativeBuffer q4 = new((nuint)(nb * sizeof(BlockQ4K)));
        using NativeBuffer q6 = new((nuint)(nb * sizeof(BlockQ6K)));
        using NativeBuffer y = new((nuint)(nb * sizeof(BlockQ8K)));
        BlockQ4K* w4 = (BlockQ4K*)q4.Pointer;
        BlockQ6K* w6 = (BlockQ6K*)q6.Pointer;
        BlockQ8K* a = (BlockQ8K*)y.Pointer;
        for (int b = 0; b < nb; b++)
        {
            w4[b].D = HalfBits.FromSingle(0.5f);
            w4[b].Dmin = HalfBits.FromSingle(0.25f);
            w6[b].D = HalfBits.FromSingle(0.5f);
            for (int j = 0; j < Qk.ScaleBytes; j++)
                w4[b].Scales[j] = 0xFF;
            for (int j = 0; j < Qk.SuperBlock / 2; j++)
            {
                w4[b].Qs[j] = 0xFF;
                w6[b].Ql[j] = (byte)(b == 0 ? 0xFF : 0x00);
            }
            for (int j = 0; j < Qk.SuperBlock / 4; j++)
                w6[b].Qh[j] = (byte)(b == 0 ? 0xFF : 0x00);
            for (int j = 0; j < Qk.SuperBlock / 16; j++)
                w6[b].Scales[j] = (sbyte)(j % 2 == 0 ? 127 : -128);

            a[b].D = 0.01f;
            int bs = 0;
            for (int j = 0; j < Qk.SuperBlock; j++)
            {
                a[b].Qs[j] = (sbyte)(b == 0 ? 127 : -128);
                bs += a[b].Qs[j];
                if ((j & 15) == 15)
                {
                    a[b].Bsums[j / 16] = (short)bs;
                    bs = 0;
                }
            }
        }

        float s4 = VecDotQ4K.DotScalar(w4, a, n);
        float v4 = VecDotQ4K.DotVec(w4, a, n);
        Assert.True(MathF.Abs(s4 - v4) <= 1e-4f * MathF.Abs(s4), $"q4 scalar {s4} vs vec {v4}");
        float s6 = Q6K.DotScalar(w6, a, n);
        float v6 = Q6K.DotVec(w6, a, n);
        Assert.True(MathF.Abs(s6 - v6) <= 1e-4f * MathF.Abs(s6), $"q6 scalar {s6} vs vec {v6}");
    }

    [Fact]
    public unsafe void KQuantGemv_MixedTargets_MatchPerRowScalar()
    {
        const int nIn = Qk.SuperBlock * 2;
        const int nb = nIn / Qk.SuperBlock;
        const int n0 = 13, n1 = 7, n2 = 21;
        using NativeBuffer w0 = new((nuint)(n0 * nb * sizeof(BlockQ4K)));
        using NativeBuffer w1 = new((nuint)(n1 * nb * sizeof(BlockQ4K)));
        using NativeBuffer w2 = new((nuint)(n2 * nb * sizeof(BlockQ6K)));
        using NativeBuffer y = new((nuint)(nb * sizeof(BlockQ8K)));
        using NativeBuffer d = new((nuint)((n0 + n1 + n2) * sizeof(float)));
        FillQ4K((BlockQ4K*)w0.Pointer, n0 * nb, 122);
        FillQ4K((BlockQ4K*)w1.Pointer, n1 * nb, 123);
        FillQ6K((BlockQ6K*)w2.Pointer, n2 * nb, 124);
        FillQ8K((BlockQ8K*)y.Pointer, nb, 125);
        float* d0 = (float*)d.Pointer;
        float* d1 = d0 + n0;
        float* d2 = d1 + n1;
        using CpuThreadPool pool = new(4);
        KQuantGemv.Multi((BlockQ8K*)y.Pointer, nIn, pool,
            new KGemvTarget((BlockQ4K*)w0.Pointer, null, d0, n0),
            new KGemvTarget((BlockQ4K*)w1.Pointer, null, d1, n1),
            new KGemvTarget(null, (BlockQ6K*)w2.Pointer, d2, n2));

        for (int r = 0; r < n0; r++)
            AssertClose(VecDotQ4K.DotScalar((BlockQ4K*)w0.Pointer + r * nb, (BlockQ8K*)y.Pointer, nIn), d0[r], $"q {r}");
        for (int r = 0; r < n1; r++)
            AssertClose(VecDotQ4K.DotScalar((BlockQ4K*)w1.Pointer + r * nb, (BlockQ8K*)y.Pointer, nIn), d1[r], $"k {r}");
        for (int r = 0; r < n2; r++)
            AssertClose(Q6K.DotScalar((BlockQ6K*)w2.Pointer + r * nb, (BlockQ8K*)y.Pointer, nIn), d2[r], $"v {r}");

        static void AssertClose(float e, float g, string what) =>
            Assert.True(MathF.Abs(e - g) <= 1e-3f * MathF.Max(1f, MathF.Abs(e)), $"{what}: {g} vs {e}");
    }

    /// <summary>
    /// Ragged shapes for the packed outer-product GEMM: token count not a multiple of the
    /// 6-token tile, rows not a multiple of the 2V-row panel, worker ranges with a partial panel group.
    /// </summary>
    [Theory]
    [InlineData(2, 1)]
    [InlineData(13, 37)]
    [InlineData(64, 100)]
    public unsafe void Q4K_RowGemm_RaggedShapes_MatchDequantDot(int tokens, int nOut)
    {
        const int nIn = Qk.SuperBlock * 2;
        const int nb = nIn / Qk.SuperBlock;
        float[] a = RandomRow(tokens * nIn, 126);
        using NativeBuffer q4 = new((nuint)((long)nOut * nb * Qk.Q4KSize));
        using NativeBuffer got = new((nuint)((long)tokens * nOut * sizeof(float)));
        using NativeBuffer dw = new((nuint)(nOut * nIn * sizeof(float)));
        FillQ4K((BlockQ4K*)q4.Pointer, nOut * nb, 127);
        BlockQ4K* rows = (BlockQ4K*)q4.Pointer;
        float* dwp = (float*)dw.Pointer;
        for (int r = 0; r < nOut; r++)
            Q4K.DequantizeRow(rows + r * nb, dwp + r * nIn, nIn);
        using CpuThreadPool pool = new(3);
        fixed (float* ap = a)
        {
            // tokens > 1 with packed == null takes the portable VecGemmF path on every host.
            MulMatQ4K.Gemm(null, rows, ap, (float*)got.Pointer, nIn, nOut, tokens, pool);
            for (int t = 0; t < tokens; t++)
            {
                for (int r = 0; r < nOut; r++)
                {
                    double expected = 0;
                    for (int i = 0; i < nIn; i++)
                        expected += dwp[r * nIn + i] * ap[t * nIn + i];
                    float g = ((float*)got.Pointer)[t * nOut + r];
                    Assert.True(Math.Abs(expected - g) < 1e-3 * Math.Max(1, Math.Abs(expected)), $"t={t} r={r} exp={expected} got={g}");
                }
            }
        }
    }

    private static unsafe void FillQ6K(BlockQ6K* w, int nb, int seed)
    {
        Random r = new(seed);
        for (int b = 0; b < nb; b++)
        {
            w[b].D = HalfBits.FromSingle(0.01f + (b % 7) * 0.001f);
            for (int j = 0; j < Qk.SuperBlock / 2; j++)
                w[b].Ql[j] = (byte)r.Next(256);
            for (int j = 0; j < Qk.SuperBlock / 4; j++)
                w[b].Qh[j] = (byte)r.Next(256);
            for (int j = 0; j < Qk.SuperBlock / 16; j++)
                w[b].Scales[j] = (sbyte)r.Next(-128, 128);
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
