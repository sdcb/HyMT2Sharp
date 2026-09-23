using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using Sdcb.HyMT2Sharp.Kernels;

namespace Sdcb.HyMT2Sharp.Tests;

public sealed class Q6KPanelTests
{
    [Fact]
    public unsafe void PanelGemm_MatchesPerRowVecDot()
    {
        const int nIn = 256 * 2;
        const int nOut = 16;
        const int tokens = 4;
        Random rng = new(17);
        using NativeBuffer q6 = new((nuint)((long)nOut * (nIn / Qk.SuperBlock) * Qk.Q6KSize));
        using NativeBuffer q6x8 = new((nuint)((long)(nOut / 8) * (nIn / Qk.SuperBlock) * Qk.Q6Kx8Size));
        using NativeBuffer q8x4 = new((nuint)((nIn / Qk.SuperBlock) * Qk.Q8Kx4Size));
        using NativeBuffer q8 = new((nuint)Q8K.RowBytes(nIn));
        using NativeBuffer expected = new((nuint)((long)tokens * nOut * sizeof(float)));
        using NativeBuffer got = new((nuint)((long)tokens * nOut * sizeof(float)));
        byte* raw = (byte*)q6.Pointer;
        for (int i = 0; i < (int)q6.Bytes; i++)
            raw[i] = (byte)rng.Next(256);
        BlockQ6K* rows = (BlockQ6K*)q6.Pointer;
        int nb = nIn / Qk.SuperBlock;
        float[] a = RandomRow(tokens * nIn, 18);
        fixed (float* ap = a)
        {
            RepackQ6K.Rows(rows, (BlockQ6Kx8*)q6x8.Pointer, nIn, nOut);
            QuantizeQ8Kx4.Quantize4x8Scalar(ap, (BlockQ8Kx4*)q8x4.Pointer, nIn);

            float* exp = (float*)expected.Pointer;
            for (int t = 0; t < tokens; t++)
            {
                Q8K.QuantizeRow(ap + t * nIn, (BlockQ8K*)q8.Pointer, nIn);
                for (int r = 0; r < nOut; r++)
                    exp[t * nOut + r] = Q6K.DotScalar(rows + r * nb, (BlockQ8K*)q8.Pointer, nIn);
            }

            GemmQ6K.GemmScalar(nIn, (float*)got.Pointer, nOut, (BlockQ6Kx8*)q6x8.Pointer, (BlockQ8Kx4*)q8x4.Pointer, tokens, nOut);
            AssertClose((float*)expected.Pointer, (float*)got.Pointer, tokens * nOut, 3e-2f);

            if (System.Runtime.Intrinsics.X86.Avx2.IsSupported)
            {
                NativeMemory.Clear(got.Pointer, got.Bytes);
                GemmQ6K.GemmAvx2(nIn, (float*)got.Pointer, nOut, (BlockQ6Kx8*)q6x8.Pointer, (BlockQ8Kx4*)q8x4.Pointer, tokens, nOut);
                AssertClose((float*)expected.Pointer, (float*)got.Pointer, tokens * nOut, 3e-2f);
            }
        }
    }

    [Fact]
    public unsafe void Quantize4x8Avx2_MatchesScalar()
    {
        if (!System.Runtime.Intrinsics.X86.Avx2.IsSupported)
            return;

        const int nIn = 256 * 3;
        const int tokens = 4;
        float[] a = RandomRow(tokens * nIn, 19);
        using NativeBuffer s = new((nuint)((nIn / Qk.SuperBlock) * Qk.Q8Kx4Size));
        using NativeBuffer v = new((nuint)((nIn / Qk.SuperBlock) * Qk.Q8Kx4Size));
        fixed (float* ap = a)
        {
            QuantizeQ8Kx4.Quantize4x8Scalar(ap, (BlockQ8Kx4*)s.Pointer, nIn);
            QuantizeQ8Kx4.Quantize4x8Avx2(ap, (BlockQ8Kx4*)v.Pointer, nIn);
            BlockQ8Kx4* a0 = (BlockQ8Kx4*)s.Pointer;
            BlockQ8Kx4* a1 = (BlockQ8Kx4*)v.Pointer;
            int nb = nIn / Qk.SuperBlock;
            for (int i = 0; i < nb; i++)
            {
                for (int r = 0; r < 4; r++)
                    Assert.True(MathF.Abs(a0[i].D[r] - a1[i].D[r]) < 1e-5f, $"d[{i},{r}] {a0[i].D[r]} vs {a1[i].D[r]}");
                for (int j = 0; j < Qk.SuperBlock * 4; j++)
                    Assert.True(a0[i].Qs[j] == a1[i].Qs[j], $"qs[{i},{j}] {a0[i].Qs[j]} vs {a1[i].Qs[j]}");
                for (int j = 0; j < Qk.SuperBlock / 4; j++)
                    Assert.True(a0[i].Bsums[j] == a1[i].Bsums[j], $"bsums[{i},{j}] {a0[i].Bsums[j]} vs {a1[i].Bsums[j]}");
            }
        }
    }

    [Fact]
    public unsafe void ScaleProducts_Avx2MatchesVnni()
    {
        if (!Avx2.IsSupported || !AvxVnni.IsSupported)
            return;

        byte* q = stackalloc byte[32];
        sbyte* a = stackalloc sbyte[32];
        Random rng = new(6);
        for (int i = 0; i < 32; i++)
        {
            q[i] = (byte)rng.Next(64);
            a[i] = (sbyte)rng.Next(-127, 128);
        }

        Vector256<short> sc = Vector256.Create((short)3, 3, 3, 3, 3, 3, 3, 3, -5, -5, -5, -5, -5, -5, -5, -5);
        Vector256<int> avx = Q6K.ScaleQ6Avx2(Avx.LoadVector256(q), Avx.LoadVector256(a), sc);
        Vector256<int> vnni = Q6K.ScaleQ6Vnni(Avx.LoadVector256(q), Avx.LoadVector256(a), sc);
        for (int i = 0; i < 8; i++)
            Assert.Equal(avx.GetElement(i), vnni.GetElement(i));
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
        float maxRel = 0;
        float maxAbs = 0;
        for (int i = 0; i < n; i++)
        {
            float d = MathF.Abs(a[i] - b[i]);
            if (d > maxAbs)
                maxAbs = d;
            float scale = MathF.Max(1e-3f, MathF.Max(MathF.Abs(a[i]), MathF.Abs(b[i])));
            float r = d / scale;
            if (r > maxRel)
                maxRel = r;
        }

        Assert.True(maxRel < rel, $"maxRel={maxRel} maxAbs={maxAbs}");
    }
}
