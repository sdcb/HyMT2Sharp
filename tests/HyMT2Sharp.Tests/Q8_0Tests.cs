using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using Sdcb.HyMT2Sharp.Kernels;

namespace Sdcb.HyMT2Sharp.Tests;

public sealed class Q8_0Tests
{
    [Fact]
    public void BlockSizes_MatchGgufLayout()
    {
        Assert.Equal(34, Unsafe.SizeOf<BlockQ8_0>());
        Assert.Equal(Qk.Q8_0x8Size, Unsafe.SizeOf<BlockQ8_0x8>());
        Assert.Equal(Qk.Q8_0x4Size, Unsafe.SizeOf<BlockQ8_0x4>());
        Assert.Equal(Qk.Q8_0ActSize, Unsafe.SizeOf<BlockQ8_0Act>());
    }

    [Fact]
    public unsafe void Dequantize_MatchesScaleTimesQuant()
    {
        const int n = 64;
        using NativeBuffer raw = new((nuint)(2 * Qk.Q8_0Size));
        using NativeBuffer dst = new((nuint)(n * sizeof(float)));
        BlockQ8_0* blocks = (BlockQ8_0*)raw.Pointer;
        blocks[0].D = HalfBits.FromSingle(0.25f);
        blocks[1].D = HalfBits.FromSingle(-0.5f);
        for (int i = 0; i < 32; i++)
        {
            blocks[0].Qs[i] = (sbyte)(i - 16);
            blocks[1].Qs[i] = (sbyte)(40 - i);
        }

        blocks[0].Qs[3] = -128;
        Q8_0.DequantizeRow(blocks, (float*)dst.Pointer, n);
        float* y = (float*)dst.Pointer;
        Assert.Equal(0.25f * -128, y[3], 1e-5f);
        Assert.Equal(-0.5f * blocks[1].Qs[7], y[32 + 7], 1e-4f);
    }

    [Fact]
    public unsafe void Dot32_MatchesScalar_IncludingNegative128()
    {
        if (!Avx2.IsSupported)
            return;

        sbyte* w = stackalloc sbyte[32];
        sbyte* a = stackalloc sbyte[32];
        int sum = 0;
        for (int i = 0; i < 32; i++)
        {
            w[i] = (sbyte)(i == 0 ? -128 : i * 7 - 40);
            a[i] = (sbyte)(i * 3 - 50);
            sum += a[i];
        }

        int expected = 0;
        for (int i = 0; i < 32; i++)
            expected += w[i] * a[i];
        Assert.Equal(expected, Q8_0.Dot32Avx2(w, a));
        if (AvxVnni.IsSupported)
            Assert.Equal(expected, Q8_0.Dot32Vnni(w, a, sum));
    }

    [Fact]
    public unsafe void PanelGemm_MatchesScalar()
    {
        const int nIn = 64;
        const int nOut = 16;
        const int tokens = 8;
        int nb = nIn / Qk.Q8_0Block;
        using NativeBuffer rows = new((nuint)((long)nOut * nb * Qk.Q8_0Size));
        using NativeBuffer panel = new((nuint)((long)(nOut / 8) * nb * Qk.Q8_0x8Size));
        using NativeBuffer acts = new((nuint)((long)(tokens / 4) * nb * Qk.Q8_0x4Size));
        using NativeBuffer rowAct = new((nuint)((long)tokens * nb * Qk.Q8_0ActSize));
        using NativeBuffer got = new((nuint)((long)tokens * nOut * sizeof(float)));
        using NativeBuffer expect = new((nuint)((long)tokens * nOut * sizeof(float)));
        FillBlocks((BlockQ8_0*)rows.Pointer, nOut * nb, 11);
        float[] input = new float[tokens * nIn];
        Random rng = new(12);
        for (int i = 0; i < input.Length; i++)
            input[i] = (float)(rng.NextDouble() * 4 - 2);
        fixed (float* ip = input)
        {
            RepackQ8_0.Rows((BlockQ8_0*)rows.Pointer, (BlockQ8_0x8*)panel.Pointer, nIn, nOut);
            for (int g = 0; g < tokens / 4; g++)
                Q8_0.Quantize4(ip + g * 4 * nIn, (BlockQ8_0x4*)acts.Pointer + g * nb, nIn);
            for (int t = 0; t < tokens; t++)
                Q8_0.QuantizeActs(ip + t * nIn, (BlockQ8_0Act*)rowAct.Pointer + t * nb, nIn);

            float* exp = (float*)expect.Pointer;
            for (int t = 0; t < tokens; t++)
            {
                for (int r = 0; r < nOut; r++)
                    exp[t * nOut + r] = Q8_0.DotScalar((BlockQ8_0*)rows.Pointer + r * nb, (BlockQ8_0Act*)rowAct.Pointer + t * nb, nIn);
            }

            GemmQ8_0.GemmScalar(nIn, (float*)got.Pointer, nOut, (BlockQ8_0x8*)panel.Pointer, (BlockQ8_0x4*)acts.Pointer, tokens, nOut);
            AssertClose(exp, (float*)got.Pointer, tokens * nOut, 1e-4f);

            if (Avx2.IsSupported)
            {
                NativeMemory.Clear(got.Pointer, got.Bytes);
                GemmQ8_0.GemmAvx2(nIn, (float*)got.Pointer, nOut, (BlockQ8_0x8*)panel.Pointer, (BlockQ8_0x4*)acts.Pointer, tokens, nOut);
                AssertClose(exp, (float*)got.Pointer, tokens * nOut, 1e-3f);
            }

            if (AvxVnni.IsSupported)
            {
                NativeMemory.Clear(got.Pointer, got.Bytes);
                GemmQ8_0.GemmVnni(nIn, (float*)got.Pointer, nOut, (BlockQ8_0x8*)panel.Pointer, (BlockQ8_0x4*)acts.Pointer, tokens, nOut);
                AssertClose(exp, (float*)got.Pointer, tokens * nOut, 1e-3f);
            }
        }
    }

    [Fact]
    public unsafe void Dot4_Avx2MatchesVnni()
    {
        if (!Avx2.IsSupported || !AvxVnni.IsSupported)
            return;

        byte* w = stackalloc byte[32];
        int act = 0;
        sbyte* ap = (sbyte*)&act;
        Random rng = new(4);
        for (int i = 0; i < 32; i++)
            w[i] = (byte)rng.Next(256);
        ap[0] = -127;
        ap[1] = 127;
        ap[2] = -1;
        ap[3] = 3;
        Vector256<byte> wv = Avx.LoadVector256(w);
        Vector256<int> a = GemmQ8_0.Dot4Avx2(wv, &act);
        Vector256<int> b = GemmQ8_0.Dot4Vnni(wv, &act);
        for (int lane = 0; lane < 8; lane++)
        {
            int scalar = 0;
            for (int t = 0; t < 4; t++)
                scalar += (sbyte)w[lane * 4 + t] * ap[t];
            Assert.True(a.GetElement(lane) == scalar && b.GetElement(lane) == scalar,
                $"lane {lane} scalar={scalar} avx2={a.GetElement(lane)} vnni={b.GetElement(lane)} w={w[lane * 4]},{w[lane * 4 + 1]},{w[lane * 4 + 2]},{w[lane * 4 + 3]} a={ap[0]},{ap[1]},{ap[2]},{ap[3]}");
        }
    }

    private static unsafe void FillBlocks(BlockQ8_0* blocks, int count, int seed)
    {
        Random rng = new(seed);
        for (int i = 0; i < count; i++)
        {
            blocks[i].D = HalfBits.FromSingle((float)(rng.NextDouble() * 0.2 + 0.01));
            for (int k = 0; k < 32; k++)
                blocks[i].Qs[k] = (sbyte)rng.Next(-128, 128);
        }
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
