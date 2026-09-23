using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using Sdcb.HyMT2Sharp.Kernels;
using Sdcb.HyMT2Sharp.Model;

namespace Sdcb.HyMT2Sharp.Tests;

public sealed class FastExpTests
{
    [Fact]
    public void ExpAvx2_TracksMathF()
    {
        if (!Avx2.IsSupported || !Fma.IsSupported)
            return;
        float[] x = new float[8];
        float[] exp = new float[8];
        for (int i = 0; i < 8; i++)
            x[i] = (i - 3) * 1.25f;
        Vector256<float> got = FastExp.ExpAvx2(Vector256.Create(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]));
        for (int i = 0; i < 8; i++)
        {
            exp[i] = MathF.Exp(x[i]);
            float g = got.GetElement(i);
            float scale = MathF.Max(1e-5f, MathF.Abs(exp[i]));
            Assert.True(MathF.Abs(g - exp[i]) / scale < 2e-5f, $"x={x[i]} got={g} exp={exp[i]}");
        }
    }

    [Fact]
    public unsafe void SiLUMul_TracksScalar()
    {
        const int n = 64;
        float[] gate = new float[n];
        float[] up = new float[n];
        float[] expected = new float[n];
        for (int i = 0; i < n; i++)
        {
            gate[i] = (i - 32) * 0.15f;
            up[i] = 0.75f + i * 0.01f;
            expected[i] = gate[i] / (1f + MathF.Exp(-gate[i])) * up[i];
        }

        fixed (float* g = gate, u = up)
            Ops.SiLUMulRange(g, u, 0, n);

        for (int i = 0; i < n; i++)
        {
            float scale = MathF.Max(1e-4f, MathF.Abs(expected[i]));
            Assert.True(MathF.Abs(gate[i] - expected[i]) / scale < 2e-4f, $"i={i} got={gate[i]} exp={expected[i]}");
        }
    }

    [Fact]
    public unsafe void Quantize4x8Silu_MatchesSiLUThenQuantize()
    {
        const int k = 256;
        float[] gate = new float[4 * k];
        float[] up = new float[4 * k];
        float[] afterSilu = new float[4 * k];
        for (int i = 0; i < gate.Length; i++)
        {
            gate[i] = (i % 17 - 8) * 0.2f;
            up[i] = 0.5f + (i % 9) * 0.05f;
            afterSilu[i] = gate[i];
        }

        using NativeBuffer expected = new((nuint)Qk.Q8Kx4Size);
        using NativeBuffer got = new((nuint)Qk.Q8Kx4Size);
        fixed (float* g = gate, u = up, silu = afterSilu)
        {
            Ops.SiLUMulRange(silu, u, 0, 4 * k);
            QuantizeQ8Kx4.Quantize4x8(silu, (BlockQ8Kx4*)expected.Pointer, k);
            QuantizeQ8Kx4.Quantize4x8Silu(g, u, (BlockQ8Kx4*)got.Pointer, k);
        }

        BlockQ8Kx4* e = (BlockQ8Kx4*)expected.Pointer;
        BlockQ8Kx4* a = (BlockQ8Kx4*)got.Pointer;
        for (int r = 0; r < 4; r++)
            Assert.True(MathF.Abs(e->D[r] - a->D[r]) < 1e-4f, $"d[{r}] {e->D[r]} vs {a->D[r]}");
        for (int i = 0; i < 4 * k; i++)
            Assert.Equal(e->Qs[i], a->Qs[i]);
    }

    [Fact]
    public void ThreadPool_ForSplitsWork()
    {
        int[] hits = new int[32];
        using CpuThreadPool pool = new(8);
        pool.For(32, (int worker, int workers) =>
        {
            int begin = 32 * worker / workers;
            int end = 32 * (worker + 1) / workers;
            for (int i = begin; i < end; i++)
                Interlocked.Increment(ref hits[i]);
        });

        for (int i = 0; i < hits.Length; i++)
            Assert.Equal(1, hits[i]);
    }
}
